import os, re, math, glob, json
import numpy as np, pandas as pd
from collections import Counter

PROCESSED = "/root/feature_cache/processed/*.npz"
ASSIGN    = "/root/analysis/frame_cluster_assignments_v2.csv"
OUT_VID   = "/root/analysis/video_gloss_suggestions.csv"
OUT_CLU   = "/root/analysis/cluster_gloss_suggestions.csv"

def vstem(stem): return re.sub(r"_\d+$", "", stem)

def load_npz_map():
    m = {}
    for p in glob.glob(PROCESSED):
        st = os.path.splitext(os.path.basename(p))[0]
        m[st] = p
    return m

def face_bbox(face_kpts):
    if face_kpts.size == 0: return None
    xs, ys = face_kpts[:,0], face_kpts[:,1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())

def regions_from_face(face_kpts):
    bb = face_bbox(face_kpts)
    if bb is None:
        return {"face_c": np.array([0.5,0.4]), "mouth_c": np.array([0.5,0.55]),
                "forehead_c": np.array([0.5,0.20]), "chest_c": np.array([0.5,0.75]),
                "face_w": 0.25, "face_h": 0.25}
    x0,y0,x1,y1 = bb; w,h = x1-x0, y1-y0
    return {"face_c": np.array([x0+w*0.5, y0+h*0.45]),
            "mouth_c": np.array([x0+w*0.5, y0+h*0.75]),
            "forehead_c": np.array([x0+w*0.5, y0+h*0.20]),
            "chest_c": np.array([x0+w*0.5, y1 + h*0.35]),
            "face_w": w, "face_h": h}

def main_hand_point(hands):
    if hands.size == 0: return None
    scores = [(np.isfinite(hands[i]).sum(), i) for i in range(hands.shape[0])]
    idx = max(scores)[1]
    return np.nanmean(hands[idx], axis=0)

def smooth(x, w=3):
    if len(x) < 3: return np.array(x, float)
    k = min(w, len(x))
    return np.convolve(x, np.ones(k)/k, mode="same")

def osc_score(x, min_amp=0.03):
    x = smooth(np.array(x, float), 3)
    if len(x) < 5: return 0.0
    m = x - np.mean(x)
    zc = np.where(np.sign(m[:-1]) * np.sign(m[1:]) < 0)[0]
    amp = (np.max(m) - np.min(m))
    return float(len(zc)) * float(amp > min_amp)

def radial_outward_score(hand_xy, face_c):
    if len(hand_xy) < 2: return 0.0
    d = [np.linalg.norm(p - face_c) for p in hand_xy]
    d0 = np.mean(d[:max(1,len(d)//5)]); d1 = np.mean(d[-max(1,len(d)//5):])
    return max(0.0, (d1 - d0))

def near_score(p, c, r): return float(np.linalg.norm(p - c) < r)

def circle_score(hand_xy, center, r_tol=0.15):
    if len(hand_xy) < 5: return 0.0
    v = hand_xy - center
    r = np.linalg.norm(v, axis=1) + 1e-6
    ang = np.unwrap(np.arctan2(v[:,1], v[:,0]))
    total = abs(ang[-1] - ang[0])
    rcv = np.std(r) / max(np.mean(r), 1e-6)
    return max(0.0, total / math.pi) * float(rcv < r_tol)

def analyze_video(frame_npzs):
    hand_xy, face_xy_c = [], []
    mouth_c = forehead_c = chest_c = None
    for p in frame_npzs:
        d = np.load(p, allow_pickle=True)
        hk, fk = np.array(d.get("hand_keypoints", [])), np.array(d.get("face_keypoints", []))
        if fk.size==0: continue
        regs = regions_from_face(fk.squeeze())
        face_xy_c.append(regs["face_c"])
        mh = main_hand_point(hk)
        if mh is not None: hand_xy.append(mh)
        if mouth_c is None:
            mouth_c, forehead_c, chest_c = regs["mouth_c"], regs["forehead_c"], regs["chest_c"]
    if len(hand_xy) < 3 or len(face_xy_c) < 3: return {}
    hand_xy = np.array(hand_xy); face_c = np.mean(face_xy_c, axis=0)
    s_thank = near_score(hand_xy[0], mouth_c, 0.15) * radial_outward_score(hand_xy, face_c)
    s_hello = near_score(hand_xy[0], forehead_c, 0.18) * osc_score(hand_xy[:,0], 0.05)
    s_yes   = osc_score(np.array(face_xy_c)[:,1], 0.015)
    s_no    = osc_score(np.array(face_xy_c)[:,0], 0.015)
    s_please= near_score(np.mean(hand_xy,axis=0), chest_c, 0.22) * circle_score(hand_xy, chest_c)
    scores = {"THANK_YOU": float(s_thank), "HELLO": float(s_hello),
              "YES": float(s_yes), "NO": float(s_no), "PLEASE": float(s_please)}
    mx = max(scores.values()) if scores else 0.0
    if mx <= 1e-6: return {}
    sorted_s = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best, b = sorted_s[0]; sec = sorted_s[1][1] if len(sorted_s)>1 else 0.0
    conf = float((b - sec) / (b + 1e-6))
    return {"suggested":best, "conf":conf, "scores":scores}

def main():
    m = load_npz_map()
    assign = pd.read_csv(ASSIGN)
    assign["video_stem"] = assign["img_stem"].apply(vstem)
    vids = assign.groupby("video_stem")["img_stem"].apply(list).to_dict()
    rows = []
    for vs, stems in vids.items():
        stems = sorted(stems, key=lambda s:int(re.findall(r"_(\d+)$", s)[0]) if re.findall(r"_(\d+)$", s) else 0)
        npzs = [m.get(s) for s in stems if s in m]
        if not npzs: continue
        r = analyze_video(npzs)
        if not r: continue
        rows.append({"video_stem":vs, "suggested":r["suggested"], "conf":r["conf"], **r["scores"]})
    vdf = pd.DataFrame(rows)
    vdf.to_csv(OUT_VID, index=False)
    vcl = assign[["video_stem","cluster_id_km"]].drop_duplicates()
    vdf = vdf.merge(vcl, on="video_stem", how="left")
    if len(vdf)==0:
        pd.DataFrame(columns=["cluster_id_km","suggested_gloss","vote_count","videos_scored","confidence"]).to_csv(OUT_CLU, index=False)
        print("[cluster] ->", OUT_CLU, "rows: 0"); return
    sug = vdf.groupby("cluster_id_km")["suggested"].agg(lambda s: s.value_counts().idxmax())
    cnt = vdf.groupby("cluster_id_km")["suggested"].agg(lambda s: s.value_counts().max())
    tot = vdf.groupby("cluster_id_km").size()
    cdf = pd.DataFrame({"cluster_id_km":sug.index,
                        "suggested_gloss":sug.values,
                        "vote_count":cnt.values,
                        "videos_scored":tot.values})
    cdf["confidence"] = (cdf["vote_count"] / cdf["videos_scored"].clip(lower=1)).astype(float)
    cdf = cdf.sort_values(["confidence","videos_scored"], ascending=[False,False])
    cdf.to_csv(OUT_CLU, index=False)
    print("[video] ->", OUT_VID, "rows:", len(vdf))
    print("[cluster] ->", OUT_CLU, "rows:", len(cdf))
    print(cdf.head(12).to_string(index=False))

if __name__ == "__main__":
    main()
