import os, re, math, glob
import numpy as np, pandas as pd
from collections import Counter

PROCESSED="/root/feature_cache/processed/*.npz"
ASSIGN   ="/root/analysis/frame_cluster_assignments_v2.csv"
OUT_VID  ="/root/analysis/video_gloss_suggestions.csv"
OUT_CLU  ="/root/analysis/cluster_gloss_suggestions.csv"

def vstem(stem): return re.sub(r"_\d+$","", stem)

def load_npz_map():
    m={}
    for p in glob.glob(PROCESSED):
        st=os.path.splitext(os.path.basename(p))[0]; m[st]=p
    return m

def face_bbox(face_kpts):
    if face_kpts.size==0: return None
    xs,ys=face_kpts[:,0],face_kpts[:,1]
    return float(xs.min()),float(ys.min()),float(xs.max()),float(ys.max())

def regions_from_face(face_kpts):
    bb=face_bbox(face_kpts)
    if bb is None:
        return {"face_c":np.array([0.5,0.4]),"mouth_c":np.array([0.5,0.55]),
                "forehead_c":np.array([0.5,0.22]),"chest_c":np.array([0.5,0.78]),
                "face_w":0.25,"face_h":0.25}
    x0,y0,x1,y1=bb; w,h=x1-x0,y1-y0
    return {"face_c":np.array([x0+w*0.5, y0+h*0.45]),
            "mouth_c":np.array([x0+w*0.5, y0+h*0.72]),
            "forehead_c":np.array([x0+w*0.5, y0+h*0.18]),
            "chest_c":np.array([x0+w*0.5, y1+h*0.35]),
            "face_w":w,"face_h":h}

def main_hand_point(hands):
    if hands.size==0: return None
    idx=max((np.isfinite(hands[i]).sum(),i) for i in range(hands.shape[0]))[1]
    return np.nanmean(hands[idx],axis=0)

def smooth(x, w=3):
    if len(x)<3: return np.array(x,float)
    k=min(w,len(x)); return np.convolve(x,np.ones(k)/k,mode="same")

def osc_score(x, min_amp=0.03):
    x=smooth(np.array(x,float),3)
    if len(x)<6: return 0.0
    m=x-np.mean(x); zc=np.where(np.sign(m[:-1])*np.sign(m[1:])<0)[0]
    amp=(np.max(m)-np.min(m))
    return float(len(zc)) if amp>min_amp else 0.0

def radial_outward(hand_xy, face_c):
    d=[np.linalg.norm(p-face_c) for p in hand_xy]
    if len(d)<4: return 0.0,0.0,0.0
    d0=np.mean(d[:max(2,len(d)//5)]); d1=np.mean(d[-max(2,len(d)//5):])
    # monotonic-ish outward fraction
    inc=sum(d[i+1]>=d[i] for i in range(len(d)-1))/max(1,(len(d)-1))
    return float(d1-d0), float(inc), float(np.std(d)/ (np.mean(d)+1e-6))

def cosine(a,b):
    na=np.linalg.norm(a); nb=np.linalg.norm(b)
    if na<1e-6 or nb<1e-6: return 0.0
    return float(np.dot(a,b)/(na*nb))

def circle_score(hand_xy, center, r_tol=0.25, min_turns=0.7):
    if len(hand_xy)<6: return 0.0
    v=hand_xy-center
    r=np.linalg.norm(v,axis=1)+1e-6
    ang=np.unwrap(np.arctan2(v[:,1],v[:,0]))
    total=abs(ang[-1]-ang[0])/(2*math.pi)  # turns
    rcv=np.std(r)/np.mean(r)
    return float(total>=min_turns and rcv<r_tol)

def analyze_video(npzs):
    hand_xy=[]; face_xy=[]
    mouth_c=forehead_c=chest_c=None
    for p in npzs:
        d=np.load(p,allow_pickle=True)
        hk=np.array(d.get("hand_keypoints", []))
        fk=np.array(d.get("face_keypoints", []))
        if fk.size==0: continue
        regs=regions_from_face(fk.squeeze())
        face_xy.append(regs["face_c"])
        mh=main_hand_point(hk)
        if mh is not None: hand_xy.append(mh)
        if mouth_c is None:
            mouth_c,forehead_c,chest_c=regs["mouth_c"],regs["forehead_c"],regs["chest_c"]
    if len(hand_xy)<4 or len(face_xy)<4: return {}
    hand_xy=np.array(hand_xy); face_c=np.mean(np.array(face_xy),axis=0)

    # THANK_YOU: starts near mouth, strong outward, mostly monotonic outward, outward direction
    start_near_mouth = np.linalg.norm(hand_xy[0]-mouth_c) < 0.12
    delta, inc_frac, d_cv = radial_outward(hand_xy, face_c)
    outward_enough = delta >= 0.07 and inc_frac >= 0.65 and d_cv < 0.35
    dir_cos = cosine(hand_xy[-1]-hand_xy[0], hand_xy[0]-face_c)   # away from face center
    direction_ok = dir_cos <= -0.5
    s_thank = float(start_near_mouth and outward_enough and direction_ok)

    # PLEASE: stays near chest & circular-ish; avoid large radial drift
    chest_near = np.linalg.norm(np.mean(hand_xy,axis=0)-chest_c) < 0.20
    circ = circle_score(hand_xy, chest_c, r_tol=0.30, min_turns=0.6)
    small_drift = radial_outward(hand_xy, chest_c)[0] < 0.05
    s_please = float(chest_near and circ and small_drift)

    # HELLO: hand near forehead initially + lateral waving (x oscillation), y relatively stable
    start_near_forehead = np.linalg.norm(hand_xy[0]-forehead_c) < 0.14
    xosc = osc_score(hand_xy[:,0], 0.05)
    yvar = np.std(hand_xy[:,1])
    s_hello = float(start_near_forehead and xosc>=3 and yvar < 0.08)

    # YES/NO: head nod/shake via face center, only if hand not strongly outward
    face_arr=np.array(face_xy)
    s_yes = float(osc_score(face_arr[:,1], 0.02) >= 2 and not outward_enough)
    s_no  = float(osc_score(face_arr[:,0], 0.02) >= 2 and not outward_enough)

    scores={"THANK_YOU":s_thank,"PLEASE":s_please,"HELLO":s_hello,"YES":s_yes,"NO":s_no}
    if max(scores.values())<=0: return {}
    sorted_s=sorted(scores.items(), key=lambda x:x[1], reverse=True)
    best, b = sorted_s[0]; sec = sorted_s[1][1] if len(sorted_s)>1 else 0.0
    conf = float((b - sec + 1e-6)/(b + 1e-6))  # >~1 when hard pass; fine as a ranking
    return {"suggested":best,"conf":conf,"scores":scores}

def main():
    m=load_npz_map()
    assign=pd.read_csv(ASSIGN)  # img_stem, cluster_id_km, cluster_id_hdb
    assign["video_stem"]=assign["img_stem"].apply(vstem)
    vids=assign.groupby("video_stem")["img_stem"].apply(list).to_dict()

    rows=[]
    for vs, stems in vids.items():
        stems=sorted(stems, key=lambda s:int(re.findall(r"_(\d+)$", s)[0]) if re.findall(r"_(\d+)$", s) else 0)
        npzs=[m.get(s) for s in stems if s in m]
        if not npzs: continue
        r=analyze_video(npzs)
        if not r: continue
        rows.append({"video_stem":vs, "suggested":r["suggested"], "conf":r["conf"]})
    vdf=pd.DataFrame(rows)
    vdf.to_csv(OUT_VID, index=False)

    vcl=assign[["video_stem","cluster_id_km"]].drop_duplicates()
    vdf=vdf.merge(vcl,on="video_stem",how="left")
    if len(vdf)==0:
        pd.DataFrame(columns=["cluster_id_km","suggested_gloss","vote_count","videos_scored","confidence"]).to_csv(OUT_CLU,index=False)
        print("[cluster] ->", OUT_CLU, "rows: 0"); return

    sug=vdf.groupby("cluster_id_km")["suggested"].agg(lambda s:s.value_counts().idxmax())
    cnt=vdf.groupby("cluster_id_km")["suggested"].agg(lambda s:s.value_counts().max())
    tot=vdf.groupby("cluster_id_km").size()
    cdf=pd.DataFrame({"cluster_id_km":sug.index,"suggested_gloss":sug.values,
                      "vote_count":cnt.values,"videos_scored":tot.values})
    cdf["confidence"]=(cdf["vote_count"]/cdf["videos_scored"].clip(lower=1)).astype(float)
    cdf=cdf.sort_values(["confidence","videos_scored"],ascending=[False,False])
    cdf.to_csv(OUT_CLU,index=False)
    print("[cluster] ->", OUT_CLU, "rows:", len(cdf))
    print(cdf.head(12).to_string(index=False))

if __name__=="__main__": main()
