import os, json, pandas as pd

ASSIGN = "/root/analysis/frame_cluster_assignments_v2.csv"   # has img_stem, cluster_id_km
CLUST  = "/root/labels/cluster_gloss_v2.csv"                  # has cluster_id_km, gloss
IMGDIR = "/root/dataset_images/artemis_dataset_20k/images/train"
OUTCSV = "/root/labels/labels.csv"
OUTMAP = "/root/labels/label_map.json"

def load_or_empty(path, cols):
    if os.path.isfile(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=cols)

def main():
    # 1) Load inputs
    assign = pd.read_csv(ASSIGN)
    if "cluster_id_km" not in assign.columns or "img_stem" not in assign.columns:
        raise RuntimeError(f"{ASSIGN} must contain columns ['img_stem','cluster_id_km']")
    cl = load_or_empty(CLUST, ["cluster_id_km","gloss"]).copy()
    if len(cl)==0:
        print(f"[warn] No cluster labels at {CLUST}; everything will be UNKNOWN.")
    # normalize types
    if "cluster_id_km" in cl.columns:
        cl["cluster_id_km"] = cl["cluster_id_km"].astype(int)
    assign["cluster_id_km"] = assign["cluster_id_km"].astype(int)

    # strip helper prefixes like AUTO_
    if "gloss" in cl.columns:
        cl["gloss"] = cl["gloss"].astype(str).str.replace(r"^AUTO_", "", regex=True).str.strip()

    # 2) Merge: assignment -> cluster gloss
    df = assign.merge(cl, on="cluster_id_km", how="left")

    # Build img_path from stem
    df["img_stem"] = df["img_stem"].astype(str)
    df["img_path"] = df["img_stem"].apply(lambda s: os.path.join(IMGDIR, f"{s}.jpg"))

    # Defaults for unlabeled rows
    df["gloss"] = df["gloss"].fillna("UNKNOWN")

    # 3) If an older labels.csv exists with manual labels, PREFER those
    prev = load_or_empty(OUTCSV, ["img_path","img_stem","label_id","gloss"])
    if len(prev):
        keep = prev[(prev.get("label_id",-1) >= 0) | (prev.get("gloss","UNKNOWN")!="UNKNOWN")][["img_stem","gloss"]].copy()
        if len(keep):
            keep = keep.drop_duplicates("img_stem")
            df = df.merge(keep, on="img_stem", how="left", suffixes=("", "_prev"))
            df["gloss"] = df["gloss_prev"].combine_first(df["gloss"])
            df = df.drop(columns=[c for c in df.columns if c.endswith("_prev")], errors="ignore")

    # 4) Label map (stable ids). UNKNOWN -> -1
    label_map = {}
    if os.path.isfile(OUTMAP):
        try:
            label_map = json.load(open(OUTMAP, "r"))
        except Exception:
            label_map = {}
    # ensure UNKNOWN reserved
    label_map = {k:int(v) for k,v in label_map.items()}
    label_map.pop("UNKNOWN", None)  # always treat UNKNOWN as -1

    # assign ids to any new glosses (excluding UNKNOWN)
    known = set(label_map.keys())
    new_glosses = sorted(set(df["gloss"].unique()) - {"UNKNOWN"} - known)
    next_id = (max(label_map.values())+1) if label_map else 0
    for g in new_glosses:
        label_map[g] = next_id
        next_id += 1

    # map to label_id
    def gid(g):
        return -1 if g=="UNKNOWN" else int(label_map.get(g, -1))

    df["label_id"] = df["gloss"].map(gid).astype(int)

    # 5) Final shape & save
    out = df[["img_path","img_stem","label_id","gloss"]].copy()
    out.to_csv(OUTCSV, index=False)
    json.dump(label_map, open(OUTMAP,"w"))
    # report
    n = len(out)
    n_l = int((out["label_id"]>=0).sum())
    n_cls = out.loc[out["label_id"]>=0,"gloss"].nunique()
    print(f"[done] wrote {OUTCSV} rows={n} labeled={n_l} classes={n_cls}")
    print(f"[map]  wrote {OUTMAP} entries={len(label_map)}")
    print(out.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
