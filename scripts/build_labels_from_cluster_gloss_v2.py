import os, json, pandas as pd

PACK   = "/root/features/feature_pack_v1.parquet"           # has img_path, img_stem
ASSIGN = "/root/analysis/frame_cluster_assignments_v2.csv"  # img_stem, cluster_id_km
CG     = "/root/labels/cluster_gloss_v2.csv"                # cluster_id_km, gloss

OUT_LBL = "/root/labels/labels.csv"
OUT_MAP = "/root/labels/label_map.json"

pack = pd.read_parquet(PACK)[["img_stem","img_path"]].drop_duplicates()
asgn = pd.read_csv(ASSIGN, dtype={"cluster_id_km":"Int64"})[["img_stem","cluster_id_km"]]
cg   = pd.read_csv(CG, dtype={"cluster_id_km":"Int64","gloss":"string"}) if os.path.isfile(CG) else pd.DataFrame(columns=["cluster_id_km","gloss"])

df = pack.merge(asgn, on="img_stem", how="left").merge(cg, on="cluster_id_km", how="left")
df["gloss"] = df["gloss"].fillna("UNKNOWN")

# normalize any auto/synonym tags → canonical glosses
syn = {
    "AUTO_YES": "YES",
    "AUTO_THANK_YOU": "THANK_YOU",
    "AUTO_HELLO": "HELLO",
}
df["gloss"] = df["gloss"].replace(syn)

# make label ids (UNKNOWN → -1)
glosses = [g for g in sorted(df["gloss"].unique()) if g!="UNKNOWN"]
g2id = {g:i for i,g in enumerate(glosses)}
df["label_id"] = df["gloss"].map(g2id).fillna(-1).astype(int)

df[["img_path","img_stem","label_id","gloss"]].to_csv(OUT_LBL, index=False)
with open(OUT_MAP,"w") as f: json.dump(g2id, f)

print(f"[done] wrote {OUT_LBL} rows={len(df)} labeled={(df['label_id']>=0).sum()} classes={len(g2id)}")
print(f"[map]  wrote {OUT_MAP}: {g2id}")
