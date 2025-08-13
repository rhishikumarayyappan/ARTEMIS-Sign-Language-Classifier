import os, re, json, math, glob
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from PIL import Image, ImageOps

FP   = "/root/features/feature_pack_v1.parquet"
OVER = "/root/final_outputs"
RAW  = "/root/dataset_images/artemis_dataset_20k/images/train"
OUTD = "/root/analysis"
os.makedirs(OUTD, exist_ok=True)
EXD  = os.path.join(OUTD, "cluster_examples"); os.makedirs(EXD, exist_ok=True)

def video_stem(stem:str)->str:
    return re.sub(r"_\d+$","", stem)

def load_features():
    df = pd.read_parquet(FP)
    meta = {"img_path","img_stem","signer_id","label_id","gloss"}
    feat_cols = [c for c in df.columns if c not in meta and np.issubdtype(df[c].dtype, np.number)]
    assert len(feat_cols)>=100, f"Unexpected feature cols ({len(feat_cols)})"
    df["video_stem"] = df["img_stem"].apply(video_stem)
    return df, feat_cols

def per_video(df, feat_cols):
    # mean features per video_stem; keep exemplar frame (first)
    exemplars = df.sort_values("img_stem").groupby("video_stem")["img_stem"].first().reset_index()
    X = df.groupby("video_stem")[feat_cols].mean().reset_index()
    X = X.merge(exemplars, on="video_stem", how="left", suffixes=("",""))
    return X  # columns: video_stem, <feat...>, img_stem (one exemplar)

def best_k(Xz, ks, max_eval=5000, seed=42):
    idx = np.random.RandomState(seed).choice(len(Xz), size=min(max_eval, len(Xz)), replace=False)
    Xs = Xz[idx]
    best, bestk = -1.0, ks[0]
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xs)
        sil = silhouette_score(Xs, km.labels_)
        print(f"[k-scan] k={k} silhouette={sil:.3f}")
        if sil > best:
            best, bestk = sil, k
    return bestk

def img_path_for(stem):
    p = os.path.join(OVER, f"final_{stem}.jpg")
    if os.path.isfile(p): return p
    p = os.path.join(RAW, f"{stem}.jpg")
    return p if os.path.isfile(p) else None

def montage(stems, out_path, cols=4, rows=2, size=256):
    canv = Image.new("RGB", (cols*size, rows*size), (30,30,30))
    for i,stem in enumerate(stems[:cols*rows]):
        p = img_path_for(stem)
        if not p: continue
        try:
            im = Image.open(p).convert("RGB")
            im = ImageOps.contain(im, (size,size))
            x = (i % cols)*size; y = (i//cols)*size
            canv.paste(im, (x,y))
        except Exception: pass
    canv.save(out_path)

def main():
    print("[load] features…")
    df, feat_cols = load_features()
    V  = per_video(df, feat_cols)
    X  = V[feat_cols].values
    print(f"[shape] videos={len(V)} frames={len(df)} dims={len(feat_cols)}")

    print("[prep] scale + pca…")
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=min(50, Xs.shape[1]))
    Z   = pca.fit_transform(Xs)
    print(f"[pca] -> {Z.shape[1]} dims (explained ~{pca.explained_variance_ratio_.sum():.2f})")

    print("[auto-k] scanning…")
    k = best_k(Z, ks=[20,30,40,50,60,80])
    print(f"[auto-k] choosing k={k}")

    print("[fit] kmeans…")
    km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(Z)
    V["cluster_id"] = km.labels_

    # attach cluster to every frame
    df["video_stem"] = df["img_stem"].apply(video_stem)
    assign = df[["img_stem","video_stem"]].merge(V[["video_stem","cluster_id"]], on="video_stem", how="left")

    # sizes
    sizes_vid = V.groupby("cluster_id").size().rename("n_videos")
    sizes_frm = assign.groupby("cluster_id").size().rename("n_frames")
    rep = V.groupby("cluster_id")["img_stem"].apply(list).rename("example_stems").reset_index()
    rep = rep.merge(sizes_vid, on="cluster_id").merge(sizes_frm, on="cluster_id")
    rep = rep.sort_values(["n_videos","n_frames"], ascending=False)

    # montages
    for cid, stems in zip(rep["cluster_id"], rep["example_stems"]):
        out = os.path.join(EXD, f"cluster_{cid:03d}.jpg")
        montage(stems, out)

    # save reports
    rep_out = os.path.join(OUTD, "cluster_report.csv")
    rep.to_csv(rep_out, index=False)
    assign.to_csv(os.path.join(OUTD, "frame_cluster_assignments.csv"), index=False)
    V[["video_stem","cluster_id"]].to_csv(os.path.join(OUTD, "video_cluster_assignments.csv"), index=False)

    print(f"[write] {rep_out}  (clusters={rep.shape[0]})")
    print(rep[["cluster_id","n_videos","n_frames"]].head(10).to_string(index=False))
    print(f"[examples] {EXD}/cluster_*.jpg")
if __name__=="__main__":
    main()
