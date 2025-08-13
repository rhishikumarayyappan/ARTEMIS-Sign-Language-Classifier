# --- shortened header ---
import os, re, json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap
try:
    import hdbscan; HAS_HDBSCAN=True
except Exception:
    HAS_HDBSCAN=False
from PIL import Image, ImageOps

FP="/root/features/feature_pack_v1.parquet"
OVER="/root/final_outputs"
RAW ="/root/dataset_images/artemis_dataset_20k/images/train"
OUT="/root/analysis"; EX=os.path.join(OUT,"cluster_examples_v2")
FIX_K=int(os.environ.get("FIX_K","0"))
RUN_HDB=os.environ.get("RUN_HDBSCAN","1")=="1" and HAS_HDBSCAN
HDB_MIN=int(os.environ.get("HDBSCAN_MIN","50")); SEED=42
os.makedirs(EX,exist_ok=True)

def vstem(s): return re.sub(r"_\d+$","",s)
def load():
    df=pd.read_parquet(FP)
    meta={"img_path","img_stem","signer_id","label_id","gloss"}
    feats=[c for c in df.columns if c not in meta and np.issubdtype(df[c].dtype, np.number)]
    assert len(feats)==220
    w=np.r_[np.ones(136,dtype=np.float32), np.full(84,1.5,dtype=np.float32)]
    for i,c in enumerate(feats): df[c]*=w[i]
    df["video_stem"]=df["img_stem"].apply(vstem)
    return df,feats
def per_video(df,feats):
    first=(df.sort_values("img_stem").groupby("video_stem")["img_stem"].first().reset_index())
    X=df.groupby("video_stem")[feats].mean().reset_index().merge(first,on="video_stem")
    return X
def img_path(stem):
    p=os.path.join(OVER,f"final_{stem}.jpg")
    return p if os.path.isfile(p) else (os.path.join(RAW,f"{stem}.jpg") if os.path.isfile(os.path.join(RAW,f"{stem}.jpg")) else None)
def montage(stems,out,cols=4,rows=2,size=256):
    import PIL; canv=Image.new("RGB",(cols*size,rows*size),(30,30,30))
    for i,s in enumerate(stems[:cols*rows]):
        p=img_path(s); 
        if not p: continue
        try:
            im=Image.open(p).convert("RGB")
            im=ImageOps.contain(im,(size,size))
            canv.paste(im,((i%cols)*size,(i//cols)*size))
        except Exception: pass
    canv.save(out)

df,feats=load(); V=per_video(df,feats); X=V[feats].values
print(f"[shape] videos={len(V)} frames={len(df)} dims={len(feats)}")
Xz=StandardScaler().fit_transform(X)
pca=PCA(n_components=min(Xz.shape[1],100),random_state=SEED).fit(Xz)
k=np.searchsorted(np.cumsum(pca.explained_variance_ratio_),0.95)+1
Zp=PCA(n_components=int(k),random_state=SEED).fit_transform(Xz)
Zu=umap.UMAP(n_components=15,random_state=SEED,n_neighbors=30,min_dist=0.1).fit_transform(Zp)
def scan(ks=(40,60,80,100,120)):
    import numpy as np
    rs=np.random.RandomState(SEED); idx=rs.choice(len(Zu),size=min(5000,len(Zu)),replace=False)
    best=-1;kbest=ks[0]
    for k in ks:
        km=KMeans(n_clusters=k,n_init=30,random_state=SEED).fit(Zu[idx])
        s=silhouette_score(Zu[idx],km.labels_); print(f"[k-scan] k={k} silhouette={s:.3f}")
        if s>best: best=s;kbest=k
    return kbest
K=FIX_K or scan(); print(f"[kmeans] using k={K}")
km=KMeans(n_clusters=K,n_init=30,random_state=SEED).fit(Zu)
V["cluster_id_km"]=km.labels_
try: print(f"[kmeans] silhouette ~ {silhouette_score(Zu,km.labels_):.3f}")
except Exception: pass
if RUN_HDB:
    print(f"[hdbscan] min_cluster_size={HDB_MIN}")
    h=hdbscan.HDBSCAN(min_cluster_size=HDB_MIN,min_samples=HDB_MIN//2,prediction_data=True).fit(Zu)
    V["cluster_id_hdb"]=h.labels_
else:
    V["cluster_id_hdb"]=-1
df["video_stem"]=df["img_stem"].apply(vstem)
assign=df[["img_stem","video_stem"]].merge(V[["video_stem","cluster_id_km","cluster_id_hdb"]],on="video_stem",how="left")
rep=(V.groupby("cluster_id_km")["img_stem"].apply(list).rename("example_stems").reset_index())
rep["n_videos"]=rep["example_stems"].apply(len); rep=rep.sort_values("n_videos",ascending=False)
for cid,stems in zip(rep["cluster_id_km"],rep["example_stems"]):
    montage(stems, os.path.join(EX,f"km_cluster_{int(cid):03d}.jpg"))
rep.to_csv(os.path.join(OUT,"cluster_report_kmeans.csv"),index=False)
assign.to_csv(os.path.join(OUT,"frame_cluster_assignments_v2.csv"),index=False)
V[["video_stem","cluster_id_km","cluster_id_hdb"]].to_csv(os.path.join(OUT,"video_cluster_assignments_v2.csv"),index=False)
print("[done] wrote reports to", OUT, "and montages to", EX)
