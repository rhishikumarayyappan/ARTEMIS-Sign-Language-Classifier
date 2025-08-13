import os, re, json, pandas as pd, numpy as np
from pathlib import Path
import torch, torch.nn as nn

PACK="/root/features/feature_pack_v1.parquet"
ASSIGN="/root/analysis/frame_cluster_assignments_v2.csv"
LABELS="/root/labels/labels.csv"

CLUSTER=int(os.environ.get("CLUSTER", "45"))
TARGET=os.environ.get("TARGET", "THANK_YOU")
THRESH=float(os.environ.get("THRESH","0.995"))
MODEL_DIR=os.environ.get("MODEL_DIR","/root/models/mlp_v6_filtered")

MODEL_PT  = str(Path(MODEL_DIR)/"model.pt")
ID2_JSON  = str(Path(MODEL_DIR)/"id_to_gloss.json")

def detect_feature_cols(df):
    import pandas as pd, re
    pat = re.compile(r'^x\d+$')
    f = [c for c in df.columns if pat.fullmatch(str(c))]
    if f:
        return sorted(f, key=lambda x:int(x[1:]))
    # fallback: any float columns excluding meta
    ignore = {'img_path','img_stem','label_id','gloss','signer_id','cluster_id_km'}
    f = [c for c in df.columns if c not in ignore and hasattr(df[c], 'dtype') and df[c].dtype.kind in 'fc']
    return f

def build_mlp_from_checkpoint(state, n_classes):
    # collect linear layer "bases" from checkpoint (e.g., net.0, net.3, net.6, net.9)
    bases = []
    for k, v in state.items():
        if not (k.startswith("net.") and k.endswith(".weight")): 
            continue
        if getattr(v, "ndim", 0) != 2: 
            continue
        idx = int(k.split(".")[1])
        base = k[:-7]  # strip ".weight"
        bases.append((idx, base))
    bases = [b for _, b in sorted(set(bases), key=lambda t:t[0])]

    # infer dims from weights in order
    dims = []
    for i, base in enumerate(bases):
        W = state[base + ".weight"]
        in_dim, out_dim = W.shape[1], W.shape[0]
        if i == 0:
            dims = [in_dim, out_dim]
        else:
            dims.append(out_dim)

    class MLP(nn.Module):
        def __init__(self, dims):
            super().__init__()
            layers = []
            for i in range(len(dims)-1):
                layers.append(nn.Linear(dims[i], dims[i+1]))
                if i < len(dims)-2:
                    layers.append(nn.ReLU())
            self.net = nn.Sequential(*layers)
        def forward(self, x): 
            return self.net(x)

    model = MLP(dims)

    # map checkpoint weights -> our model (linear layers are at indices 0,2,4,... in Sequential)
    sd = model.state_dict()
    lin_idx = 0
    for base in bases:
        wkey = f"net.{lin_idx*2}.weight"
        bkey = f"net.{lin_idx*2}.bias"
        if wkey in sd:
            sd[wkey].copy_(state[base + ".weight"])
        if bkey in sd and (base + ".bias") in state and state[base + ".bias"] is not None:
            sd[bkey].copy_(state[base + ".bias"])
        lin_idx += 1

    model.load_state_dict(sd, strict=False)
    return model, dims

# ---- data
df = pd.read_parquet(PACK)
fcols = detect_feature_cols(df)
if len(fcols) == 0:
    raise RuntimeError("No feature columns matched x\\d+ (expected x0..x219).")

assign = pd.read_csv(ASSIGN)[["img_stem","cluster_id_km"]]
df = df.merge(assign, on="img_stem", how="left")
cand = df[df["cluster_id_km"]==CLUSTER].copy()
if cand.empty:
    print(f"[warn] no frames for cluster {CLUSTER}")
    raise SystemExit

with open(ID2_JSON,"r") as f:
    id2 = {int(k):v for k,v in json.load(f).items()}
n_classes = len(id2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(MODEL_PT, map_location=device)
state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt

model, dims = build_mlp_from_checkpoint(state, n_classes)
model.to(device).eval()

X = torch.tensor(cand[fcols].astype(np.float32).values, device=device)
with torch.no_grad():
    logits = model(X)
    if logits.shape[1] != n_classes:
        raise RuntimeError(f"Checkpoint head outputs {logits.shape[1]} classes, but id2 has {n_classes}. Use a matching model_dir.")
    probs = torch.softmax(logits, dim=1).cpu().numpy()

pred_id   = probs.argmax(1)
pred_conf = probs.max(1)
pred_gloss= [id2[i] for i in pred_id]

cand["pred_gloss"] = pred_gloss
cand["pred_conf"]  = pred_conf
sel = cand[(cand["pred_gloss"]==TARGET) & (cand["pred_conf"]>=THRESH)]

print(f"[load] dims={dims} classes={n_classes}")
print(f"[cluster {CLUSTER}] frames={len(cand)} | adopt {len(sel)} as {TARGET} @ p≥{THRESH}")

# update labels.csv
lab = pd.read_csv(LABELS)
lab = lab.merge(sel[["img_stem"]].assign(gloss=TARGET), on="img_stem", how="left", suffixes=("","_new"))
lab["gloss"] = lab["gloss_new"].fillna(lab["gloss"])
lab = lab.drop(columns=["gloss_new"])
lab.to_csv(LABELS, index=False)
print(f"[labels] updated {LABELS} | adopted={len(sel)}")
