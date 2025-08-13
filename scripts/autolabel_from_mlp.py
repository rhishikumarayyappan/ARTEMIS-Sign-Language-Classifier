import os, json, torch, numpy as np, pandas as pd

FP="/root/features/feature_pack_v1.parquet"
MODEL="/root/models/mlp_v1_clustered/model.pt"
ID2="/root/models/mlp_v1_clustered/id_to_gloss.json"
OUT="/root/labels/autolabel_adoptions.csv"

df = pd.read_parquet(FP)
labeled = df["label_id"]>=0
unl = df[~labeled].copy()

id2 = json.load(open(ID2))
id2 = {int(k):v for k,v in id2.items()}

ckpt = torch.load(MODEL, map_location="cpu")
feat_cols = ckpt.get("feat_cols", [c for c in df.columns if c.startswith(("x","f"))])
in_dim = ckpt["in_dim"]; n_classes = ckpt["n_classes"]

class MLP(torch.nn.Module):
    def __init__(self, in_dim, n_classes, hidden=(512,256), p=0.3):
        super().__init__()
        layers=[]; d=in_dim
        for h in hidden:
            layers += [torch.nn.Linear(d,h), torch.nn.ReLU(inplace=True),
                       torch.nn.BatchNorm1d(h), torch.nn.Dropout(p)]
            d=h
        layers += [torch.nn.Linear(d,n_classes)]
        self.net=torch.nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(in_dim, n_classes).to(device)
model.load_state_dict(ckpt["model_state"], strict=True)
model.eval()

X = unl[feat_cols].astype("float32").values
B=2048; preds=[]; probs=[]
with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device.type=='cuda')):
    for i in range(0, len(X), B):
        xb = torch.from_numpy(X[i:i+B]).to(device)
        logits = model(xb)
        p = torch.softmax(logits, dim=1)
        conf, lab = torch.max(p, dim=1)
        preds.append(lab.cpu().numpy()); probs.append(conf.cpu().numpy())
preds = np.concatenate(preds); probs = np.concatenate(probs)

out = unl[["img_path","img_stem"]].copy()
out["pred_id"] = preds
out["pred_gloss"] = out["pred_id"].map(id2)
out["confidence"] = probs
# adopt only *very* confident predictions
TH = float(os.environ.get("ADOPT_TH", "0.99"))
MIN = int(os.environ.get("ADOPT_MIN", "50"))   # only adopt classes with >= MIN predictions
sel = out[out["confidence"]>=TH].copy()
# avoid adopting tiny tails
vc = sel["pred_id"].value_counts()
big = set(vc[vc>=MIN].index.tolist())
sel = sel[sel["pred_id"].isin(big)]
sel.to_csv(OUT, index=False)
print(f"[done] candidates>={TH}: {len(out[out['confidence']>=TH])} | adopted (>=MIN): {len(sel)} -> {OUT}")
print(sel["pred_gloss"].value_counts().to_string())
