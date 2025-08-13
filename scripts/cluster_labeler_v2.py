import os, pandas as pd, gradio as gr

EXD = "/root/analysis/cluster_examples_v2"
REP = "/root/analysis/cluster_report_kmeans.csv"
OUT = "/root/labels/cluster_gloss_v2.csv"
CLASS_TXT = "/root/labels/class_list.txt"

def load_classes():
    if not os.path.isfile(CLASS_TXT): return []
    return [ln.strip() for ln in open(CLASS_TXT, "r", encoding="utf-8").read().splitlines()
            if ln.strip() and not ln.startswith("#")]

def load_report():
    df = pd.read_csv(REP)
    # ensure n_frames exists (merge if needed)
    if "n_frames" not in df.columns:
        try:
            assign = pd.read_csv("/root/analysis/frame_cluster_assignments_v2.csv",
                                 usecols=["img_stem","cluster_id_km"])
            nf = assign.groupby("cluster_id_km").size().rename("n_frames").reset_index()
            df = df.merge(nf, on="cluster_id_km", how="left")
            df.to_csv(REP, index=False)
        except Exception:
            pass
    df = df.sort_values(["n_videos", "n_frames"], ascending=False).reset_index(drop=True)
    return df

def load_existing():
    return pd.read_csv(OUT) if os.path.isfile(OUT) else pd.DataFrame(columns=["cluster_id_km","gloss"])

def img_for(cid):
    p = os.path.join(EXD, f"km_cluster_{int(cid):03d}.jpg")
    return p if os.path.isfile(p) else None

def show(i):
    df = load_report()
    labs = load_existing()
    if len(df)==0:
        return None, "", "", "No clusters."
    i = max(0, min(i, len(df)-1))
    row = df.iloc[i]
    cid = int(row["cluster_id_km"])
    img = img_for(cid)
    gmap = dict(zip(labs["cluster_id_km"], labs["gloss"]))
    gloss = gmap.get(cid, None)
    nf = int(row["n_frames"]) if "n_frames" in row else -1
    info = f"cluster {cid} • videos:{int(row['n_videos'])} • frames:{nf} • {i+1}/{len(df)}"
    return img, str(cid), gloss, info, i, len(df)

def save_and_next(i, cid, gloss):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df = load_existing()
    cid = int(cid)
    if (df["cluster_id_km"]==cid).any():
        df.loc[df["cluster_id_km"]==cid, "gloss"] = gloss
    else:
        df = pd.concat([df, pd.DataFrame([{"cluster_id_km":cid,"gloss":gloss}])], ignore_index=True)
    df.to_csv(OUT, index=False)
    return show(i+1)

with gr.Blocks(title="Cluster Labeler v2") as demo:
    gr.Markdown("Assign a gloss per **cluster**. Label big/clean clusters first.")
    img = gr.Image(type="filepath", label="Cluster montage", height=512)
    cid = gr.Textbox(label="cluster_id_km", interactive=False)
    gloss = gr.Dropdown(choices=load_classes(), allow_custom_value=True, label="gloss")
    info = gr.Markdown()
    prevb = gr.Button("Prev"); saveb = gr.Button("Save + Next", variant="primary"); nextb = gr.Button("Next")
    idx = gr.Slider(0, 1, step=1, value=0, label="Index")

    demo.load(lambda: show(0), outputs=[img,cid,gloss,info,idx,idx])
    prevb.click(lambda i: show(max(i-1,0)), [idx], [img,cid,gloss,info,idx,idx])
    nextb.click(lambda i: show(i+1), [idx], [img,cid,gloss,info,idx,idx])
    saveb.click(save_and_next, [idx,cid,gloss], [img,cid,gloss,info,idx,idx])

demo.launch(server_name="0.0.0.0", server_port=7862, share=True)
