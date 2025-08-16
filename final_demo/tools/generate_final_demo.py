# (same imports & colors as your last version)
import argparse, json, os, time
from pathlib import Path
from itertools import chain
import numpy as np
import onnxruntime as ort
import cv2

BG=(28,28,28); CARD=(40,40,40); DIV=(70,70,70); MUTED=(170,170,170)
TEXT=(235,235,235); OK=(80,220,120); WARN=(255,191,0); ERR=(60,60,255)
OOV=(0,215,255); BAR_BG=(60,60,60); BAR_FG=(60,230,140)
EMO_COL={"happy":(90,230,90),"neutral":(190,190,190),"sad":(180,120,240),
         "angry":(60,60,255),"fear":(40,200,255),"surprise":(80,200,255),
         "n/a":(150,150,150),"na":(150,150,150)}

TITLE="ARTEMIS | Sign & Emotion Analysis"
PANEL_W,PANEL_H=660,720; PADX,PADY=22,18; RADIUS=10
SCORE_COL_W=90; BAR_H=16
try: FONT=cv2.FONT_HERSHEY_DUPLEX
except: FONT=cv2.FONT_HERSHEY_SIMPLEX
S_H1,TH_H1=0.95,1; S_H2,TH_H2=1.55,2; S_TX,TH_TX=0.95,1; S_FT,TH_FT=0.90,1

DEF_FEAT=Path('~/Desktop/wlasl_mediapipe_features_rel').expanduser()
DEF_VIDEOS=Path('~/Desktop/WLASL/start_kit/videos').expanduser()
ALT_VIDEOS=Path('~/Desktop/WLASL/start_kit/raw_videos').expanduser()
DEF_MODEL=Path('models/model_v2.onnx').expanduser()
DEF_LABELS=Path('models/label2gloss.json').expanduser()
EMO_CSV=Path('logs/emotion_dataset_robust.csv').expanduser()

def softmax(a,t=1.0):
    a=np.asarray(a,np.float32)/max(t,1e-6); e=np.exp(a-a.max()); return e/e.sum()

def ascii_only(s:str)->str:
    return (s.replace("“","\"").replace("”","\"").replace("’","'")
             .replace("–","-").replace("—","-"))

def text_metrics(scale,th):
    (w,h),base=cv2.getTextSize("Hg",FONT,scale,th)
    return h, base

# text metrics (height & baseline)
H1,B1=text_metrics(S_H1,TH_H1)
H2,B2=text_metrics(S_H2,TH_H2)
HT,BT=text_metrics(S_TX,TH_TX)
HF,BF=text_metrics(S_FT,TH_FT)

def baseline_in_rect(y_top, rect_h, text_h):
    # center the *baseline* so the glyph body (height=text_h) is visually centered
    return int(y_top + (rect_h + text_h)//2)

def put(img,txt,x,baseline_y,col,s,th):
    cv2.putText(img,ascii_only(txt),(int(x),int(baseline_y)),FONT,s,col,th,cv2.LINE_AA)

def right_txt(img,txt,right_x,baseline_y,col,s,th):
    txt=ascii_only(txt); (tw,_),_=cv2.getTextSize(txt,FONT,s,th)
    put(img,txt,right_x-tw,baseline_y,col,s,th)

def round_rect(img,x1,y1,x2,y2,r,color,alpha=1.0):
    if alpha<1: overlay=img.copy()
    canvas=img if alpha==1 else overlay
    cv2.rectangle(canvas,(x1+r,y1),(x2-r,y2),color,-1,cv2.LINE_AA)
    cv2.rectangle(canvas,(x1,y1+r),(x2,y2-r),color,-1,cv2.LINE_AA)
    for (cx,cy,a1,a2) in [(x1+r,y1+r,180,270),(x2-r,y1+r,270,360),
                          (x1+r,y2-r,90,180),(x2-r,y2-r,0,90)]:
        cv2.ellipse(canvas,(cx,cy),(r,r),0,a1,a2,color,-1,cv2.LINE_AA)
    if alpha<1: cv2.addWeighted(overlay,alpha,img,1-alpha,0,img)

def bar(img,x,y_top,w,val,h=BAR_H,fg=BAR_FG,bg=BAR_BG):
    val=float(np.clip(val,0,1))
    cv2.rectangle(img,(x,y_top),(x+w,y_top+h),bg,-1,cv2.LINE_AA)
    cv2.rectangle(img,(x,y_top),(x+int(w*val),y_top+h),fg,-1,cv2.LINE_AA)

def load_labels(path:Path):
    data=json.load(open(path,'r'))
    if isinstance(data,dict):
        return [v for _,v in sorted(((int(k),v) for k,v in data.items()), key=lambda x:x[0])]
    if isinstance(data,list): return data
    raise ValueError("label2gloss.json must be list or dict(index->gloss)")

def find_npz_and_gloss(feat_dir:Path,sid:str):
    for f in chain(feat_dir.glob(f'val*__*{sid}.npz'),
                   feat_dir.glob(f'test*__*{sid}.npz'),
                   feat_dir.glob(f'train*__*{sid}.npz')):
        return f, f.stem.split('__')[1]
    return None, None

def find_video(vdir:Path,alt:Path,sid:str):
    for p in [vdir/f"{sid.zfill(5)}.mp4", vdir/f"{sid}.mp4",
              alt/f"{sid.zfill(5)}.mp4", alt/f"{sid}.mp4"]:
        if p.exists(): return p
    return None

def panel_build(true_gloss,pred_label,pred_conf,top5,emotion,status_text,status_color,latency_ms):
    W,H=PANEL_W,PANEL_H
    panel=np.full((H,W,3),BG,np.uint8)
    xL,xR=PADX,W-PADX; y=PADY

    # Title
    put(panel,TITLE,xL,y+H1,(220,230,255),S_H1,TH_H1)
    cv2.line(panel,(xL,y+H1+8),(xR,y+H1+8),DIV,1,cv2.LINE_AA)
    y+=H1+18

    # Ground Truth card
    gt_top=y; gt_inner_top=gt_top+PADY
    card_h=PADY+(H1+10+H2)+PADY
    round_rect(panel,xL,gt_top,xR,gt_top+card_h,10,CARD,0.92)
    put(panel,"Ground Truth",xL+PADX,gt_inner_top+H1,MUTED,S_H1,TH_H1)
    put(panel,(true_gloss or "--").upper(),xL+PADX,gt_inner_top+H1+10+H2,TEXT,S_H2,TH_H2)
    y+=card_h+10

    # Prediction card
    pred_top=y; pred_inner_top=pred_top+PADY
    card_h=PADY+(H1+10+H2+12+BAR_H)+PADY
    round_rect(panel,xL,pred_top,xR,pred_top+card_h,10,CARD,0.92)

    bl_h1=pred_inner_top+H1
    bl_h2=bl_h1+10+H2
    put(panel,f"Model Prediction ({status_text})",xL+PADX,bl_h1,status_color,S_H1,TH_H1)
    put(panel,(pred_label or "--").upper(),xL+PADX,bl_h2,status_color,S_H2,TH_H2)

    if pred_conf is not None:
        bar_left=xL+PADX
        bar_right=xR-PADX-SCORE_COL_W
        bar_w=max(0,bar_right-bar_left)
        # bar sits in its own rect just under the big label
        bar_top = bl_h2 + 10
        bar(panel,bar_left,bar_top,bar_w,pred_conf,h=BAR_H,fg=status_color,bg=BAR_BG)
        # number baseline centered inside the bar
        num_y=baseline_in_rect(bar_top,BAR_H,HT)
        right_txt(panel,f"{pred_conf:.2f}",xR-PADX,num_y,TEXT,S_TX,TH_TX)

    y+=card_h+10

    # Top-5 card (rows are their own boxes)
    top_top=y; top_inner_top=top_top+PADY
    row_h=max(HT,BAR_H); row_gap=row_h+12
    rows=min(5,len(top5))
    card_h=PADY+(H1+10+rows*row_gap)+PADY
    round_rect(panel,xL,top_top,xR,top_top+card_h,10,CARD,0.92)

    put(panel,"Top-5",xL+PADX,top_inner_top+H1,MUTED,S_H1,TH_H1)
    rows_y0=top_inner_top+H1+10

    num_w=cv2.getTextSize("5.",FONT,S_TX,TH_TX)[0][0]
    lab_w=0
    for lbl,_ in top5[:rows]:
        lab_w=max(lab_w,cv2.getTextSize(ascii_only(lbl),FONT,S_TX,TH_TX)[0][0])

    lab_x=xL+PADX+num_w+12
    bar_x=lab_x+lab_w+16
    bar_w=(xR-PADX-SCORE_COL_W)-bar_x

    for i,(lbl,sc) in enumerate(top5[:rows],start=1):
        row_top=rows_y0+(i-1)*row_gap
        # center baselines & bars within the same row box
        baseline=baseline_in_rect(row_top,row_h,HT)
        bar_top=row_top+(row_h-BAR_H)//2
        put(panel,f"{i}.",xL+PADX,baseline,TEXT,S_TX,TH_TX)
        put(panel,lbl,lab_x,baseline,TEXT,S_TX,TH_TX)
        bar(panel,bar_x,bar_top,max(0,bar_w),sc,h=BAR_H,fg=BAR_FG,bg=BAR_BG)
        right_txt(panel,f"{sc:.2f}",xR-PADX,baseline,TEXT,S_TX,TH_TX)

    y+=card_h+10

    # Emotion card
    emo_top=y; emo_inner_top=emo_top+PADY
    card_h=PADY+(H1+10+H2)+PADY
    round_rect(panel,xL,emo_top,xR,emo_top+card_h,10,CARD,0.92)
    put(panel,"Robust Emotion",xL+PADX,emo_inner_top+H1,MUTED,S_H1,TH_H1)
    emo=(emotion or "N/A").lower().strip(); col=EMO_COL.get(emo,MUTED)
    put(panel,(emo if emo!="n/a" else "N/A").upper(),xL+PADX,emo_inner_top+H1+10+H2,col,S_H2,TH_H2)

    right_txt(panel,f"Inference: {latency_ms:.1f} ms     Press Q to exit",
              PANEL_W-PADX,PANEL_H-14,MUTED,S_FT,TH_FT)
    return panel

def main():
    ap=argparse.ArgumentParser("ARTEMIS Replay Demo (aligned ASCII UI)")
    ap.add_argument('--sample_id',required=True)
    ap.add_argument('--features_dir',default=str(DEF_FEAT))
    ap.add_argument('--videos_dir',default=str(DEF_VIDEOS))
    ap.add_argument('--alt_videos',default=str(ALT_VIDEOS))
    ap.add_argument('--model',default=str(DEF_MODEL))
    ap.add_argument('--labels',default=str(DEF_LABELS))
    ap.add_argument('--emotion_csv',default=str(EMO_CSV))
    ap.add_argument('--temperature', type=float, default=0.55)
    ap.add_argument('--unknown_thr', type=float, default=0.55)
    ap.add_argument('--title',default="ARTEMIS Replay Demo")
    args=ap.parse_args()

    os.chdir(Path(__file__).resolve().parent.parent)

    labels=load_labels(Path(args.labels))
    sess=ort.InferenceSession(str(Path(args.model))); inp=sess.get_inputs()[0].name

    npz,true_gloss=find_npz_and_gloss(Path(args.features_dir),args.sample_id)
    if not npz: raise SystemExit(f"[ERROR] NPZ not found for {args.sample_id}")
    vid=find_video(Path(args.videos_dir),Path(args.alt_videos),args.sample_id)
    if not vid: raise SystemExit(f"[ERROR] Video not found for {args.sample_id}")

    emotion="N/A"
    try:
        import pandas as pd
        df=pd.read_csv(Path(args.emotion_csv))
        mp={int(r["video_id"]):str(r["robust_emotion"]) for _,r in df.iterrows()}
        emotion=mp.get(int(args.sample_id),"N/A")
    except Exception:
        pass

    is_known=true_gloss in set(labels)
    pred_label="(--)" ; pred_conf=None ; latency_ms=0.0 ; top5=[]
    status_text,status_color=("OUT-OF-VOCABULARY",OOV)

    if is_known:
        x=np.load(npz)['x'].astype(np.float32)
        t0=time.perf_counter(); logits=sess.run(None,{inp:np.expand_dims(x,0)})[0][0]; t1=time.perf_counter()
        latency_ms=(t1-t0)*1000
        prob=softmax(logits,t=args.temperature)
        idx=int(prob.argmax()); pred_conf=float(prob[idx])
        pred_label=labels[idx] if pred_conf>=args.unknown_thr else "(Abstain)"
        order=np.argsort(prob)[::-1][:5]; top5=[(labels[i],float(prob[i])) for i in order]
        if pred_label=="(Abstain)": status_text,status_color=("ABSTAINED",WARN)
        elif pred_label==true_gloss: status_text,status_color=("CORRECT",OK)
        else:                        status_text,status_color=("WRONG",ERR)

    panel=panel_build(true_gloss,pred_label,pred_conf,top5,emotion,status_text,status_color,latency_ms)
    cap=cv2.VideoCapture(str(vid)); 
    if not cap.isOpened(): raise SystemExit(f"[ERROR] Could not open {vid}")

    while True:
        ok,frame=cap.read()
        if not ok: cap.set(cv2.CAP_PROP_POS_FRAMES,0); continue
        fh=PANEL_H; fw=int(frame.shape[1]*(fh/frame.shape[0]))
        video=cv2.resize(frame,(fw,fh))
        cv2.imshow(args.title,np.hstack([video,panel]))
        if cv2.waitKey(30)&0xFF==ord('q'): break
    cap.release(); cv2.destroyAllWindows()

if __name__=="__main__": main()
