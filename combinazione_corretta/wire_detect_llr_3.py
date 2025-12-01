# scopo: LLR = NCC(img, proto_pos) - NCC(img, proto_neg) dentro ROI
# Se LLR >= thr allora "filo presente" (da valutare se mettere 0 come threshold)

# PER COMPILARE: cd progetto
#                python wire_detect_llr.py --pos .\data\pos --neg .\data\neg --roi .\roi_mask.png --dir .\test --thr 0.0

import os, glob, argparse
import numpy as np
import cv2

def preprocess(bgr):
    # converte in scala di grigi se l'immagine è BGR
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim==3 else bgr
    # ridurre rumore ad alta frequenza LPF
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    # stabilizza il contrasto su immagini piccole: CLAHE leggero va bene
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    gray = clahe.apply(gray)
    return gray

def load_imgs(folder):
    # predisposizione: non solo per .png
    paths = []
    for ext in ("*.png","*.jpg","*.jpeg","*.bmp"): 
        paths += glob.glob(os.path.join(folder, ext))
    imgs = []
    # pre-process per ogni immagine 
    for p in sorted(paths):
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None: continue
        imgs.append(preprocess(im))
    return (np.stack(imgs,0) if imgs else None), sorted(paths)

def ncc(A, B, mask=None):
    # normalized cross-correlation tra A e B su ROI 
    A = A.astype(np.float32); B = B.astype(np.float32)
    if mask is not None:
        m = (mask>0).astype(np.float32)
        if m.sum() < 10: return 0.0 # controllo per non avere ROI troppo piccole (capire se tenerlo)
        # applica la maschera
        A = A*m; B = B*m
        # togli media solo nella ROI
        muA = A.sum()/m.sum()
        muB = B.sum()/m.sum()
        A = (A - muA)*m
        B = (B - muB)*m
        # prodotto delle norme ed evita il valore 0
        denom = (np.sqrt((A*A).sum()) * np.sqrt((B*B).sum()) + 1e-8)
        return float((A*B).sum() / denom)
    else: # se non ho una mask
        A -= A.mean(); B -= B.mean()
        denom = (np.linalg.norm(A) * np.linalg.norm(B) + 1e-8)
        return float((A*B).sum() / denom)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", required=True) #valutare se mettere help="..."
    ap.add_argument("--neg", required=True)
    ap.add_argument("--roi", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--thr", type=float, default=0.10)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    # maschera in scala di grigi: 
    roi = cv2.imread(args.roi, cv2.IMREAD_GRAYSCALE)
    if roi is None: raise FileNotFoundError(args.roi)

    # carica dataset di training 
    pos_stack, _ = load_imgs(args.pos)
    neg_stack, _ = load_imgs(args.neg)
    if pos_stack is None or neg_stack is None:
        raise RuntimeError("Put examples in --pos and in --neg")

    # prototipi (mediana robusta)
    proto_pos = np.median(pos_stack, axis=0).astype(np.uint8)
    proto_neg = np.median(neg_stack, axis=0).astype(np.uint8)

    # predisposizione: non solo per .png
    paths = []
    for ext in ("*.png","*.jpg","*.jpeg","*.bmp"):
        paths += glob.glob(os.path.join(args.dir, ext))
    if not paths: raise RuntimeError(f"No images in {args.dir}")

    # cartella di output 
    os.makedirs(args.outdir or args.dir, exist_ok=True)

    for p in sorted(paths):
        # pre-processa e prepara tutto alla stessa risoluzione
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None: print("[SKIP]", p); continue
        g = preprocess(bgr)

        # adatta ROI e prototipi se risoluzione diversa
        H, W = g.shape[:2]
        roi_rs = cv2.resize(roi, (W,H), interpolation=cv2.INTER_NEAREST)
        pp = cv2.resize(proto_pos, (W,H), interpolation=cv2.INTER_AREA)
        pn = cv2.resize(proto_neg, (W,H), interpolation=cv2.INTER_AREA)

        # Calcolo punteggi di correlazione nella ROI
        s_pos = abs(ncc(g, pp, mask=roi_rs)) # abs perchè sia +1 che -1 significativi abs(ncc(g, pp, mask=roi_rs))
        s_neg = ncc(g, pn, mask=roi_rs) # da capire se mettere abs anche qui, ma non credo
        llr = s_pos - s_neg # se filo nero non dovrei aver problemi(?) fare test
        present = llr >= args.thr # capire se mettere thr = 0
        #present = s_pos >= s_neg

        # visualizzazione  (non serve ai fini del progetto -> da togliere una volta risolto riconoscimento di errori) 
        vis = bgr.copy()
        color = (0,255,0) if present else (0,0,255) # comodo in fase di debug
        cnts,_ = cv2.findContours((roi_rs>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  #
        cv2.drawContours(vis, cnts, -1, color, 2)                                                           #
        cv2.putText(vis, f"LLR={llr:.3f}",
                    (5, max(18,H-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)

        outdir = args.outdir or args.dir
        base = os.path.splitext(os.path.basename(p))[0]
        cv2.imwrite(os.path.join(outdir, f"out_{base}.png"), vis)
        print(f"[{'OK' if present else 'ERROR'}] {os.path.basename(p)}  LLR={llr:.3f} (pos={s_pos:.3f} neg={s_neg:.3f})")

if __name__ == "__main__":
    main()
