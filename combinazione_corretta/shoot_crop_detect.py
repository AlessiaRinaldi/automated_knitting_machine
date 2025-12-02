#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, glob
from datetime import datetime
from gpiozero import Button
from picamera2 import Picamera2
import cv2
import numpy as np

# ==========================
# CONFIGURAZIONE CAMERA / CROP
# ==========================
SAVE_DIR = "/home/aless/foto"
CROP_DIR = os.path.join(SAVE_DIR, "crops")

CAM_WIDTH, CAM_HEIGHT = 1280, 960
ROI_MODE = "relative"   # "relative" or "pixels"
ROI_REL = (0.47, 0.75, 0.06, 0.08)
ROI_PIX = (560, 760, 120, 120)

BUTTON_BOUNCE = 0.05
BUTTON_PIN = 17
SAVE_EXT = "jpg"

# ==========================
# CONFIGURATION LLR
# ==========================
POS_DIR = "/home/aless/llr/pos"                     # pos directory
NEG_DIR = "/home/aless/llr/neg"                     # neg directory
ROI_MASK_PATH = "/home/aless/llr/mask_soft.png"     # mask ROI
THR = -0.035                                        # threshold LLR (=-0.035 to avoid false negative)
SAVE_DEBUG_VIS = True                               # if True, save img whit LLR in OUT_DIR

# ==========================
# UTILS CROP
# ==========================
def ensure_dir(d): 
    os.makedirs(d, exist_ok=True)

def clamp_roi(x, y, w, h, W, H):
    x = max(0, min(int(x), W-1))
    y = max(0, min(int(y), H-1))
    w = max(1, min(int(w), W - x))
    h = max(1, min(int(h), H - y))
    return x, y, w, h

def roi_from_mode(W, H):
    if ROI_MODE == "relative":
        rx, ry, rw, rh = ROI_REL
        x = int(rx * W); y = int(ry * H)
        w = int(rw * W); h = int(rh * H)
    else:
        x, y, w, h = ROI_PIX
    return clamp_roi(x, y, w, h, W, H)

def crop_image(img_path, save=True):
    """
    Legge immagine da disco, fa il crop in base alla ROI
    e (opzionale) salva il ritaglio in CROP_DIR.
    return (crop_bgr, crop_path_oppure_None).
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[WARN] Impossible reading {img_path}")
        return None, None

    H, W = img.shape[:2]
    x, y, w, h = roi_from_mode(W, H)
    crop = img[y:y+h, x:x+w]

    if save:
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(CROP_DIR, f"{base}_crop.png")
        if cv2.imwrite(out_path, crop):
            print(f"[OK] crop -> {out_path} (ROI x={x} y={y} w={w} h={h}, img={W}x{H})")
            return crop, out_path
        else:
            print(f"[ERR] img not saved {img_path}")
            return crop, None
    else:
        print(f"[OK] crop in memory (ROI in {W}x{H})")
        return crop, None

# ==========================
# UTILS LLR 
# ==========================
def preprocess(bgr):
    # if BGR conversion in gray scale
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    # riduction rumore at high freq (LPF)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    # stabilizza il contrasto su immagini piccole: CLAHE leggero va bene
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    gray = clahe.apply(gray)
    return gray

def load_imgs(folder):
    paths = []
    for ext in ("*.png","*.jpg","*.jpeg","*.bmp"): 
        paths += glob.glob(os.path.join(folder, ext))
    imgs = []
    for p in sorted(paths):
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None:
            continue
        imgs.append(preprocess(im))
    return (np.stack(imgs,0) if imgs else None), sorted(paths)

def ncc(A, B, mask=None):
    # normalized cross-correlation between A e B in ROI 
    A = A.astype(np.float32); B = B.astype(np.float32)
    if mask is not None:
        m = (mask>0).astype(np.float32)
        if m.sum() < 10:
            return 0.0
        A = A*m; B = B*m
        muA = A.sum()/m.sum()
        muB = B.sum()/m.sum()
        A = (A - muA)*m
        B = (B - muB)*m
        denom = (np.sqrt((A*A).sum()) * np.sqrt((B*B).sum()) + 1e-8)
        return float((A*B).sum() / denom)
    else:
        A -= A.mean(); B -= B.mean()
        denom = (np.linalg.norm(A) * np.linalg.norm(B) + 1e-8)
        return float((A*B).sum() / denom)

def classify_array(bgr, proto_pos, proto_neg, roi_mask, thr=0.10, debug_name=None):
    """
    riceives img crop, computes LLR and prints results 
    return (present, llr, s_pos, s_neg).
    """
    if bgr is None:
        print("[SKIP] null img")
        return None, None, None, None

    g = preprocess(bgr)
    H, W = g.shape[:2]

    # adaptation ROI and prototipes to crop's resolution
    roi_rs = cv2.resize(roi_mask, (W, H), interpolation=cv2.INTER_NEAREST)
    pp = cv2.resize(proto_pos, (W, H), interpolation=cv2.INTER_AREA)
    pn = cv2.resize(proto_neg, (W, H), interpolation=cv2.INTER_AREA)

    s_pos = abs(ncc(g, pp, mask=roi_rs))
    s_neg = ncc(g, pn, mask=roi_rs)
    llr = s_pos - s_neg
    present = llr >= thr

    print(f"[{'OK' if present else 'NO'}] LLR={llr:.3f} (pos={s_pos:.3f} neg={s_neg:.3f})")

    # opzionalenot necessry: saves img with LLR
    if SAVE_DEBUG_VIS and debug_name is not None:
        vis = bgr.copy()
        color = (0,255,0) if present else (0,0,255)
        cv2.putText(vis, f"LLR={llr:.3f}",
                    (5, max(18,H-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)
        ensure_dir(OUT_DIR)
        out_path = os.path.join(OUT_DIR, f"llr_{debug_name}.png")
        cv2.imwrite(out_path, vis)
        print(f"[DBG] Salvata immagine LLR -> {out_path}")

    return present, llr, s_pos, s_neg

# ==========================
# CALLBACK SCATTO + LLR
# ==========================
def make_scatta_foto(picam2, proto_pos, proto_neg, roi_mask):
    def _cb():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"shot_{ts}.{SAVE_EXT}"
        filepath = os.path.join(SAVE_DIR, filename)
        try:
            print(f"[INFO] Scatto -> {filepath}")
            picam2.capture_file(filepath)

            # Crop immediato
            crop, crop_path = crop_image(filepath, save=True)

            if crop is None:
                return

            # Riconoscimento errori con LLR sul crop
            debug_name = os.path.splitext(os.path.basename(crop_path or filename))[0]
            classify_array(crop, proto_pos, proto_neg, roi_mask, thr=THR, debug_name=debug_name)

        except Exception as e:
            print(f"[ERR] Errore durante lo scatto: {e}")
    return _cb

# ==========================
# MAIN
# ==========================
def main():
    # Cartelle
    ensure_dir(SAVE_DIR)
    ensure_dir(CROP_DIR)
    ensure_dir(OUT_DIR)

    # --- Carica ROI per LLR ---
    roi_mask = cv2.imread(ROI_MASK_PATH, cv2.IMREAD_GRAYSCALE)
    if roi_mask is None:
        raise FileNotFoundError(f"ROI mask non trovata: {ROI_MASK_PATH}")

    # --- Carica dataset training pos/neg e crea prototipi ---
    pos_stack, _ = load_imgs(POS_DIR)
    neg_stack, _ = load_imgs(NEG_DIR)
    if pos_stack is None or neg_stack is None:
        raise RuntimeError("Metti almeno una immagine in POS_DIR e in NEG_DIR")

    proto_pos = np.median(pos_stack, axis=0).astype(np.uint8)
    proto_neg = np.median(neg_stack, axis=0).astype(np.uint8)
    print("[OK] Prototipi LLR caricati.")

    # --- Inizializza camera ---
    picam2 = Picamera2()
    still_cfg = picam2.create_still_configuration(
        main={"size": (CAM_WIDTH, CAM_HEIGHT), "format": "XRGB8888"}
    )
    picam2.configure(still_cfg)
    picam2.start()
    time.sleep(0.3)

    # --- Inizializza pulsante ---
    button = Button(BUTTON_PIN, pull_up=True, bounce_time=BUTTON_BOUNCE)
    button.when_pressed = make_scatta_foto(picam2, proto_pos, proto_neg, roi_mask)

    print("[OK] Pronto. Premi il microinterruttore per SCATTO + LLR (CTRL+C per uscire).")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()
        print("\n[OK] Uscita.")

if __name__ == "__main__":
    main()

