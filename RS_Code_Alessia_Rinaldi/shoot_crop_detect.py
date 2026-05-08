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
OUT_DIR = os.path.join(SAVE_DIR, "llr_debug")

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
POS_DIR = "/home/aless/llr/pos"
NEG_DIR = "/home/aless/llr/neg"
ROI_MASK_PATH = "/home/aless/llr/mask_soft.png"

THR = -0.035
SAVE_DEBUG_VIS = True

# ==========================
# CONFIGURATION WEIGHTED MASK
# ==========================
EPS = 1e-8

# PREDISPOSIZIONE:
# 1.0 = pesi lineari: 0 nero, 0.5 grigio, 1 bianco ->  0 = ignora; 255 = massimo peso; valori intermedi = peso intermedio  
# >1.0 = enfatizza di più solo le zone molto chiare
# <1.0 = dà più importanza anche alle zone grigie
MASK_GAMMA = 1.0

# Somma minima dei pesi per evitare ROI troppo piccole, valutare se tenere
MIN_ROI_WEIGHT = 10.0


# ==========================
# UTILS CROP
# ==========================
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def clamp_roi(x, y, w, h, W, H):
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    w = max(1, min(int(w), W - x))
    h = max(1, min(int(h), H - y))
    return x, y, w, h


def roi_from_mode(W, H):
    if ROI_MODE == "relative":
        rx, ry, rw, rh = ROI_REL
        x = int(rx * W)
        y = int(ry * H)
        w = int(rw * W)
        h = int(rh * H)
    else:
        x, y, w, h = ROI_PIX

    return clamp_roi(x, y, w, h, W, H)

# Legge immagine da disco, fa il crop in base alla ROI e, opzionalmente, salva il ritaglio in CROP_DIR
def crop_image(img_path, save=True):

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    if img is None:
        print(f"[WARN] Impossible reading {img_path}")
        return None, None

    H, W = img.shape[:2]
    x, y, w, h = roi_from_mode(W, H)

    crop = img[y:y + h, x:x + w]

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
# Come in build_mean_diff_mask.py
def preprocess(bgr):
    # conversione in scala di grigi se immagine BGR
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr

    # riduzione del rumore ad alta frequenza
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # stabilizza il contrasto su immagini piccole
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)

    return gray

# Predisposizione per futuro
def load_imgs(folder):
    paths = []

    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        paths += glob.glob(os.path.join(folder, ext))

    paths = sorted(paths)
    imgs = []

    for p in paths:
        im = cv2.imread(p, cv2.IMREAD_COLOR)

        if im is None:
            continue

        imgs.append(preprocess(im))

    return (np.stack(imgs, 0) if imgs else None), paths

# mask 0 = peso 0 = ignora; mask 255 = peso 1 = massimo peso; valori intermedi = peso intermedio
def normalize_weight_mask(mask, gamma=1.0):
    
    m = mask.astype(np.float32)

    max_val = m.max()

    if max_val <= 0:
        return m

    m = m / max_val

    if gamma != 1.0:
        m = np.power(m, gamma)

    return m

# Normalized Cross-Correlation tra A e B, se mask è none calcola la NCC classica su tutta l'immagine
def ncc(A, B, mask=None, min_roi_weight=10.0, mask_gamma=1.0):
    
    A = A.astype(np.float32)
    B = B.astype(np.float32)

    if A.shape != B.shape:
        raise ValueError(f"A and B must have the same shape, actually {A.shape} and {B.shape}")

    if mask is not None:
        if mask.shape != A.shape:
            raise ValueError(f"mask and img must have the same shape, actually {mask.shape} and {A.shape}")

        #maschera pesata tra 0 e 1
        m = normalize_weight_mask(mask, gamma=mask_gamma)

        #somma dei pesi: equivalente "pesato" del numero di pixel nella ROI
        wsum = m.sum()

        if wsum < min_roi_weight:
            return 0.0

        #media pesata nella ROI
        muA = (m * A).sum() / wsum
        muB = (m * B).sum() / wsum

        #rimozione della media
        A0 = A - muA
        B0 = B - muB

        #numeratore pesato
        num = (m * A0 * B0).sum()

        #denominatore pesato
        normA = np.sqrt((m * A0 * A0).sum())
        normB = np.sqrt((m * B0 * B0).sum())

        denom = normA * normB + EPS

        return float(num / denom)

    else:
        #NCC classica senza maschera
        A0 = A - A.mean()
        B0 = B - B.mean()

        denom = np.linalg.norm(A0) * np.linalg.norm(B0) + EPS

        return float((A0 * B0).sum() / denom)

# CLASSIFICATORE: riceve crop, calcola LLR, stampa
def classify_array(bgr, proto_pos, proto_neg, roi_mask, thr=0.10, debug_name=None):
    
    if bgr is None:
        print("[SKIP] null img")
        return None, None, None, None

    g = preprocess(bgr)

    H, W = g.shape[:2]

    # IMPORTANTE:
    # con una maschera pesata uso INTER_LINEAR, non INTER_NEAREST, così se la maschera ha sfumature, le mantiene.
    roi_rs = cv2.resize(roi_mask, (W, H), interpolation=cv2.INTER_LINEAR)

    pp = cv2.resize(proto_pos, (W, H), interpolation=cv2.INTER_AREA)
    pn = cv2.resize(proto_neg, (W, H), interpolation=cv2.INTER_AREA)

    # NCC pesata nella ROI soft
    s_pos = ncc(
        g,
        pp,
        mask=roi_rs,
        min_roi_weight=MIN_ROI_WEIGHT,
        mask_gamma=MASK_GAMMA
    )

    s_neg = ncc(
        g,
        pn,
        mask=roi_rs,
        min_roi_weight=MIN_ROI_WEIGHT,
        mask_gamma=MASK_GAMMA
    )

    # LLR
    llr = s_pos - s_neg

    # Decisione finale
    present = llr >= thr

    print(f"[{'OK' if present else 'NO'}] LLR={llr:.3f} (pos={s_pos:.3f} neg={s_neg:.3f})")

    #visualizzazione debug
    if SAVE_DEBUG_VIS and debug_name is not None:
        vis = bgr.copy()

        color = (0, 255, 0) if present else (0, 0, 255)

        # Per disegnare il contorno uso una versione binaria della ROI
        roi_bin = (roi_rs > 0).astype(np.uint8)

        cnts, _ = cv2.findContours(
            roi_bin,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(vis, cnts, -1, color, 1)

        cv2.putText(
            vis,
            f"LLR={llr:.3f}",
            (5, max(18, H - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2,
            cv2.LINE_AA
        )

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

            classify_array(
                crop,
                proto_pos,
                proto_neg,
                roi_mask,
                thr=THR,
                debug_name=debug_name
            )

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
        raise FileNotFoundError(f"ROI mask non found: {ROI_MASK_PATH}")

    # --- Carica dataset training pos/neg e crea prototipi ---
    pos_stack, _ = load_imgs(POS_DIR)
    neg_stack, _ = load_imgs(NEG_DIR)

    if pos_stack is None or neg_stack is None:
        raise RuntimeError("you need at least1 img in  POS_DIR and in NEG_DIR")

    # Prototipi robusti tramite mediana
    proto_pos = np.median(pos_stack, axis=0).astype(np.uint8)
    proto_neg = np.median(neg_stack, axis=0).astype(np.uint8)

    print("[OK] LLR loaded.")
    print(f"[OK] soft_mask loaded: {ROI_MASK_PATH}")
    print(f"[INFO] THR={THR}, MASK_GAMMA={MASK_GAMMA}, MIN_ROI_WEIGHT={MIN_ROI_WEIGHT}")

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

    button.when_pressed = make_scatta_foto(
        picam2,
        proto_pos,
        proto_neg,
        roi_mask
    )

    print("[OK] Ready. Press microswitch (CTRL+C exit).")

    try:
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        pass

    finally:
        picam2.stop()
        print("\n[OK] Exit.")


if __name__ == "__main__":
    main()