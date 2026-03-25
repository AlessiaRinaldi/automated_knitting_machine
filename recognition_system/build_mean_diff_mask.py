# python build_mean_diff_mask.py --pos ./pos --neg ./neg --out ./mask_out_mean   
# la maschera migliore l'ho ottenuta facendo:
# python build_mean_diff_mask.py --pos ./pos__mask --neg ./neg_mask --out ./mask_out_mean
# questo perchè ha il contrasto migliore rispetto agli altri colori di filato 

import os, glob, argparse
import numpy as np
import cv2

H_IMG, W_IMG = 76, 76      # dimensione attesa delle immagini
ROI_H = 40                 # altezza rettangolo in basso 

def preprocess(bgr_or_gray, use_clahe=True):
    # to gray
    if bgr_or_gray.ndim == 3:
        g = cv2.cvtColor(bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        g = bgr_or_gray.copy()
    # riduzione rumore
    g = cv2.GaussianBlur(g, (3,3), 0)
    # equalizzazione locale per preservare ombre/texture
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        g = clahe.apply(g)
    return g

def load_stack(folder, target_hw=(H_IMG, W_IMG)):
    paths = sorted(glob.glob(os.path.join(folder, "*")))
    imgs = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        # uniforma canali/dimensione
        img = preprocess(img)
        if img.shape[:2] != target_hw:
            img = cv2.resize(img, target_hw[::-1], interpolation=cv2.INTER_AREA)
        imgs.append(img.astype(np.float32))
    if not imgs:
        raise RuntimeError(f"Nessuna immagine valida in {folder}")
    stack = np.stack(imgs, axis=0)  # [N, H, W]
    return stack, paths

def bottom_roi_mask(h=H_IMG, w=W_IMG, roi_h=ROI_H):
    """Maschera binaria 0/1 con 1 solo nel rettangolo in basso alto roi_h e largo w."""
    m = np.zeros((h, w), dtype=np.float32)
    y0 = max(0, h - roi_h)
    m[y0:h, 0:w] = 1.0
    return m

def build_mask(pos_stack, neg_stack, bilateral=True):
    # statistiche per-pixel
    mu_pos = pos_stack.mean(axis=0)
    mu_neg = neg_stack.mean(axis=0)
    std_pos = pos_stack.std(axis=0)
    std_neg = neg_stack.std(axis=0)

    # fisher-like score
    eps = 1e-6
    S = np.abs(mu_pos - mu_neg) / (std_pos + std_neg + eps)

    # normalizza in [0,1]
    S -= S.min()
    if S.max() > 0:
        S /= S.max()

    # applica ROI: tieni solo il rettangolo in basso 55x76, sopra nero
    roi = bottom_roi_mask(h=S.shape[0], w=S.shape[1], roi_h=ROI_H)
    S = S * roi

    # edge-preserving per non perdere texture (solo dentro ROI)
    if bilateral:
        # per non sporcare fuori ROI, filtra solo la regione e ricompone
        S8 = (S * 255.0).astype(np.uint8)
        # bilateral su tutta l'immagine è ok perché fuori ROI è già zero
        S8 = cv2.bilateralFilter(S8, d=5, sigmaColor=15, sigmaSpace=5)
        S = S8.astype(np.float32) / 255.0
        S *= roi  

    # Hard mask via Otsu (solo nella ROI, fuori ROI 0)
    S8 = (S*255.0).astype(np.uint8)
    # applica Otsu direttamente: fuori ROI 0 quindi nero
    thr, hard = cv2.threshold(S8, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    hard = (hard.astype(np.float32) * roi).astype(np.uint8)  # garantisci nero fuori ROI

    return S, hard, thr, mu_pos, mu_neg

def save_masks(S, hard, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    soft_path = os.path.join(out_dir, "mask_soft.png")
    hard_path = os.path.join(out_dir, "mask_hard_otsu.png")
    cv2.imwrite(soft_path, (S*255.0).astype(np.uint8))
    cv2.imwrite(hard_path, hard)
    return soft_path, hard_path

def masked_ncc(img, template, mask_float01):
    """
    NCC pesata da maschera (valori 0..1).
    img e template devono già essere preprocessati e della stessa size.
    """
    g = img.astype(np.float32)
    t = template.astype(np.float32)
    w = mask_float01.astype(np.float32)

    # centra rispetto a media pesata
    def wmean(x, w):
        return (x*w).sum() / (w.sum() + 1e-6)

    gm = g - wmean(g, w)
    tm = t - wmean(t, w)

    num = (w * gm * tm).sum()
    den = np.sqrt((w * gm*gm).sum()) * np.sqrt((w * tm*tm).sum()) + 1e-6
    return float(num / den)

def main():
    ap = argparse.ArgumentParser(description="Maschera discriminativa da pos/neg (76x76) solo nel rettangolo in basso 55x76.")
    ap.add_argument("--pos", required=True)
    ap.add_argument("--neg", required=True)
    ap.add_argument("--out", default="./mask_out")
    ap.add_argument("--no-bilateral", action="store_true")
    args = ap.parse_args()

    pos_stack, _ = load_stack(args.pos, target_hw=(H_IMG, W_IMG))
    neg_stack, _ = load_stack(args.neg, target_hw=(H_IMG, W_IMG))

    S, hard, thr, mu_pos, mu_neg = build_mask(pos_stack, neg_stack, bilateral=(not args.no_bilateral))
    soft_path, hard_path = save_masks(S, hard, args.out)

    print(f"[OK] Soft mask salvata in: {soft_path}")
    print(f"[OK] Hard mask (Otsu) salvata in: {hard_path} (threshold Otsu={thr:.1f})")

    
    test_img = neg_stack[0]  
    ncc_pos = masked_ncc(test_img, mu_pos, S)  # usa soft mask come pesi (solo nel rettangolo)
    ncc_neg = masked_ncc(test_img, mu_neg, S)
    llr_like = ncc_pos - ncc_neg
    print(f"Esempio LLR-like su una immagine di test: NCC_pos={ncc_pos:.3f}, NCC_neg={ncc_neg:.3f}, diff={llr_like:.3f}")

    # Salva anche la ROI binaria per ispezione
    roi_bin = (bottom_roi_mask(H_IMG, W_IMG, ROI_H) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(args.out, "roi_bottom_55x76.png"), roi_bin)

if __name__ == "__main__":
    main()
