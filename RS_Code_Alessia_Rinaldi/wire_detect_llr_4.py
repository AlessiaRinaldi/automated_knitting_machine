# scopo: LLR = NCC(img, proto_pos) - NCC(img, proto_neg) dentro ROI pesata
# Se LLR >= thr allora "filo presente"
#
# PER COMPILARE:
# cd progetto
# python wire_detect_llr.py --pos .\data\pos --neg .\data\neg --roi .\roi_mask.png --dir .\test --thr 0.0

import os, glob, argparse
import numpy as np
import cv2

EPS = 1e-8


def preprocess(bgr):
    # converte in scala di grigi se l'immagine è BGR
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr

    # riduce rumore ad alta frequenza
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # stabilizza il contrasto su immagini piccole
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)

    return gray


def load_imgs(folder):
    # predisposizione: non solo per .png
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        paths += glob.glob(os.path.join(folder, ext))

    paths = sorted(paths)
    imgs = []

    # pre-process per ogni immagine
    for p in paths:
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None:
            continue
        imgs.append(preprocess(im))

    return (np.stack(imgs, 0) if imgs else None), paths


def normalize_weight_mask(mask, gamma=1.0):
    """
    Converte la maschera in pesi float tra 0 e 1.

    mask = 0   -> peso 0, pixel ignorato
    mask = 255 -> peso 1, pixel molto importante
    valori intermedi -> peso intermedio

    gamma:
        gamma = 1.0 mantiene i pesi lineari
        gamma > 1.0 rende più importanti solo le zone molto chiare
        gamma < 1.0 rende più importanti anche le zone grigie
    """

    m = mask.astype(np.float32)

    max_val = m.max()
    if max_val <= 0:
        return m

    m = m / max_val

    if gamma != 1.0:
        m = np.power(m, gamma)

    return m


def ncc(A, B, mask=None, min_roi_weight=10.0, mask_gamma=1.0):
    """
    Normalized Cross-Correlation tra A e B.

    Se mask è None:
        calcola la NCC classica su tutta l'immagine.

    Se mask è presente:
        calcola una NCC pesata, dove i valori della maschera indicano
        quanto ogni pixel deve contribuire al risultato.
    """

    A = A.astype(np.float32)
    B = B.astype(np.float32)

    if A.shape != B.shape:
        raise ValueError(f"A e B devono avere la stessa shape. Trovato {A.shape} e {B.shape}")

    if mask is not None:
        if mask.shape != A.shape:
            raise ValueError(f"mask e immagini devono avere la stessa shape. Trovato {mask.shape} e {A.shape}")

        # maschera pesata tra 0 e 1
        m = normalize_weight_mask(mask, gamma=mask_gamma)

        # somma dei pesi: equivalente "pesato" del numero di pixel nella ROI
        wsum = m.sum()

        # controllo per non avere ROI troppo piccole o praticamente vuote
        if wsum < min_roi_weight:
            return 0.0

        # media pesata solo nella ROI
        muA = (m * A).sum() / wsum
        muB = (m * B).sum() / wsum

        # rimuove la media
        A0 = A - muA
        B0 = B - muB

        # numeratore pesato
        num = (m * A0 * B0).sum()

        # denominatore pesato
        normA = np.sqrt((m * A0 * A0).sum())
        normB = np.sqrt((m * B0 * B0).sum())
        denom = normA * normB + EPS

        return float(num / denom)

    else:
        # NCC classica senza maschera
        A0 = A - A.mean()
        B0 = B - B.mean()

        denom = np.linalg.norm(A0) * np.linalg.norm(B0) + EPS

        return float((A0 * B0).sum() / denom)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--pos", required=True)
    ap.add_argument("--neg", required=True)
    ap.add_argument("--roi", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--thr", type=float, default=0.10)
    ap.add_argument("--outdir", default=None)

    # nuovi parametri opzionali per la maschera pesata
    ap.add_argument(
        "--mask-gamma",
        type=float,
        default=1.0,
        help="Esponente applicato alla maschera pesata. 1.0 = lineare, >1 enfatizza zone chiare."
    )

    ap.add_argument(
        "--min-roi-weight",
        type=float,
        default=10.0,
        help="Somma minima dei pesi della ROI per considerare valido il calcolo NCC."
    )

    args = ap.parse_args()

    # maschera in scala di grigi
    # Ora i livelli di grigio sono importanti:
    # 0 = ignora
    # 255 = massimo peso
    # valori intermedi = peso intermedio
    roi = cv2.imread(args.roi, cv2.IMREAD_GRAYSCALE)

    if roi is None:
        raise FileNotFoundError(args.roi)

    # carica dataset di training
    pos_stack, _ = load_imgs(args.pos)
    neg_stack, _ = load_imgs(args.neg)

    if pos_stack is None or neg_stack is None:
        raise RuntimeError("Put examples in --pos and in --neg")

    # prototipi: mediana robusta
    proto_pos = np.median(pos_stack, axis=0).astype(np.uint8)
    proto_neg = np.median(neg_stack, axis=0).astype(np.uint8)

    # immagini di test
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        paths += glob.glob(os.path.join(args.dir, ext))

    paths = sorted(paths)

    if not paths:
        raise RuntimeError(f"No images in {args.dir}")

    # cartella di output
    outdir = args.outdir or args.dir
    os.makedirs(outdir, exist_ok=True)

    for p in paths:
        # pre-processa immagine di test
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)

        if bgr is None:
            print("[SKIP]", p)
            continue

        g = preprocess(bgr)

        # adatta ROI e prototipi se risoluzione diversa
        H, W = g.shape[:2]

        # IMPORTANTE:
        # con una maschera pesata uso INTER_LINEAR, non INTER_NEAREST.
        # Così, se la maschera ha sfumature, le mantiene.
        roi_rs = cv2.resize(roi, (W, H), interpolation=cv2.INTER_LINEAR)

        pp = cv2.resize(proto_pos, (W, H), interpolation=cv2.INTER_AREA)
        pn = cv2.resize(proto_neg, (W, H), interpolation=cv2.INTER_AREA)

        # Calcolo punteggi di correlazione nella ROI pesata
        s_pos = ncc(
            g,
            pp,
            mask=roi_rs,
            min_roi_weight=args.min_roi_weight,
            mask_gamma=args.mask_gamma
        )

        s_neg = ncc(
            g,
            pn,
            mask=roi_rs,
            min_roi_weight=args.min_roi_weight,
            mask_gamma=args.mask_gamma
        )

        # Log-likelihood ratio semplificato
        llr = s_pos - s_neg

        # decisione finale
        present = llr >= args.thr

        # visualizzazione debug
        vis = bgr.copy()
        color = (0, 255, 0) if present else (0, 0, 255)

        # Per disegnare il contorno uso ancora una versione binaria della ROI.
        # Questo serve solo per visualizzare la zona considerata.
        roi_bin = (roi_rs > 0).astype(np.uint8)
        cnts, _ = cv2.findContours(
            roi_bin,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(vis, cnts, -1, color, 2)

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

        base = os.path.splitext(os.path.basename(p))[0]
        cv2.imwrite(os.path.join(outdir, f"out_{base}.png"), vis)

        print(
            f"[{'OK' if present else 'NO'}] "
            f"{os.path.basename(p)}  "
            f"LLR={llr:.3f} "
            f"(pos={s_pos:.3f} neg={s_neg:.3f})"
        )


if __name__ == "__main__":
    main()