#!/usr/bin/env python3
"""
THE slicing algorithm of this repository (optional workflow steps 4 and 6).

Slice images into square patches, score each patch with the trained 1D
projection-band model (workspace file: 2.1_best_projection_band_2.keras,
input = 256-length edge-projection band, output = P(patch shows horizontal
text lines)), and save the lowest- or highest-scoring patch per image until
a limit is reached.  Images whose best patch is vertical, or whose best
probability stays under --min-prob-text, are routed to the review folder
instead of the output folder.

Where it sits in the workflow
-----------------------------
* OPTIONAL STEP 4 : slice the (jpg-converted) RVL dataset  -> training data.
* OPTIONAL STEP 6 : slice the step-5 sample                -> training data.
* STEP 8          : 2.2_0_mirror_text_slices.py imports this very module and
  re-uses its functions to build the "mirror" tree of the flattened corpus
  that the rotation methods (steps 9-11) operate on.  Behaviour differences
  of the mirror script (keep original basenames, skip instead of review,
  no max-save) are implemented there; this file stays the single source of
  truth for the slicing/scoring algorithm.

Multithreaded CPU preprocessing (ThreadPoolExecutor, ops pinned to /CPU:0)
with a worker per CPU core, plus optional resume-from-checkpoint support:

Resume usage:  --resume-from "<basename of last fully processed image>" --resume-count <N>
skips everything up to and including that image in the seed-shuffled order and restores
saved_count=N. Omit --resume-from to run fresh from the beginning.
"""

import os
import argparse
from pathlib import Path
import concurrent.futures
import multiprocessing

import numpy as np
import cv2
import tensorflow as tf

from workflow_common import load_keras_model, resolve_model_path

# -------------------- ARGS --------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=None,
                    help="Root folder containing input images (searched recursively). "
                         "Prompted for when omitted.")
    ap.add_argument("--model", type=str, default="2.1_best_projection_band_2.keras",
                    help="Path to the trained band-based textline detector. Default: the "
                         "model shipped in this workspace (found next to the scripts).")
    ap.add_argument("--dst", type=str, default=None,
                    help="Destination folder to save selected patches. Prompted for "
                         "when omitted.")
    ap.add_argument("--pick", choices=["lowest", "highest"], default="highest",
                    help="Choose whether to save the patch with the lowest or highest prediction score.")
    ap.add_argument("--max-save", type=int, default=400000,
                    help="Maximum number of patches to save. 0 or negative means no limit.")
    ap.add_argument("--side-frac", type=float, default=0.25,
                    help="Side length of each square as a fraction of min(H, W). Default 0.25.")
    ap.add_argument("--seed", type=int, default=37,
                    help="Random seed for shuffling input images and target line sampling.")
    ap.add_argument("--exts", nargs="+",
                    default=[".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
                             ".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF", ".WEBP"],
                    help="Image extensions to include.")
    ap.add_argument("--min-prob-text", type=float, default=0.57,
                    help="Minimum predicted probability for a patch to be considered text.")
    ap.add_argument("--review-dst", type=str, default=None,
                    help="Folder to save originals of images where no patch reaches "
                         "min-prob-text. Default: <dst>/_review.")
    ap.add_argument("--resume-from", type=str, default=None,
                    help="OPTIONAL resume mode: basename of the last image fully processed "
                         "in the interrupted run. Everything up to and including it (in the "
                         "seed-shuffled order) is skipped. Omit to disable resume and run fresh.")
    ap.add_argument("--resume-count", type=int, default=0,
                    help="Patches already saved before the checkpoint; used as the initial "
                         "saved_count when resuming. Only meaningful with --resume-from.")
    return ap.parse_args()

# -------------------- PREPROCESS PIPELINE --------------------

def standardize_1d(x, eps=1e-6):
    x = tf.cast(x, tf.float32)
    mean = tf.reduce_mean(x)
    std  = tf.math.reduce_std(x)
    return x

def fft_band_variance_5_50(profile_1d):
    profile_1d = tf.cast(profile_1d, tf.float32)         
    fft = tf.signal.rfft(profile_1d)                     
    mag = tf.abs(fft)                                    
    band = mag[5:51]                                     
    var = tf.math.reduce_variance(band)
    return var

def image_to_projection_sum_band_from_gray_np(gray_np_uint8):
    # Force preprocessing to use CPU to prevent multithreading GPU lockups
    with tf.device('/CPU:0'):
        gray_f = gray_np_uint8.astype(np.float32) / 255.0
        gray_tf = tf.convert_to_tensor(gray_f, dtype=tf.float32)
        gray_tf = tf.expand_dims(gray_tf, axis=-1)  

        gray = tf.image.resize(gray_tf, (256, 256), antialias=True) 
        gray = tf.squeeze(gray, axis=-1)                            
        gray_uint8 = tf.clip_by_value(gray * 255.0, 0, 255)
        gray_uint8 = tf.cast(gray_uint8, tf.uint8)

        def adaptive_gaussian_and_canny(gray_np):
            binary = cv2.adaptiveThreshold(
                gray_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 11
            )
            edges = cv2.Canny(binary, 50, 200)
            return (edges.astype(np.float32) / 255.0)

        edges = adaptive_gaussian_and_canny(gray_uint8.numpy())      
        edges_tf = tf.convert_to_tensor(edges, dtype=tf.float32)     

        vert  = tf.reduce_sum(edges_tf, axis=0)  
        horiz = tf.reduce_sum(edges_tf, axis=1)  

        var_vert  = fft_band_variance_5_50(vert)
        var_horiz = fft_band_variance_5_50(horiz)

        is_horiz_higher = var_horiz > var_vert

        chosen_profile = tf.cond(
            is_horiz_higher,
            lambda: horiz,
            lambda: vert
        )  

        chosen_z = standardize_1d(chosen_profile)  
        band = tf.expand_dims(chosen_z, axis=-1)   
        
        return band, is_horiz_higher

# -------------------- IMAGE / PATCH UTILS --------------------

def read_rgb(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def save_png_bgr(path: Path, bgr: np.ndarray):
    ok, buf = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if ok:
        buf.tofile(str(path))

def slice_into_squares(img: np.ndarray, side: int):
    H, W = img.shape[:2]
    if side <= 0:
        return []

    n_h = H // side
    n_w = W // side
    patches = []

    for iy in range(n_h):
        y0 = iy * side
        for ix in range(n_w):
            x0 = ix * side
            patch = img[y0:y0+side, x0:x0+side, :]
            patches.append((y0, x0, patch))
    return patches

# -------------------- MULTITHREADING PREP FUNCTION --------------------

def preprocess_single_image(img_path, root, side_frac):
    """Handles the CPU-heavy loading, slicing, and FFT generation for a single image."""
    rgb = read_rgb(img_path)
    if rgb is None:
        return None

    rel_path = img_path.relative_to(root)
    H, W = rgb.shape[:2]
    min_hw = min(H, W)
    side = int(round(side_frac * float(min_hw)))
    side = max(1, side)

    bgr_full = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    patches = slice_into_squares(rgb, side)
    if not patches:
        return None

    bands = []
    is_horiz_flags = []
    patch_bgrs = []
    coords = []

    for (y0, x0, patch_rgb) in patches:
        coords.append((y0, x0))
        patch_bgr = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR)
        patch_bgrs.append(patch_bgr)

        gray_patch = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)  
        band, is_horiz = image_to_projection_sum_band_from_gray_np(gray_patch)  
        
        bands.append(band)
        is_horiz_flags.append(is_horiz.numpy())

    return {
        'img_path': img_path,
        'rel_path': rel_path,
        'bgr_full': bgr_full,
        'bands': bands,
        'is_horiz_flags': is_horiz_flags,
        'patch_bgrs': patch_bgrs,
        'coords': coords,
        'side': side,
        'H': H, 'W': W, 'min_hw': min_hw
    }

# -------------------- MAIN LOGIC --------------------

def main():
    args = parse_args()

    if not args.root:
        args.root = input("Root folder containing the images to slice: ").strip().strip('"').strip("'")
    if not args.dst:
        args.dst = input("Destination folder for the selected text patches: ").strip().strip('"').strip("'")
    if not args.root or not args.dst:
        raise SystemExit("[Error] Both --root and --dst are required.")

    root = Path(args.root)
    dst  = Path(args.dst)
    
    if args.review_dst is not None:
        review_root = Path(args.review_dst)
    else:
        review_root = dst / "_review"

    tf.keras.utils.set_random_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    exts = set(args.exts)
    image_paths = [p for p in root.rglob("*") if p.is_file() and p.suffix in exts]

    if not image_paths:
        raise SystemExit(f"[Error] No images found under {root}")

    # Randomize order (Must remain identical to original run)
    rng.shuffle(image_paths)

    print(f"[Info] Found {len(image_paths)} images.")
    
    # -------------------- RESUME LOGIC (enabled via --resume-from) --------------------
    if args.resume_from:
        resume_idx = -1
        for i, p in enumerate(image_paths):
            if p.name == args.resume_from:
                resume_idx = i
                break

        if resume_idx != -1:
            print(f"[Info] Found resume target '{args.resume_from}' at index {resume_idx}. "
                  f"Resuming from next image with saved_count={args.resume_count}...")
            image_paths = image_paths[resume_idx + 1:]
            saved_count = args.resume_count
        else:
            print(f"[Warn] Resume target '{args.resume_from}' not found. Starting from beginning.")
            saved_count = 0
    else:
        print("[Info] Resume disabled (--resume-from not set). Starting from beginning.")
        saved_count = 0
    # ----------------------------------------------------------------------------------

    print(f"[Info] Loading model from: {resolve_model_path(args.model)}")
    model = load_keras_model(args.model)
    print("[Info] Model loaded.")

    max_save = args.max_save
    num_cores = multiprocessing.cpu_count()
    print(f"[Info] Utilizing {num_cores} CPU cores for preprocessing.")

    # Process in chunks to keep memory usage low while mapping threads
    chunk_size = num_cores * 4 

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_cores) as executor:
        for i in range(0, len(image_paths), chunk_size):
            if max_save > 0 and saved_count >= max_save:
                print(f"[Info] Reached max-save={max_save}. Stopping.")
                break

            chunk_paths = image_paths[i : i + chunk_size]
            
            # executor.map ensures results are returned in the exact same order as chunk_paths
            results = executor.map(lambda p: preprocess_single_image(p, root, args.side_frac), chunk_paths)

            for res in results:
                if max_save > 0 and saved_count >= max_save:
                    break
                    
                if res is None:
                    continue

                # -------------------- PREDICT (Main Thread) --------------------
                band_batch = tf.stack(res['bands'], axis=0)
                probs = model.predict(band_batch, verbose=0).ravel()  

                if args.pick == "lowest":
                    best_idx = int(np.argmin(probs))
                else:  
                    best_idx = int(np.argmax(probs))

                best_prob = float(probs[best_idx])
                best_is_horiz = res['is_horiz_flags'][best_idx]
                stem = res['img_path'].stem

                # -------------------- THRESHOLD & DIRECTION ROUTING --------------------
                needs_review = False
                review_reason = ""

                if not best_is_horiz:
                    needs_review = True
                    review_reason = "VERT_VAR_HIGHER"
                elif args.pick == "highest" and best_prob < args.min_prob_text:
                    needs_review = True
                    review_reason = f"LOW_PROB_p{best_prob:.4f}"

                if needs_review:
                    review_name = f"{stem}__NOT_TEXT_{review_reason}.png"
                    review_path = review_root / res['rel_path'].parent / review_name
                    review_path.parent.mkdir(parents=True, exist_ok=True) 
                    
                    save_png_bgr(review_path, res['bgr_full'])
                    print(
                        f"[Review] Routed {res['img_path'].name} to review ({review_reason}). "
                        f"Saved to: {review_path.parent.name}/{review_name}"
                    )
                    continue

                # -------------------- ADJUST PATCH SIDE -----------------
                best_band = res['bands'][best_idx].numpy().squeeze()  
                if best_band.ndim != 1 or best_band.size != 256:
                    M = None
                else:
                    fft = np.fft.rfft(best_band)          
                    mag = np.abs(fft)                     
                    sub_mag = mag[5:51]
                    
                    if np.all(sub_mag == 0):
                        M = None  
                    else:
                        peak_rel = int(np.argmax(sub_mag))
                        M = 5 + peak_rel                   

                y0_best, x0_best = res['coords'][best_idx]
                side_target = res['side']
                y0_target = y0_best
                x0_target = x0_best

                if M is not None and M > 0:
                    T = 6
                    scale = T / float(M)
                    new_side = int(round(res['side'] * scale))
                    new_side = max(4, new_side)
                    new_side = min(new_side, res['min_hw'])

                    y_center = y0_best + res['side'] // 2
                    x_center = x0_best + res['side'] // 2

                    side_final = new_side
                    side_final = min(side_final, res['H'], res['W'])

                    y0_new = int(np.clip(y_center - side_final // 2, 0, res['H'] - side_final))
                    x0_new = int(np.clip(x_center - side_final // 2, 0, res['W'] - side_final))

                    y0_target, x0_target, side_target = y0_new, x0_new, side_final
                
                y1_target = y0_target + side_target
                x1_target = x0_target + side_target
                y1_target = min(y1_target, res['H'])
                x1_target = min(x1_target, res['W'])

                final_patch_bgr = res['bgr_full'][y0_target:y1_target, x0_target:x1_target]

                if final_patch_bgr.size == 0:
                    final_patch_bgr = res['patch_bgrs'][best_idx]

                final_patch_resized = cv2.resize(final_patch_bgr, (256, 256), interpolation=cv2.INTER_AREA)

                out_name = f"{stem}__patch_p{best_prob:.4f}.png"
                out_path = dst / res['rel_path'].parent / out_name
                out_path.parent.mkdir(parents=True, exist_ok=True) 
                
                save_png_bgr(out_path, final_patch_resized)

                saved_count += 1
                print(
                    f"[{saved_count}] Saved patch from {res['img_path'].name} → {out_path.parent.name}/{out_name} "
                    f"(p={best_prob:.4f})"
                )

    print(f"[Done] Total patches saved: {saved_count}")

if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
            print(f"[Info] GPUs: {len(gpus)} (memory growth enabled)")
    except Exception as e:
        print(f"[Warn] GPU setup: {e}")

    main()