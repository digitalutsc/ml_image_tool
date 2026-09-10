#!/usr/bin/env python3

import os
# MUST be set at the very top before any imports to avoid Windows MAX_PATH limits
os.environ["NUMBA_CACHE_DIR"] = os.path.join(os.getcwd(), ".numba_cache")

import argparse
from pathlib import Path
import random
import numpy as np
import cv2
import concurrent.futures
import multiprocessing
from tqdm import tqdm

from augraphy import *

# ==========================================
# 1. ARGS & GLOBALS
# ==========================================

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=r"C:\D\UofT\DSU Paper\RVL_UP",
                    help="Root folder containing input images to augment.")
    ap.add_argument("--dst", type=str, default=r"C:\D\UofT\DSU Paper\RVL_UP_Augmented",
                    help="Destination folder to save augmented images.")
    ap.add_argument("--exts", nargs="+",
                    default=[".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
                             ".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF", ".WEBP"],
                    help="Image extensions to include.")
    return ap.parse_args()

# Moved to global so both the warm-up and the workers can access it cleanly
AUGRAPHY_CLASSES = [
    BadPhotoCopy, BindingsAndFasteners, BleedThrough,
    Brightness, BrightnessTexturize, ColorPaper,
    ColorShift, DelaunayTessellation, DepthSimulatedBlur,
    DirtyDrum, DirtyRollers, DirtyScreen,
    Dithering, DotMatrix, DoubleExposure,
    Faxify, Gamma, Hollow, InkBleed,
    InkMottling, LCDScreenPattern,
    Jpeg, LensFlare, Letterpress, LightingGradient,
    LinesDegradation, LowInkPeriodicLines, LowInkRandomLines,
    LowLightNoise, Moire, NoiseTexturize,
    NoisyLines, PatternGenerator, ReflectedLight,
    ShadowCast, SubtleNoise, VoronoiTessellation, 
    Folding, Geometric, 
    GlitchEffect, InkShifter, PageBorder, 
    Rescale, SectionShift, Squish
]

def read_bgr(path: Path):
    """Unicode-safe image read, returns BGR np.ndarray or None."""
    data = np.fromfile(str(path), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return bgr

def save_png_bgr(path: Path, bgr: np.ndarray):
    """Unicode-safe PNG write."""
    ok, buf = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if ok:
        buf.tofile(str(path))

# ==========================================
# 2. NUMBA WARM-UP FUNCTION
# ==========================================

def warmup_numba_cache():
    """Forces Numba to compile all augmentations in a safe, single thread before multiprocessing starts."""
    print("[Info] Warming up Numba JIT cache (This may take a minute or two, please wait)...")
    
    # Create a dummy image
    dummy_img = np.zeros((500, 500, 3), dtype=np.uint8)
    
    # Instantiate ALL classes with p=1.0 to force compilation
    warmup_augs = [cls(p=1.0) for cls in AUGRAPHY_CLASSES]
    
    pipeline = AugraphyPipeline(ink_phase=[], paper_phase=[], post_phase=warmup_augs)
    
    try:
        # Running this compiles all the math safely without race conditions
        pipeline.augment(dummy_img)
    except Exception as e:
        # Some augmentations might throw minor math errors on a pure black image, 
        # but the compilation will still succeed. We can safely ignore them here.
        pass 
        
    print("[Info] Cache warm-up complete! Multiprocessing is now safe.")

# ==========================================
# 3. WORKER FUNCTION
# ==========================================

def process_single_image(img_path, root, dst):
    # Prevent OpenCV thread deadlock inside multiprocessing
    cv2.setNumThreads(0)
    
    # Re-seed random for each process
    random.seed()
    np.random.seed()

    # Read image
    img_bgr = read_bgr(img_path)
    if img_bgr is None:
        return None, f"Failed to read {img_path.name}"

    rel_path = img_path.relative_to(root)
    out_name = f"{img_path.stem}_aug.png"
    out_path = dst / rel_path.parent / out_name
    
    # Select exactly 3 augmentations (class references)
    selected_classes = random.sample(AUGRAPHY_CLASSES, 2)
    
    # Instantiate ONLY the 3 selected classes with a random p between 0.2 and 1.0
    selected_augmentations = [
        aug_class(p=random.uniform(0.1, 0.9)) for aug_class in selected_classes
    ]

    pipeline = AugraphyPipeline(ink_phase=[], paper_phase=[], post_phase=selected_augmentations)
    
    try:
        augmented_data = pipeline.augment(img_bgr)
        img_bgr_aug = augmented_data["output"]
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_png_bgr(out_path, img_bgr_aug)
        
        return out_path, None
    except Exception as e:
        return None, f"Augraphy error on {img_path.name}: {e}"

# ==========================================
# 4. MULTIPROCESSING EXECUTION
# ==========================================

def main():
    args = parse_args()
    
    root = Path(args.root)
    dst = Path(args.dst)
    exts = set(args.exts)

    if not root.exists():
        raise SystemExit(f"[Error] Root directory {root} does not exist.")

    image_paths = [p for p in root.rglob("*") if p.is_file() and p.suffix in exts]
    total_images = len(image_paths)

    if total_images == 0:
        raise SystemExit(f"[Error] No images found under {root}")

    print(f"[Info] Found {total_images} images to augment.")
    print(f"[Info] Output directory: {dst}")
    
    # Run the cache warmup BEFORE launching multiple threads
    warmup_numba_cache()

    max_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"[Info] 🚀 Starting parallel augmentation using {max_workers} CPU cores...")

    success_count = 0

    with tqdm(total=total_images, desc="Augmenting", unit="img") as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            
            img_iter = iter(image_paths)
            running_futures = set()
            
            # Fill the initial processing queue
            for _ in range(max_workers * 2):
                try:
                    p = next(img_iter)
                    running_futures.add(executor.submit(process_single_image, p, root, dst))
                except StopIteration:
                    break

            # Continuously monitor and feed the queue
            while running_futures:
                done, running_futures = concurrent.futures.wait(
                    running_futures, return_when=concurrent.futures.FIRST_COMPLETED
                )
                
                for future in done:
                    out_path, err = future.result()
                    
                    if err:
                        pbar.write(f"[Warn] {err}")
                    else:
                        success_count += 1
                        
                    pbar.update(1)
                    
                    try:
                        p = next(img_iter)
                        running_futures.add(executor.submit(process_single_image, p, root, dst))
                    except StopIteration:
                        pass 

    print(f"\n✅ Augmentation Complete. {success_count}/{total_images} images successfully saved to {dst}.")

if __name__ == "__main__":
    main()