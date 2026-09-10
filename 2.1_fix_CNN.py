r"""
OPTIONAL STEP 7 of the workflow — one-off CNN repair/conversion tool.

WHEN TO USE — two telltale situations:

1. BATCH-DEPENDENT predictions.  The same image scores differently when
   predicted alone vs. inside a batch; accuracy shifts with the predict()
   batch size; validation results are erratic.  Root cause: the checkpoint
   was built with `feats = base(x, training=True)` baked into the graph, so
   MobileNetV2's BatchNormalization keeps using the CURRENT batch
   statistics instead of the learned moving mean/variance at inference.

2. LEGACY CHECKPOINTS that modern Keras refuses to load (this includes the
   shipped 2.2_Horizontal_Vertical.keras / 2.2_Up_Down.keras, whose HDF5
   payloads contain TFOpLambda layers).  This script rebuilds the exact
   architecture cleanly and transfers ALL trained weights (including BN
   moving statistics) — no retraining required.

WHAT IT DOES: rebuilds the binary orientation CNN (MobileNetV2 alpha=1.0
backbone + GAP + Dense(128) + Dropout(0.2) + Dense(1) + sigmoid "prob"),
transfers the weights via set_weights(), and saves a FIXED copy as a native
(zip) .keras file that every other script in this repo loads directly.

Run it once per affected model, then point the rotation scripts at the
fixed file if you do not want the automatic legacy fallback.

Usage
-----
    python 2.1_fix_CNN.py --broken 2.2_Up_Down.keras
    # -> writes 2.2_Up_Down_fixed.keras next to the original

    python 2.1_fix_CNN.py --broken "D:\old\my_model.h5" --img-size 128 256 --out "D:\old\my_model_fixed.keras"
"""

import argparse
import os

import tensorflow as tf

from workflow_common import rebuild_binary_orientation_cnn, resolve_model_path, prompt_if_missing


def main():
    ap = argparse.ArgumentParser(
        description="Optional step 7: rebuild a binary orientation CNN and transfer its "
                    "weights (fixes training=True graphs / legacy saves).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--broken", default=None,
                    help="Path of the affected checkpoint (legacy .h5 or HDF5-payload "
                         ".keras). Prompted for when omitted.")
    ap.add_argument("--out", default=None,
                    help="Output path of the fixed model. Default: <broken stem>_fixed.keras "
                         "next to the broken file.")
    ap.add_argument("--img-size", type=int, nargs=2, default=[256, 256],
                    help="Training input size (H W), e.g. 256 256, or 128 256 if the "
                         "model was trained with --crop-long-rectangle 2.")
    args = ap.parse_args()

    broken = prompt_if_missing(args.broken, "Path of the broken/legacy checkpoint")
    broken = resolve_model_path(broken)
    if not os.path.exists(broken):
        raise SystemExit(f"[Error] Checkpoint not found: {broken}")

    out = args.out
    if not out:
        out = os.path.splitext(broken)[0] + "_fixed.keras"
    out = os.path.abspath(out)

    img_size = tuple(args.img_size)

    print(f"[1/4] Loading broken model: {broken}")
    broken_model = tf.keras.models.load_model(broken, compile=False)

    print(f"[2/4] Rebuilding clean architecture at input size {img_size} ...")
    fixed_model = rebuild_binary_orientation_cnn(img_size)

    n_broken = broken_model.count_params()
    n_fixed = fixed_model.count_params()
    print(f"      broken params = {n_broken:,} | rebuilt params = {n_fixed:,}")
    if n_broken != n_fixed:
        raise SystemExit("[Error] Parameter counts differ — adjust --img-size so the "
                         "architecture matches the checkpoint, then retry.")

    print("[3/4] Transferring weights (includes BatchNorm moving statistics) ...")
    fixed_model.set_weights(broken_model.get_weights())

    print(f"[4/4] Saving fixed model to: {out}")
    fixed_model.save(out)  # native .keras (zip) format — loads everywhere
    print("[Done] The fixed model contains NO training=True op and loads with any "
          "recent Keras. Point the rotation scripts at it (or rely on the automatic "
          "legacy fallback in workflow_common.load_binary_cnn).")


if __name__ == "__main__":
    main()
