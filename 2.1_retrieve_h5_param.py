r"""
Inspection helper (no workflow number — run it whenever you want to know
what a checkpoint contains).

Prints the structure of any model file shipped/used in this repo:
* native .keras files load directly;
* legacy .h5 files load through the HDF5 path;
* HDF5 payloads with a .keras extension (all three workspace models) are
  handled transparently, including the TFOpLambda rebuild fallback for the
 two binary orientation CNNs.

Usage
-----
    python 2.1_retrieve_h5_param.py                       # prompts for a model
    python 2.1_retrieve_h5_param.py --model 2.2_Up_Down.keras
    python 2.1_retrieve_h5_param.py --model "D:\some\old_model.h5"
"""

import argparse

from workflow_common import load_binary_cnn, load_keras_model, prompt_if_missing, resolve_model_path


def main():
    ap = argparse.ArgumentParser(
        description="Summarize a saved model (input/output shapes, layers, head).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model", default="2.1_best_projection_band_2.keras",
                    help="Model to inspect. Defaults to the workspace projection-band "
                         "model; works for the 2.2_* binary CNNs too.")
    ap.add_argument("--full-summary", action="store_true",
                    help="Also print the per-layer Keras summary table.")
    args = ap.parse_args()

    path = prompt_if_missing(args.model, "Path of the model to inspect")
    path = resolve_model_path(path)

    # Try a plain load first; fall back to the binary-CNN rebuild recipe.
    try:
        model = load_keras_model(path, verbose=True)
    except (ValueError, TypeError):
        print("[Info] Plain load failed (legacy binary CNN?) — retrying with the "
              "MobileNetV2 rebuild recipe ...")
        model = load_binary_cnn(path)

    print("=" * 70)
    print("Model        :", path)
    print("Inputs       :", [f"{i.shape} ({i.dtype})" for i in model.inputs])
    print("Outputs      :", [f"{o.shape} ({o.dtype})" for o in model.outputs])
    print("Total params :", f"{model.count_params():,}")
    print("Last layer   :", type(model.layers[-1]).__name__)

    print()
    print("Class legend of the binary sigmoid head (threshold 0.5):")
    name = path.lower()
    if "up_down" in name:
        print("  score <  0.5 -> UPRIGHT      (rotation 0)")
        print("  score >= 0.5 -> UPSIDE-DOWN  (rotation 180)")
    elif "horizontal" in name:
        print("  score <  0.5 -> HORIZONTAL   (rotation +-90)")
        print("  score >= 0.5 -> VERTICAL     (rotation 0 or 180)")
    else:
        print("  (single sigmoid output; meaning depends on training labels)")

    if args.full_summary:
        print()
        model.summary()


if __name__ == "__main__":
    main()
