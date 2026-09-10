r"""
FINE-TUNING helper for the two binary orientation CNNs — the second half of
optional step 5 (sample selection -> slice -> fine-tune).

It fine-tunes 2.2_Horizontal_Vertical.keras or 2.2_Up_Down.keras with extra
labelled text slices (e.g. produced by optional steps 2/4/6: artificial
slices, RVL slices, or slices of your own corpus sample).

Expected training-data layout (class folders "0" and "1")
---------------------------------------------------------
    <train_dir>/0/   NEGATIVE class of the model you fine-tune
    <train_dir>/1/   POSITIVE class of the model you fine-tune

Class semantics (threshold 0.5 sigmoid head):
    2.2_Up_Down.keras              0 = upright      1 = upside-down
    2.2_Horizontal_Vertical.keras  0 = horizontal   1 = vertical

Folder-name aliases are accepted for convenience:
    0: 0, neg, negative, up, upright, horizontal
    1: 1, pos, positive, down, upside, upside_down, vertical

Safety notes
------------
* Only the head (Dense layers) is unfrozen by default; use --unfreeze-backbone
  to also open the MobileNetV2 backbone at a very low learning rate.
* NO flipping/rotating augmentation is applied — those transforms change the
  orientation label!  Optional --augment adds only brightness/contrast jitter.
* The fine-tuned model is saved as a NATIVE .keras (zip) file, so it loads
  directly in every workflow script without the legacy fallback.

Usage
-----
    python 2.1_CNN_finetune.py --model 2.2_Up_Down.keras --train-dir "D:\slices\ud" --out 2.2_Up_Down_ft.keras
"""

import argparse
import os
import random

import numpy as np

from workflow_common import (
    cpu_workers,
    load_binary_cnn,
    preprocess_for_binary_cnn,
    prompt_if_missing,
    read_image,
    resolve_model_path,
)

CLASS_ALIASES = {
    0: {"0", "neg", "negative", "up", "upright", "horizontal"},
    1: {"1", "pos", "positive", "down", "upside", "upside_down", "vertical"},
}


def parse_args():
    ap = argparse.ArgumentParser(
        description="Fine-tune one of the binary orientation CNNs with labelled slices.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model", default="2.2_Up_Down.keras",
                    help="Base model: 2.2_Up_Down.keras or 2.2_Horizontal_Vertical.keras.")
    ap.add_argument("--train-dir",
                    help="Folder with the two class sub-folders (see module docstring).")
    ap.add_argument("--out", default=None,
                    help="Output path. Default: <model stem>_finetuned.keras next to the base.")
    ap.add_argument("--img-size", type=int, nargs=2, default=[256, 256],
                    help="Model input size (H W).")
    ap.add_argument("--crop-ratio", type=float, default=None,
                    help="Set only if the base model was trained with a long-rectangle crop.")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4, help="Fine-tuning learning rate.")
    ap.add_argument("--val-split", type=float, default=0.15,
                    help="Fraction of slices held out for validation/early stopping.")
    ap.add_argument("--augment", action="store_true",
                    help="Brightness/contrast jitter only (never flip/rotate).")
    ap.add_argument("--unfreeze-backbone", action="store_true",
                    help="Additionally unfreeze the MobileNetV2 backbone (use a tiny --lr).")
    ap.add_argument("--seed", type=int, default=37)
    return ap.parse_args()


def find_class_dir(train_dir, label):
    for name in sorted(os.listdir(train_dir)):
        if name.lower() in CLASS_ALIASES[label]:
            return os.path.join(train_dir, name)
    return None


def main():
    args = parse_args()

    train_dir = prompt_if_missing(args.train_dir, "Training folder (must contain the two class sub-folders)")
    if not train_dir or not os.path.isdir(train_dir):
        raise SystemExit(f"[Error] Training folder not found: {train_dir}")

    dir0 = find_class_dir(train_dir, 0)
    dir1 = find_class_dir(train_dir, 1)
    if dir0 is None or dir1 is None:
        raise SystemExit(
            "[Error] Could not find the two class sub-folders in "
            f"{train_dir}.\nExpected aliases:\n  0: {sorted(CLASS_ALIASES[0])}\n"
            f"  1: {sorted(CLASS_ALIASES[1])}\nRemember: class meaning depends on the "
            "model (see the module docstring).")
    print(f"[Data] class 0 folder: {dir0}")
    print(f"[Data] class 1 folder: {dir1}")

    from workflow_common import gather_image_files
    files0 = gather_image_files(dir0)
    files1 = gather_image_files(dir1)
    if not files0 or not files1:
        raise SystemExit("[Error] One of the class folders is empty.")
    print(f"[Data] class 0: {len(files0):,} | class 1: {len(files1):,} slices.")

    random.seed(args.seed)
    np.random.seed(args.seed)

    items = [(f, 0) for f in files0] + [(f, 1) for f in files1]
    random.shuffle(items)
    n_val = max(1, int(len(items) * args.val_split))
    val_items, train_items = items[:n_val], items[n_val:]
    print(f"[Data] train: {len(train_items):,} | validation: {len(val_items):,}")

    import tensorflow as tf
    h, w = args.img_size

    def augment(arr):
        arr = arr * (1.0 + float(np.random.uniform(-0.15, 0.15)))  # contrast
        arr = arr + float(np.random.uniform(-0.08, 0.08))          # brightness
        return np.clip(arr, 0.0, 1.0)

    def make_dataset(item_list, training):
        def gen():
            for path, label in item_list:
                img = read_image(path)
                arr = preprocess_for_binary_cnn(img, img_size=(h, w), crop_ratio=args.crop_ratio)
                if arr is None:
                    continue
                if training and args.augment:
                    arr = augment(arr)
                yield arr, np.float32(label)

        return (tf.data.Dataset.from_generator(
                    gen, output_signature=(
                        tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
                        tf.TensorSpec(shape=(), dtype=tf.float32)))
                .map(lambda x, y: (tf.image.resize(x, (h, w)), y),
                     num_parallel_calls=tf.data.AUTOTUNE)
                .batch(args.batch_size)
                .prefetch(tf.data.AUTOTUNE))

    train_ds = make_dataset(train_items, training=True).shuffle(min(2048, len(train_items)))
    val_ds = make_dataset(val_items, training=False)

    print(f"[Model] Loading base model: {args.model}")
    model = load_binary_cnn(args.model, img_size=(h, w))

    head_names = {"dense", "dense_1", "dropout", "prob"}
    for layer in model.layers:
        layer.trainable = args.unfreeze_backbone or layer.name in head_names
    print("[Model] Trainable: " +
          ("head (Dense) layers" if not args.unfreeze_backbone
           else "all layers incl. backbone") +
          f" | learning rate = {args.lr}")

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max",
                                                  patience=3, restore_best_weights=True)]

    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=args.epochs, callbacks=callbacks, verbose=2)

    out = args.out
    if not out:
        base = resolve_model_path(args.model)
        out = os.path.splitext(os.path.basename(base))[0] + "_finetuned.keras"
    model.save(out)
    val_acc = max(h or 0.0 for h in (history.history.get("val_accuracy") or [0.0]))
    print("-" * 60)
    print(f"[Done] Fine-tuned model saved to: {os.path.abspath(out)}")
    print(f"[Done] Best validation accuracy: {val_acc:.4f}")
    print("[Next] Validate visually with 2.1_CNN_Test.py, then point the rotation "
          "scripts at the new file if you are satisfied.")


if __name__ == "__main__":
    main()
