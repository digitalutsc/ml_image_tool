r"""
FULL TRAINING script for the two binary orientation CNNs — the companion of
2.1_CNN_finetune.py (which only fine-tunes an existing checkpoint).

It follows the original recipe of ml_image_tool-main/ocr_hori_vs_vert.py
(MobileNetV2 backbone, baked-in mobilenet_v2.preprocess_input, GAP,
Dense(128), Dropout(0.2), single sigmoid "prob" node, brightness/contrast/
zoom/translation augmentation, class weights, ModelCheckpoint +
EarlyStopping + ReduceLROnPlateau, final confusion matrix).  Differences
from the original:

1.  --mode hv | ud selects WHICH classifier to train (you are prompted for
    it when the flag is omitted):

        hv : Horizontal-TEXT vs Vertical-TEXT slices
             class 0 = HorizontalText folder = pages at 0 deg/180 deg
                                            (text lines run horizontally)
             class 1 = VerticalText   folder = pages at +-90 deg
                                            (text lines run vertically)
             -> identical to the original script's convention, where the
                unlabeled mode used labels [0,1,0,1] for (0,90,180,270 CCW)
                and the labeled mode used y = class_index % 2 with the
                explicit class order r0, r90, r180, r270.

        ud : Up vs Down
             class 0 = 0Degree(s) folder = upright pages
             class 1 = 180Degree(s) folder = upside-down pages
             -> "0 = Upright, 1 = Upside", as documented in 2.1_CNN_Test.py.

    The label assignment is PINNED EXPLICITLY by this script — it never
    relies on the alphabetical folder order of
    tf.keras.utils.image_dataset_from_directory (that alphabetical
    behaviour is what older Keras flow_from_directory pipelines depended
    on; with these folder names it would coincidentally give the same
    result, but it is not relied upon here).  The resolved mapping is
    printed before training starts — READ IT.

2.  The backbone is called WITHOUT `training=True`, so the saved model does
    NOT suffer from the batch-statistics bug that 2.1_fix_CNN.py exists to
    repair in the legacy checkpoints.

3.  The best checkpoint is saved as a .keras file next to the scripts (or
    into --out-dir).  Under Keras 3 this is a native zip; note that older
    Keras 2 environments save HDF5 regardless of the extension —
    workflow_common.py loads BOTH transparently, including the TFOpLambda
    rebuild fallback.

Expected data layout (either works, prompts cover the rest):

    <data-root>\HorizontalText\*.png        (hv mode, class 0)
    <data-root>\VerticalText\*.png          (hv mode, class 1)

    <data-root>\0Degree(s)\*.png            (ud mode, class 0)
    <data-root>\180Degree\*.png             (ud mode, class 1)

A validation split is taken automatically via --val-split; alternatively
point --val-root at a folder with the same two class subfolders.

Usage
-----
    python 2.1_CNN_train.py                       # prompts for mode & data
    python 2.1_CNN_train.py --mode hv --data-root "C:\D\UofT\DSU_Paper\RVL-CDIP-Slice-Train\Horizontal-Vertical_Text"
    python 2.1_CNN_train.py --mode ud --data-root "C:\D\UofT\DSU_Paper\RVL-CDIP-Slice-Train\UpVsDown"
"""

import argparse
import os
import random
import sys

import numpy as np

from workflow_common import resolve_model_path  # noqa: F401  (kept for symmetry)

AUTOTUNE = None  # resolved after the tf import

# Class-folder candidates, tried in order, per mode.  The FIRST existing
# folder wins.  You can always override with --class0-dir / --class1-dir.
FOLDER_CANDIDATES = {
    "hv": {
        0: ["HorizontalText", "horizontaltext", "Horizontal", "horizontal"],
        1: ["VerticalText", "verticaltext", "Vertical", "vertical"],
    },
    "ud": {
        0: ["0Degree", "0Degrees", "Up", "upright"],
        1: ["180Degree", "180Degrees", "Down", "upside_down", "upside"],
    },
}

MODE_LEGENDS = {
    "hv": ("0 deg & 180 deg pages (horizontal text lines)  -> score 0",
           "+-90 deg pages (vertical text lines)           -> score 1"),
    "ud": ("upright pages                                   -> score 0",
           "upside-down pages                               -> score 1"),
}


def parse_args():
    ap = argparse.ArgumentParser(
        description="Train one of the two binary orientation CNNs from scratch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--mode", choices=["hv", "ud"], default=None,
                    help="hv = Horizontal vs Vertical text classifier, "
                         "ud = Up vs Down classifier. Prompted when omitted.")
    ap.add_argument("--data-root", default=None,
                    help="Dataset root containing the two class folders "
                         "(prompted when omitted).")
    ap.add_argument("--class0-dir", default=None,
                    help="Explicit class-0 folder inside --data-root "
                         "(overrides auto-detection).")
    ap.add_argument("--class1-dir", default=None,
                    help="Explicit class-1 folder inside --data-root "
                         "(overrides auto-detection).")
    ap.add_argument("--val-root", default=None,
                    help="Optional separate validation root with the same two "
                         "class subfolders. Default: split off --val-split of "
                         "the training files.")
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--img-size", type=int, nargs=2, default=[256, 256],
                    help="(H W) base resize before the optional rectangle crop.")
    ap.add_argument("--crop-long-rectangle", type=float, default=None,
                    help="If R>1, center-crop to (H/R, W) like the original recipe.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--fine-tune-at", type=int, default=1,
                    help="Unfreeze MobileNetV2 layers at this index "
                         "(<=0 freezes the whole backbone).")
    ap.add_argument("--no-imagenet", action="store_true",
                    help="Train from scratch (no ImageNet weights).")
    ap.add_argument("--mixed-precision", action="store_true")
    ap.add_argument("--out-dir", default=None,
                    help="Where to save the best checkpoint. Default: next to "
                         "the scripts.")
    ap.add_argument("--seed", type=int, default=1337)
    return ap.parse_args()


def prompt_mode():
    print("Which classifier do you want to train?")
    print("  1) hv : Horizontal vs Vertical text lines "
          "(class 0 = 0/180-deg pages, class 1 = +-90-deg pages)")
    print("  2) ud : Up vs Down (class 0 = upright, class 1 = upside-down)")
    answer = input("Enter 1/2 (or hv/ud): ").strip().lower()
    return {"1": "hv", "hv": "hv", "2": "ud", "ud": "ud"}.get(answer)


def find_class_dir(data_root, label, explicit, mode):
    if explicit:
        path = os.path.join(data_root, explicit)
        if os.path.isdir(path):
            return path
        raise SystemExit(f"[Error] --class{label}-dir not found: {path}")
    for name in FOLDER_CANDIDATES[mode][label]:
        path = os.path.join(data_root, name)
        if os.path.isdir(path):
            return path
    return None


def main():
    args = parse_args()

    mode = args.mode or prompt_mode()
    if mode not in ("hv", "ud"):
        raise SystemExit("[Error] Invalid mode selection.")
    args.mode = mode

    data_root = args.data_root
    while not data_root or not os.path.isdir(data_root):
        data_root = input("Dataset root folder: ").strip().strip('"').strip("'")
    data_root = os.path.abspath(data_root)

    dir0 = find_class_dir(data_root, 0, args.class0_dir, mode)
    dir1 = find_class_dir(data_root, 1, args.class1_dir, mode)
    if dir0 is None or dir1 is None:
        raise SystemExit(
            f"[Error] Could not find the two class folders inside {data_root}.\n"
            f"  looked for class 0: {FOLDER_CANDIDATES[mode][0]}\n"
            f"  looked for class 1: {FOLDER_CANDIDATES[mode][1]}\n"
            f"Pass --class0-dir / --class1-dir if your folders are named "
            f"differently.")

    legend0, legend1 = MODE_LEGENDS[mode]
    print("=" * 70)
    print(f"[Mode] {mode.upper()} classifier")
    print(f"[Data] class 0: {dir0}")
    print(f"       {legend0}")
    print(f"[Data] class 1: {dir1}")
    print(f"       {legend1}")
    print("[Note] The sigmoid node will output P(class 1). This mapping is "
          "pinned explicitly and does NOT depend on alphabetical folder order.")
    print("=" * 70)

    import tensorflow as tf
    global AUTOTUNE
    AUTOTUNE = tf.data.AUTOTUNE

    if args.mixed_precision:
        try:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy("mixed_float16")
            print("[Info] Mixed precision enabled (float16).")
        except Exception as exc:  # noqa: BLE001
            print(f"[Warn] Could not enable mixed precision: {exc}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

    def list_images(folder):
        found = []
        for dirpath, dirnames, filenames in os.walk(folder):
            dirnames.sort()
            for name in sorted(filenames):
                if os.path.splitext(name)[1].lower() in exts:
                    found.append(os.path.join(dirpath, name))
        return found

    files0 = list_images(dir0)
    files1 = list_images(dir1)
    if not files0 or not files1:
        raise SystemExit("[Error] One of the class folders is empty.")
    print(f"[Data] class 0: {len(files0):,} | class 1: {len(files1):,} images.")

    # ---- validation split -------------------------------------------------
    rng = np.random.default_rng(args.seed)
    if args.val_root and os.path.isdir(args.val_root):
        vdir0 = find_class_dir(args.val_root, 0, args.class0_dir, mode)
        vdir1 = find_class_dir(args.val_root, 1, args.class1_dir, mode)
        if vdir0 is None or vdir1 is None:
            raise SystemExit(f"[Error] --val-root must contain the same two "
                             f"class folders: {args.val_root}")
        train_items = [(f, 0) for f in files0] + [(f, 1) for f in files1]
        val_items = ([(f, 0) for f in list_images(vdir0)] +
                     [(f, 1) for f in list_images(vdir1)])
    else:
        items = [(f, 0) for f in files0] + [(f, 1) for f in files1]
        rng.shuffle(items)
        n_val = max(1, int(len(items) * args.val_split))
        val_items, train_items = items[:n_val], items[n_val:]
    print(f"[Data] train: {len(train_items):,} | validation: {len(val_items):,}")

    # ---- preprocessing (identical to the original recipe) -----------------
    base_h, base_w = args.img_size
    crop_ratio = args.crop_long_rectangle
    if crop_ratio is not None and crop_ratio <= 1.0:
        raise SystemExit("[Error] --crop-long-rectangle must be > 1.0.")

    def decode_img(path):
        img = tf.io.read_file(path)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
        img = tf.image.resize(img, (base_h, base_w), antialias=True)
        if crop_ratio:
            h_eff = tf.maximum(1, tf.cast(tf.round(base_h / crop_ratio), tf.int32))
            top = (base_h - h_eff) // 2
            img = tf.image.crop_to_bounding_box(img, top, 0, h_eff, base_w)
        return img

    def ds_from_items(item_list, training):
        ds = tf.data.Dataset.from_generator(
            lambda: iter(item_list),
            output_signature=(tf.TensorSpec(shape=(), dtype=tf.string),
                              tf.TensorSpec(shape=(), dtype=tf.int32)))
        ds = ds.map(lambda p, y: (decode_img(p), tf.cast(y, tf.float32)),
                    num_parallel_calls=AUTOTUNE)
        if training:
            ds = ds.shuffle(min(4096, len(item_list) * 4), seed=args.seed,
                            reshuffle_each_iteration=True)
        return ds

    augmenter = tf.keras.Sequential([
        tf.keras.layers.RandomBrightness(factor=0.2),
        tf.keras.layers.RandomContrast(factor=0.15),
        tf.keras.layers.RandomZoom(height_factor=(-0.05, 0.05),
                                   width_factor=(-0.05, 0.05)),
        tf.keras.layers.RandomTranslation(height_factor=0.05, width_factor=0.05),
        tf.keras.layers.RandomRotation(factor=0.005, fill_mode="reflect"),
        tf.keras.layers.GaussianNoise(0.005),
    ], name="augment")

    def prepare(ds, training):
        if training:
            ds = ds.map(lambda x, y: (augmenter(x, training=True), y),
                        num_parallel_calls=AUTOTUNE)
        return ds.batch(args.batch_size).prefetch(AUTOTUNE)

    train_ds = prepare(ds_from_items(list(train_items), True), True)
    val_ds = prepare(ds_from_items(list(val_items), False), False)

    # ---- class weights (original recipe) -----------------------------------
    counts = {0: 0, 1: 0}
    for p, y in train_items:
        counts[int(y)] += 1
    total = sum(counts.values())
    class_weights = {0: total / (2.0 * max(1, counts[0])),
                     1: total / (2.0 * max(1, counts[1]))}
    print(f"[Info] Estimated class weights: {class_weights}")

    # ---- model (original architecture; NO training=True in the graph) ------
    eff_h = int(round(base_h / crop_ratio)) if crop_ratio else base_h
    eff_h = max(1, eff_h)
    eff_w = base_w

    inputs = tf.keras.Input(shape=(eff_h, eff_w, 3), name="input")
    base = tf.keras.applications.MobileNetV2(
        input_shape=(eff_h, eff_w, 3), include_top=False,
        weights=None if args.no_imagenet else "imagenet", alpha=1.0)
    if args.fine_tune_at <= 0:
        base.trainable = False
    else:
        base.trainable = True
        for i, layer in enumerate(base.layers):
            layer.trainable = i >= args.fine_tune_at

    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    feats = base(x)  # NOTE: no training=True -> no BN batch-statistics bug
    pooled = tf.keras.layers.GlobalAveragePooling2D()(feats)
    head = tf.keras.layers.Dense(128, activation="relu", name="dense")(pooled)
    head = tf.keras.layers.Dropout(0.2, name="dropout")(head)
    logit = tf.keras.layers.Dense(1, dtype="float32", name="dense_1")(head)
    prob = tf.keras.layers.Activation("sigmoid", name="prob")(logit)

    model = tf.keras.Model(inputs, prob, name="orientation_cls_binary_axis")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=[tf.keras.metrics.BinaryAccuracy(name="acc", threshold=0.5),
                           tf.keras.metrics.AUC(name="auc")])

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    out_name = "2.2_Horizontal_Vertical_retrained.keras" if mode == "hv" \
        else "2.2_Up_Down_retrained.keras"
    ckpt_path = os.path.join(out_dir, out_name)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_acc",
                                           mode="max", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_acc", mode="max",
                                         patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_acc", mode="max",
                                             factor=0.5, patience=4,
                                             min_lr=1e-6, verbose=1),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
              class_weight=class_weights, callbacks=callbacks, verbose=1)

    # ---- final evaluation (original recipe) --------------------------------
    print("\n[Info] Final evaluation on the validation set:")
    y_true, y_pred = [], []
    for bx, by in val_ds.unbatch().batch(1024):
        probs = model.predict(bx, verbose=0).ravel()
        y_pred.append((probs >= 0.5).astype(np.int32))
        y_true.append(by.numpy().astype(np.int32))
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    cm = np.zeros((2, 2), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    print(f"[Confusion Matrix]  rows=true 0/1, cols=predicted 0/1\n{cm}")
    print(f"[Val Accuracy] {(cm[0, 0] + cm[1, 1]) / max(1, cm.sum()):.4f}")

    print("-" * 70)
    print(f"[Done] Best checkpoint: {ckpt_path}")
    print(f"[Legend] sigmoid output = P(class 1);")
    print(f"         class 0 | {legend0}")
    print(f"         class 1 | {legend1}")
    print("[Next] Calibrate on known-orientation slices with 2.1_CNN_Test.py "
          "before pointing the rotation scripts at the new checkpoint.")


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
            print(f"[Info] GPUs: {len(gpus)} (memory growth enabled)")
    except Exception as exc:  # noqa: BLE001
        print(f"[Warn] GPU setup: {exc}")
    main()
