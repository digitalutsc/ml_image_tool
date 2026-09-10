"""
workflow_common.py — shared helpers for the image-correction workflow.

This module is NOT a workflow step (that is why it has no number prefix).
Every numbered script imports from here so that behaviour stays consistent:

  1. Smart Keras model loading.
     The three models shipped in this repo (2.1_best_projection_band_2.keras,
     2.2_Horizontal_Vertical.keras, 2.2_Up_Down.keras) are LEGACY HDF5
     payloads that were given a ".keras" extension.  Keras 3 refuses to open
     HDF5 content through the ".keras" (zip) code path, so:
       * load_keras_model() transparently copies HDF5 payloads to a
         temporary ".h5" file and loads that, and
       * load_binary_cnn() additionally falls back to rebuilding the
         MobileNetV2 binary-classifier architecture (the exact recipe of
         2.1_fix_CNN.py) and transferring the HDF5 weights, because those
         two models contain TFOpLambda layers that Keras 3 cannot
         deserialize from a legacy HDF5 model config.

  2. Class semantics of the binary sigmoid models (threshold 0.5):
       2.2_Horizontal_Vertical.keras  ->  P(vertical text axis).
           score <  0.5 : HORIZONTAL  (image rotated +-90 deg)
           score >= 0.5 : VERTICAL    (image at 0 or 180 deg)
       2.2_Up_Down.keras              ->  P(upside-down).
           score <  0.5 : UPRIGHT     (as documented in 2.1_CNN_Test.py:
                                        "0 = Upright, 1 = Upside")
           score >= 0.5 : UPSIDE-DOWN (needs a 180 deg rotation)
     The naming convention is "A_vs_B" = {0: A, 1: B}.  Every rotation
     script prints this legend at start-up and offers an --invert-* flag in
     case a model was ever re-trained with swapped labels.

  3. Robust image IO (numpy-buffered, so non-ASCII Windows paths work),
     the 2.1_CNN_Test.py preprocessing recipe, and the
     "mirror slice  <->  original image" lookup + paired-rotation helpers
     used by the three rotation-correction methods (steps 9-11).
"""

import os
import tempfile
import shutil

import numpy as np
import cv2

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"

# NTFS allows at most 4,294,967,295 (2**32 - 1) files per folder.  Step 1
# uses half of that as its default ceiling so that even a pathological
# single folder stays well below the operating-system limit.
NTFS_MAX_FILES_PER_FOLDER = 2**32 - 1
DEFAULT_MAX_FILES_PER_FOLDER = NTFS_MAX_FILES_PER_FOLDER // 2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff", ".webp"}

# The separator introduced by 1_gather_images.py between the parent folder
# index and the enumerated file name, e.g. "0@@!!!!!!@@000123.tif".
FLATTEN_SEPARATOR = "@@!!!!!!@@"

# The suffix that marks the RIGHT half of a split two-page image (step 4).
RIGHT_PAGE_SUFFIX = "_r"

# ----------------------------------------------------------------------------
# CPU helpers
# ----------------------------------------------------------------------------

def cpu_workers(requested=None):
    """Number of workers to use for (multi)threading / (multi)processing.

    Defaults to the machine's logical core count.  Pass a smaller number via
    the --workers CLI flag of any script if you want to keep cores free.
    """
    if requested and int(requested) > 0:
        return int(requested)
    return os.cpu_count() or 1

# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------

def resolve_model_path(path):
    """Resolve a model path.  Relative names are looked up next to this
    module first (so the workspace models pair with the scripts no matter
    where the terminal is), then as given."""
    if os.path.isabs(path):
        return path
    candidate = os.path.join(SCRIPT_DIR, path)
    if os.path.exists(candidate):
        return candidate
    return path


def is_hdf5_file(path):
    try:
        with open(path, "rb") as f:
            return f.read(8) == HDF5_MAGIC
    except OSError:
        return False


def rebuild_binary_orientation_cnn(img_size=(256, 256)):
    """Rebuild the binary orientation CNN exactly as trained by the
    ocr_hori_vs_vert.py / ocr_train.py recipe (same recipe documented in
    2.1_fix_CNN.py): MobileNetV2(alpha=1.0) backbone with
    mobilenet_v2.preprocess_input baked in, GAP, Dense(128), Dropout(0.2),
    Dense(1) and a sigmoid "prob" output.  Layer names match the saved
    checkpoints so HDF5 weights transfer cleanly."""
    import tensorflow as tf  # lazy import: non-TF scripts stay light

    h, w = img_size
    inputs = tf.keras.Input(shape=(h, w, 3), name="input")
    base = tf.keras.applications.MobileNetV2(
        input_shape=(h, w, 3), include_top=False, weights=None, alpha=1.0
    )
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    feats = base(x)
    pooled = tf.keras.layers.GlobalAveragePooling2D()(feats)
    head = tf.keras.layers.Dense(128, activation="relu", name="dense")(pooled)
    head = tf.keras.layers.Dropout(0.2, name="dropout")(head)
    logits = tf.keras.layers.Dense(1, dtype="float32", name="dense_1")(head)
    outputs = tf.keras.layers.Activation("sigmoid", name="prob")(logits)
    return tf.keras.Model(inputs, outputs, name="orientation_cls_binary_axis")


def load_keras_model(path, verbose=True):
    """Load a Keras model, transparently handling the legacy-HDF5-with-
    .keras-extension files shipped in this repo."""
    import tensorflow as tf

    path = resolve_model_path(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    if is_hdf5_file(path):
        if verbose:
            print(f"[Info] {os.path.basename(path)} is a legacy HDF5 payload; "
                  f"loading through a temporary .h5 copy.")
        tmp = tempfile.mktemp(suffix=".h5")
        shutil.copyfile(path, tmp)
        try:
            return tf.keras.models.load_model(tmp, compile=False)
        finally:
            os.remove(tmp)
    return tf.keras.models.load_model(path, compile=False)


def load_binary_cnn(path, img_size=(256, 256), verbose=True):
    """Load one of the two binary orientation CNNs
    (2.2_Horizontal_Vertical.keras / 2.2_Up_Down.keras).

    Tries a normal load first; if the legacy HDF5 config cannot be
    deserialized by Keras 3 (TFOpLambda layers), rebuilds the architecture
    and transfers the weights instead — no retraining required."""
    import tensorflow as tf

    path = resolve_model_path(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    try:
        return load_keras_model(path, verbose=verbose)
    except (ValueError, TypeError) as exc:
        if "TFOpLambda" not in str(exc) and "Unknown layer" not in str(exc):
            raise
        if verbose:
            print(f"[Info] {os.path.basename(path)} stores TFOpLambda layers that "
                  f"Keras 3 cannot deserialize directly -> rebuilding the "
                  f"MobileNetV2 recipe and transferring weights.")
        tmp = tempfile.mktemp(suffix=".h5")
        shutil.copyfile(path, tmp)
        try:
            model = rebuild_binary_orientation_cnn(img_size)
            model.load_weights(tmp)  # HDF5 weights load by layer name/order
            return model
        finally:
            os.remove(tmp)

# ----------------------------------------------------------------------------
# Image IO (robust against non-ASCII Windows paths)
# ----------------------------------------------------------------------------

def read_image(path):
    """Read an image as BGR (cv2 convention).  Returns None on failure."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def write_image(path, img, jpeg_quality=None):
    """Write an image (BGR).  The format follows the destination extension.
    Returns True on success."""
    ext = os.path.splitext(str(path))[1].lower() or ".png"
    params = []
    if ext in (".jpg", ".jpeg") and jpeg_quality is not None:
        params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def gather_image_files(root, exts=None):
    """Recursively collect image files under root, in sorted, reproducible
    order."""
    exts = exts or IMAGE_EXTS
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            if os.path.splitext(name)[1].lower() in exts:
                found.append(os.path.join(dirpath, name))
    return found

# ----------------------------------------------------------------------------
# Binary CNN preprocessing (recipe of 2.1_CNN_Test.py)
# ----------------------------------------------------------------------------

def preprocess_for_binary_cnn(img_bgr, img_size=(256, 256), crop_ratio=None):
    """RGB float32 [0,1], resized with area interpolation, optional centre
    crop of the long dimension (set crop_ratio>1 if the model was trained
    that way).  Returns a tensor of shape (*img_size, 3) or None."""
    if img_bgr is None:
        return None
    h, w = img_size
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    arr = rgb.astype(np.float32) / 255.0
    if crop_ratio and crop_ratio > 1.0:
        if w >= h:  # wide: crop horizontally
            w_eff = int(round(w / crop_ratio))
            left = (w - w_eff) // 2
            arr = arr[:, left:left + w_eff, :]
        else:       # tall: crop vertically
            h_eff = int(round(h / crop_ratio))
            top = (h - h_eff) // 2
            arr = arr[top:top + h_eff, :, :]
    return arr

# ----------------------------------------------------------------------------
# Mirror <-> original lookup and paired rotation (steps 8-11)
# ----------------------------------------------------------------------------

def find_original_for_slice(slice_path, mirror_root, original_root):
    """Locate the original full-size image that produced this mirror slice.
    The mirror tree replicates the flattened layout of step 1, so the
    original lives at the SAME relative sub-path with the SAME stem (any
    image extension):

        <mirror_root>/<sub>/<stem>.png   <->   <original_root>/<sub>/<stem>.<ext>

    Returns None when no original exists."""
    slice_path = str(slice_path)
    mirror_root = str(mirror_root)
    original_root = str(original_root)
    try:
        rel = os.path.relpath(slice_path, mirror_root)
    except ValueError:  # different drives on Windows
        return None
    if rel.startswith(".."):
        return None
    sub = os.path.dirname(rel)
    stem = os.path.splitext(os.path.basename(slice_path))[0]
    for ext in sorted(IMAGE_EXTS):
        cand = os.path.join(original_root, sub, stem + ext)
        if os.path.exists(cand):
            return cand
    return None


ROTATE_FLAGS = {
    "90ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "90cw": cv2.ROTATE_90_CLOCKWISE,
    "180": cv2.ROTATE_180,
}


def rotate_file_inplace(path, rotate_name, jpeg_quality=100):
    """Rotate one image file on disk by one of {'90ccw','90cw','180'}.
    Returns True on success."""
    img = read_image(path)
    if img is None:
        return False
    rotated = cv2.rotate(img, ROTATE_FLAGS[rotate_name])
    return write_image(path, rotated, jpeg_quality=jpeg_quality)


def rotate_pair(slice_path, original_path, rotate_name):
    """Rotate BOTH the mirror slice and its original full-size image by the
    same transform, in place.  Returns (slice_ok, original_ok)."""
    slice_ok = rotate_file_inplace(slice_path, rotate_name)
    original_ok = False
    if original_path:
        original_ok = rotate_file_inplace(original_path, rotate_name)
    return slice_ok, original_ok


def iter_batch_scores(model, paths, img_size=(256, 256), crop_ratio=None,
                      batch_size=32, workers=None):
    """Yield (path, score) for every readable image in `paths`, batch-
    predicting `model` (single sigmoid output) with the 2.1_CNN_Test.py
    preprocessing recipe.  Image decoding runs on a thread pool with one
    thread per CPU core; prediction happens batch-wise on the calling
    thread, which keeps a single model instance (CPU or GPU) busy."""
    from concurrent.futures import ThreadPoolExecutor

    def load_one(p):
        return preprocess_for_binary_cnn(read_image(p), img_size=img_size,
                                         crop_ratio=crop_ratio)

    workers = cpu_workers(workers)
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            tensors = list(pool.map(load_one, batch_paths))
        keep, probs = [], []
        for p, t in zip(batch_paths, tensors):
            if t is not None:
                keep.append(p)
                probs.append(t)
        if not keep:
            continue
        import numpy as np
        batch = np.stack(probs, axis=0)
        scores = model.predict(batch, verbose=0).ravel()
        for p, s in zip(keep, scores):
            yield p, float(s)


def unmirrored_originals(mirror_root, original_root):
    """Originals (absolute paths) that have no mirror slice — these cannot be
    auto-rotated by the mirror-based methods and need manual attention."""
    slice_stems = set()
    for dirpath, _, filenames in os.walk(mirror_root):
        for name in filenames:
            if name.lower().endswith(".png"):
                slice_stems.add(os.path.splitext(name)[0])
    missing = []
    for path in gather_image_files(original_root):
        if os.path.splitext(os.path.basename(path))[0] not in slice_stems:
            missing.append(path)
    return missing

# ----------------------------------------------------------------------------
# Small CLI conveniences
# ----------------------------------------------------------------------------

def prompt_if_missing(value, label, default=None):
    """If a required CLI argument was omitted, ask for it interactively.
    Pressing Enter accepts the shown default (if any)."""
    if value:
        return value
    tail = f" [{default}]" if default else ""
    answer = input(f"{label}{tail}: ").strip().strip('"').strip("'")
    return answer or default


def confirm(question, assume_yes=False):
    if assume_yes:
        return True
    answer = input(f"{question} (y/N): ").strip().lower()
    return answer in ("y", "yes")
