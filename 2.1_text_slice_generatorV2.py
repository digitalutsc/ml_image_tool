import os
# Force Numba to use a local, shorter path for caching to avoid Windows 260-char limit
os.environ["NUMBA_CACHE_DIR"] = os.path.join(os.getcwd(), ".numba_cache")

import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import concurrent.futures
import multiprocessing

from augraphy import *

# ==========================================
# 1. CONFIGURATION
# ==========================================

OUTPUT_DIR = "augraphyV2"
NUM_IMAGES = 3000
IMG_W, IMG_H = 256, 256

# === GENERATION PARAMETERS ===
LINES_OPTIONS = [5, 6, 7]
TILT_RANGE = (-2, 2)
NOISE_MEAN = 0
NOISE_SIGMA = 25
HORIZONTAL_JITTER_RANGE = (-30, 30)
LINE_SPACING_FACTOR_RANGE = (1.5, 2.3)

# === NEW PARAMETERS ===
CAPS_PERCENTAGE = 0.1 
MIXED_NUMERALS_PERCENTAGE = 0.1 

# === DAMAGE PARAMETERS ===
DAMAGE_PERCENTAGE = 0 
DAMAGE_INTENSITY = 0

# === FONT MAPPING ===
FONT_MAP = {
    "english":  {"sans": "NotoSans-Regular.ttf",      "serif": "NotoSerif-Regular.ttf"},
    "tamil":    {"sans": "NotoSansTamil-Regular.ttf", "serif": "NotoSerifTamil-Regular.ttf"}
}

SELECTED_LANGUAGES = list(FONT_MAP.keys())
STYLES = ["sans", "serif"]

# === LANGUAGE STATISTICS ===
LANGUAGE_STATS = {
    "english":  {"avg_len": 5.1, "space": True,  "punc": ".,?!;:"},
    "german":   {"avg_len": 6.5, "space": True,  "punc": ".,?!;:"},
    "latin":    {"avg_len": 6.0, "space": True,  "punc": ".,?!;:"},
    "chinese":  {"avg_len": 1.0, "space": False, "punc": "，。？！"}, 
    "japanese": {"avg_len": 1.0, "space": False, "punc": "、。？！"},
    "korean":   {"avg_len": 3.2, "space": True,  "punc": ".,?!;:"},
    "arabic":   {"avg_len": 5.0, "space": True,  "punc": "،.؟!;:"},
    "tamil":    {"avg_len": 7.0, "space": True,  "punc": ".,?!;:"}
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def apply_structural_damage(image_np, intensity):
    h, w = image_np.shape
    noise_small = np.random.randint(0, 255, (h//8, w//8), dtype=np.uint8)
    noise_mask = cv2.resize(noise_small, (w, h), interpolation=cv2.INTER_CUBIC)
    
    mask_float = noise_mask.astype(np.float32)
    mask_float = mask_float - 110 
    mask_float = np.clip(mask_float, 0, 255)
    
    mask_float = mask_float * intensity
    
    img_float = image_np.astype(np.float32)
    final_float = cv2.add(img_float, mask_float)
    
    final_img = np.clip(final_float, 0, 255).astype(np.uint8)
    if random.random() > 0.5:
        salt_pepper = np.random.choice([0, 255], size=(h, w), p=[0.99, 0.01])
        final_img = np.where(salt_pepper == 255, 255, final_img)

    return final_img

def get_char_pools(lang):
    numerals = [str(i) for i in range(10)]

    if lang in ["english", "latin"]:
        lower = [chr(i) for i in range(0x0061, 0x007A)] 
        upper = [chr(i) for i in range(0x0041, 0x005A)] 
        return lower, upper, numerals
    elif lang == "tamil":
        vowels = [chr(i) for i in range(0x0B85, 0x0B8B)] + [chr(i) for i in range(0x0B8E, 0x0B91)] + [chr(i) for i in range(0x0B92, 0x0B96)]
        consonants = [chr(c) for c in [0x0B95, 0x0B99, 0x0B9A, 0x0B9C, 0x0B9E, 0x0B9F, 0x0BA3, 0x0BA4, 0x0BA8, 0x0BA9, 0x0BAA, 0x0BAE, 0x0BAF, 0x0BB0, 0x0BB1, 0x0BB2, 0x0BB3, 0x0BB4, 0x0BB5, 0x0BB6, 0x0BB7, 0x0BB8, 0x0BB9]]
        return (vowels + consonants), [], numerals
    return ["X"], [], numerals

def generate_natural_text(lang, min_chars):
    stats = LANGUAGE_STATS.get(lang, LANGUAGE_STATS["english"])
    lowers, uppers, nums = get_char_pools(lang)
    output_text = ""
    current_len = 0
    
    while current_len < min_chars:
        word_len = max(1, int(random.normalvariate(stats["avg_len"], 1.5)))
        word_chars = []
        for _ in range(word_len):
            if random.random() < MIXED_NUMERALS_PERCENTAGE:
                word_chars.append(random.choice(nums))
            elif uppers and random.random() < CAPS_PERCENTAGE:
                word_chars.append(random.choice(uppers))
            else:
                word_chars.append(random.choice(lowers))
                
        word = "".join(word_chars)
        if random.random() < 0.15: 
            word += random.choice(stats["punc"])
        output_text += word
        if stats["space"]: 
            output_text += " "
        current_len = len(output_text)
    return output_text

def load_font_safe(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        raise FileNotFoundError(f"Could not find font: {path}")

def apply_noise(image):
    arr = np.array(image)
    noise = np.random.normal(NOISE_MEAN, NOISE_SIGMA, arr.shape)
    noisy = arr + noise
    return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))

# ==========================================
# 3. GENERATION ENGINE
# ==========================================

def generate_one_image(idx):
    # Re-seed random for each process to ensure true randomness across multi-cores
    random.seed()
    np.random.seed()
    
    lang = random.choice(SELECTED_LANGUAGES)
    style = random.choice(STYLES)
    num_lines = random.choice(LINES_OPTIONS)
    
    spacing_factor = random.uniform(LINE_SPACING_FACTOR_RANGE[0], LINE_SPACING_FACTOR_RANGE[1])
    usable_height = IMG_H * 0.9 
    font_size = int(usable_height / (num_lines * spacing_factor))
    line_step = int(font_size * spacing_factor)
    
    font_path = FONT_MAP[lang][style]
    font = load_font_safe(font_path, font_size)

    canvas_w, canvas_h = int(IMG_W * 1.5), int(IMG_H * 1.5)
    img = Image.new("L", (canvas_w, canvas_h), color=255)
    draw = ImageDraw.Draw(img)

    total_text_block_height = num_lines * line_step
    free_space = canvas_h - total_text_block_height
    max_shift = (free_space // 4) 
    vertical_shift = random.randint(-max_shift, max_shift)

    center_y = canvas_h // 2
    start_y = (center_y - (total_text_block_height // 2)) + vertical_shift

    for i in range(num_lines):
        y = start_y + (i * line_step)
        jitter_x = random.randint(HORIZONTAL_JITTER_RANGE[0], HORIZONTAL_JITTER_RANGE[1])
        x_pos = -50 + jitter_x
        chars_needed = int(canvas_w / (font_size * 0.4)) 
        line_text = generate_natural_text(lang, chars_needed)
        draw.text((x_pos, y), line_text, font=font, fill=0)

    if random.random() < DAMAGE_PERCENTAGE:
        img_np = np.array(img)
        img_np = apply_structural_damage(img_np, intensity=DAMAGE_INTENSITY)
        img = Image.fromarray(img_np)

    angle = random.randint(TILT_RANGE[0], TILT_RANGE[1])
    img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=255)

    left = (canvas_w - IMG_W) // 2
    top = (canvas_h - IMG_H) // 2
    img = img.crop((left, top, left + IMG_W, top + IMG_H))

    img = apply_noise(img)
    
    # Define safe Augraphy pool (Markup and Scribbles removed)
    augraphy_pool = [
        BadPhotoCopy(p=1.0), BindingsAndFasteners(p=1.0), BleedThrough(p=1.0),
        Brightness(p=1.0), BrightnessTexturize(p=1.0), ColorPaper(p=1.0),
        ColorShift(p=1.0), DelaunayTessellation(p=1.0), DepthSimulatedBlur(p=1.0),
        DirtyDrum(p=1.0), DirtyRollers(p=1.0), DirtyScreen(p=1.0),
        Dithering(p=1.0), DotMatrix(p=1.0), DoubleExposure(p=1.0),
        Faxify(p=1.0), Gamma(p=1.0), Hollow(p=1.0), InkBleed(p=1.0),
        InkMottling(p=1.0), LCDScreenPattern(p=1.0),
        Jpeg(p=1.0), LensFlare(p=1.0), Letterpress(p=1.0), LightingGradient(p=1.0),
        LinesDegradation(p=1.0), LowInkPeriodicLines(p=1.0), LowInkRandomLines(p=1.0),
        LowLightNoise(p=1.0), Moire(p=1.0), NoiseTexturize(p=1.0),
        NoisyLines(p=1.0), PatternGenerator(p=1.0), ReflectedLight(p=1.0),
        ShadowCast(p=1.0), SubtleNoise(p=1.0), VoronoiTessellation(p=1.0), 
        Folding(p=1.0), Geometric(p=1.0), 
        GlitchEffect(p=1.0), InkShifter(p=1.0), PageBorder(p=1.0), 
        Rescale(p=1.0), SectionShift(p=1.0), Squish(p=1.0)
    ]
    
    selected_augmentations = random.sample(augraphy_pool, 3)
    img_bgr = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)
    pipeline = AugraphyPipeline(ink_phase=[], paper_phase=[], post_phase=selected_augmentations)
    augmented_data = pipeline.augment(img_bgr)
    img_bgr_aug = augmented_data["output"]
    
    img_rgb_aug = cv2.cvtColor(img_bgr_aug, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb_aug).convert("L")

    filename = f"img_{idx}_{lang}_{style}.png"
    img.save(os.path.join(OUTPUT_DIR, filename))
    
    return filename

# ==========================================
# 4. MULTIPROCESSING EXECUTION
# ==========================================

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Determine optimal number of workers (leave 1 core free for OS stability)
    max_workers = max(1, multiprocessing.cpu_count() - 1)

    print(f"Generating {NUM_IMAGES} images.")
    print(f"Caps: {CAPS_PERCENTAGE*100}% | Numerals: {MIXED_NUMERALS_PERCENTAGE*100}%")
    print(f"Damage: {DAMAGE_PERCENTAGE*100}% of images | Intensity: {DAMAGE_INTENSITY}")
    print(f"🚀 Starting parallel generation using {max_workers} CPU cores...")

    # Launch the process pool
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the executor
        futures = {executor.submit(generate_one_image, i): i for i in range(NUM_IMAGES)}
        
        # Track completion
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                filename = future.result()
                completed += 1
                # Print progress every 100 images
                if completed % 100 == 0:
                    print(f"[{completed}/{NUM_IMAGES}] Processed: {filename}")
            except Exception as exc:
                idx = futures[future]
                print(f"Image {idx} generated an exception: {exc}")

    print("✅ Generation Complete.")