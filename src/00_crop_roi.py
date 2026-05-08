import cv2
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# =====================================
# CONFIG
# =====================================

DATASET_ROOT = "data/raw/pasini59"

# HSV threshold
LOWER_GREEN = np.array([25, 50, 40])
UPPER_GREEN = np.array([95, 255, 255])

# Morphology
KERNEL_SIZE = 15

# Padding around detected ROI
PADDING = 60

# =====================================
# CORE FUNCTION
# =====================================

def crop_roi(image_path):

    # =====================================
    # LOAD IMAGE
    # =====================================

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Could not load {image_path}")

    height, width = img.shape[:2]

    # =====================================
    # RGB -> GAUSSIAN BLUR
    # =====================================

    blur = cv2.GaussianBlur(
        img,
        (5, 5),
        0
    )

    # =====================================
    # BLUR -> HSV
    # =====================================

    hsv = cv2.cvtColor(
        blur,
        cv2.COLOR_BGR2HSV
    )

    # =====================================
    # HSV -> THRESHOLD
    # green + saturation
    # =====================================

    mask = cv2.inRange(
        hsv,
        LOWER_GREEN,
        UPPER_GREEN
    )

    # =====================================
    # MORPHOLOGY
    # =====================================

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (KERNEL_SIZE, KERNEL_SIZE)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # =====================================
    # LARGEST CONNECTED COMPONENT
    # =====================================

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    if num_labels <= 1:
        raise ValueError(
            f"No foreground detected: {image_path}"
        )

    largest_label = 1 + np.argmax(
        stats[1:, cv2.CC_STAT_AREA]
    )

    largest_mask = np.uint8(
        labels == largest_label
    ) * 255

    # =====================================
    # BOUNDING BOX
    # =====================================

    coords = cv2.findNonZero(
        largest_mask
    )

    x, y, w, h = cv2.boundingRect(
        coords
    )

    # =====================================
    # PADDED CROP
    # =====================================

    x1 = max(x - PADDING, 0)
    y1 = max(y - PADDING, 0)

    x2 = min(x + w + PADDING, width)
    y2 = min(y + h + PADDING, height)

    crop = img[y1:y2, x1:x2]

    return {
        "original": img,
        "blur": blur,
        "mask": mask,
        "largest_mask": largest_mask,
        "crop": crop
    }

# =====================================
# SAVE FUNCTION
# =====================================

def save_image(image, path):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    cv2.imwrite(path, image)

# =====================================
# DEBUG MODE
# =====================================

def debug_mode(folder_name):

    input_dir = os.path.join(
        DATASET_ROOT,
        folder_name
    )

    output_dir = os.path.join(
        "data/processed/cropped",
        folder_name
    )

    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith((
            ".jpg",
            ".jpeg",
            ".png",
            ".tif",
            ".tiff"
        ))
    ])[:4]

    results = []

    for file in image_files:

        image_path = os.path.join(
            input_dir,
            file
        )

        result = crop_roi(image_path)

        # =====================================
        # SAVE CROPS
        # =====================================

        save_path = os.path.join(
            output_dir,
            f"crop_{file}"
        )

        save_image(
            result["crop"],
            save_path
        )

        crop_rgb = cv2.cvtColor(
            result["crop"],
            cv2.COLOR_BGR2RGB
        )

        results.append((file, crop_rgb))

    # =====================================
    # VISUALIZATION
    # =====================================

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10,10)
    )

    axes = axes.flatten()

    for ax, (name, image) in zip(axes, results):

        ax.imshow(image)

        ax.set_title(name)

        ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "debug_grid.png"
        )
    )

    plt.show()

    print(f"Debug completed: {folder_name}")

# =====================================
# FOLDER MODE
# =====================================

def folder_mode(folder_name):

    input_dir = os.path.join(
        DATASET_ROOT,
        folder_name
    )

    output_dir = os.path.join(
        "data/processed/cropped",
        folder_name
    )

    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith((
            ".jpg",
            ".jpeg",
            ".png",
            ".tif",
            ".tiff"
        ))
    ])

    for file in image_files:

        image_path = os.path.join(
            input_dir,
            file
        )

        result = crop_roi(image_path)

        save_path = os.path.join(
            output_dir,
            f"crop_{file}"
        )

        save_image(
            result["crop"],
            save_path
        )

    print(f"Folder processed: {folder_name}")

# =====================================
# DATASET MODE
# =====================================

def dataset_mode():

    folders = sorted(os.listdir(DATASET_ROOT))

    for folder in folders:

        folder_path = os.path.join(
            DATASET_ROOT,
            folder
        )

        if not os.path.isdir(folder_path):
            continue

        folder_mode(folder)

    print("Dataset completed")

# =====================================
# MAIN
# =====================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--mode",
    required=True,
    choices=[
        "debug",
        "folder",
        "dataset"
    ]
)

parser.add_argument(
    "--folder",
    type=str
)

args = parser.parse_args()

# =====================================
# EXECUTION
# =====================================

if args.mode == "debug":

    if args.folder is None:

        raise ValueError(
            "--folder required in debug mode"
        )

    debug_mode(args.folder)

elif args.mode == "folder":

    if args.folder is None:

        raise ValueError(
            "--folder required in folder mode"
        )

    folder_mode(args.folder)

elif args.mode == "dataset":

    dataset_mode()