import cv2
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from skimage.filters import frangi

# =====================================
# CONFIG
# =====================================

INPUT_ROOT = "data/processed/enhanced"

# Frangi parameters
FRANGI_SIGMAS = range(1, 5)
FRANGI_ALPHA = 0.5
FRANGI_BETA = 0.5
FRANGI_GAMMA = 8

# =====================================
# CORE FUNCTION
# =====================================

def apply_frangi(image_path):

    # =====================================
    # LOAD IMAGE
    # =====================================

    gray = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if gray is None:

        raise ValueError(
            f"Could not load image: {image_path}"
        )

    # =====================================
    # NORMALIZE
    # skimage expects float [0,1]
    # =====================================

    gray_float = gray.astype(
        np.float32
    ) / 255.0

    # =====================================
    # GAUSSIAN BLUR
    # Reduce isotropic texture noise
    # =====================================

    gray_float = cv2.GaussianBlur(
        gray_float,
        (3,3),
        0
    )

    gray_float = 1.0 - gray_float

    # =====================================
    # FRANGI FILTER
    # =====================================

    vesselness = frangi(
        gray_float,
        sigmas=FRANGI_SIGMAS,
        alpha=FRANGI_ALPHA,
        beta=FRANGI_BETA,
        gamma=FRANGI_GAMMA,
        black_ridges=True
    )

    # =====================================
    # NORMALIZE OUTPUT
    # =====================================

    vesselness = cv2.normalize(
        vesselness,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    vesselness = vesselness.astype(
        np.uint8
    )

    return {
        "gray": gray,
        "frangi": vesselness
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
        INPUT_ROOT,
        folder_name
    )

    output_dir = os.path.join(
        "data/processed/frangi",
        "debug_img"
    )

    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(input_dir)
        if (
            f.lower().endswith((
                ".jpg",
                ".jpeg",
                ".png",
                ".tif",
                ".tiff"
            ))
            and not f.startswith("debug")
        )
    ])[:4]

    results = []

    for file in image_files:

        image_path = os.path.join(
            input_dir,
            file
        )

        result = apply_frangi(
            image_path
        )

        # =====================================
        # SAVE FRANGI
        # =====================================

        save_path = os.path.join(
            output_dir,
            f"frangi_{file}"
        )

        save_image(
            result["frangi"],
            save_path
        )

        results.append((
            file,
            result["gray"],
            result["frangi"]
        ))

    # =====================================
    # VISUALIZATION
    # =====================================

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(12, 6)
    )

    axes = axes.flatten()

    for i, (name, gray, frangi_img) in enumerate(results):

        # =====================================
        # ENHANCED INPUT
        # =====================================

        axes[i * 2].imshow(
            gray,
            cmap="gray"
        )

        axes[i * 2].set_title(
            f"{name}\nEnhanced",
            fontsize=9
        )

        axes[i * 2].axis("off")

        # =====================================
        # FRANGI
        # =====================================

        axes[i * 2 + 1].imshow(
            frangi_img,
            cmap="gray"
        )

        axes[i * 2 + 1].set_title(
            f"{name}\nFrangi",
            fontsize=9
        )

        axes[i * 2 + 1].axis("off")

    plt.tight_layout(
        pad=1.0,
        w_pad=0.5,
        h_pad=1.0
    )

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
        INPUT_ROOT,
        folder_name
    )

    output_dir = os.path.join(
        "data/processed/frangi",
        folder_name
    )

    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(input_dir)
        if (
            f.lower().endswith((
                ".jpg",
                ".jpeg",
                ".png",
                ".tif",
                ".tiff"
            ))
            and not f.startswith("debug")
        )
    ])

    for file in image_files:

        image_path = os.path.join(
            input_dir,
            file
        )

        result = apply_frangi(
            image_path
        )

        save_path = os.path.join(
            output_dir,
            f"frangi_{file}"
        )

        save_image(
            result["frangi"],
            save_path
        )

    print(f"Folder processed: {folder_name}")

# =====================================
# DATASET MODE
# =====================================

def dataset_mode():

    folders = sorted(os.listdir(INPUT_ROOT))

    for folder in folders:

        folder_path = os.path.join(
            INPUT_ROOT,
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