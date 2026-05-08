import cv2
import os
import argparse
import matplotlib.pyplot as plt

# =====================================
# CONFIG
# =====================================

INPUT_ROOT = "data/processed/isolated"

# =====================================
# CORE FUNCTION
# =====================================

def convert_to_grayscale(image_path):

    # =====================================
    # LOAD IMAGE
    # =====================================

    image = cv2.imread(image_path)

    if image is None:

        raise ValueError(
            f"Could not load image: {image_path}"
        )

    # =====================================
    # BGR -> GRAYSCALE
    # =====================================

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return {
        "original": image,
        "gray": gray
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
        "data/processed/grayscale",
        "debug_img"
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
        and not f.startswith("debug")
    ])[:4]

    results = []

    for file in image_files:

        image_path = os.path.join(
            input_dir,
            file
        )

        result = convert_to_grayscale(
            image_path
        )

        # =====================================
        # SAVE GRAYSCALE
        # =====================================

        save_path = os.path.join(
            output_dir,
            f"gray_{file}"
        )

        save_image(
            result["gray"],
            save_path
        )

        original_rgb = cv2.cvtColor(
            result["original"],
            cv2.COLOR_BGR2RGB
        )

        results.append((
            file,
            original_rgb,
            result["gray"]
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

    for i, (name, original, gray) in enumerate(results):

        # =====================================
        # ORIGINAL
        # =====================================

        axes[i * 2].imshow(original)

        axes[i * 2].set_title(
            f"{name}\nOriginal",
            fontsize=9
        )

        axes[i * 2].axis("off")

        # =====================================
        # GRAYSCALE
        # =====================================

        axes[i * 2 + 1].imshow(
            gray,
            cmap="gray"
        )

        axes[i * 2 + 1].set_title(
            f"{name}\nGrayscale",
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
        "data/processed/grayscale",
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

        result = convert_to_grayscale(
            image_path
        )

        save_path = os.path.join(
            output_dir,
            f"gray_{file}"
        )

        save_image(
            result["gray"],
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