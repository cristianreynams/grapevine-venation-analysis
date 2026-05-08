import cv2
import os
import argparse
import matplotlib.pyplot as plt

# =====================================
# CONFIG
# =====================================

ROOT = "data/processed"

PIPELINE = [

    {
        "title": "Crop",
        "folder": "cropped",
        "prefix": "crop_",
        "grayscale": False
    },

    {
        "title": "Mask",
        "folder": "masks",
        "prefix": "mask_crop_",
        "grayscale": True
    },

    {
        "title": "Isolated",
        "folder": "isolated",
        "prefix": "isolated_crop_",
        "grayscale": False
    },

    {
        "title": "Gray",
        "folder": "grayscale",
        "prefix": "gray_isolated_crop_",
        "grayscale": True
    },

    {
        "title": "CLAHE",
        "folder": "enhanced",
        "prefix": "enhanced_gray_isolated_crop_",
        "grayscale": True
    },

    {
        "title": "Frangi",
        "folder": "frangi",
        "prefix": "frangi_enhanced_gray_isolated_crop_",
        "grayscale": True
    },

    {
        "title": "Binary",
        "folder": "binary",
        "prefix": "binary_frangi_enhanced_gray_isolated_crop_",
        "grayscale": True
    },

    {
        "title": "Skeleton",
        "folder": "skeleton",
        "prefix": "skeleton_binary_frangi_enhanced_gray_isolated_crop_",
        "grayscale": True
    }
]

# =====================================
# LOAD IMAGE
# =====================================

def load_image(path, grayscale=False):

    if not os.path.exists(path):

        return None

    if grayscale:

        image = cv2.imread(
            path,
            cv2.IMREAD_GRAYSCALE
        )

    else:

        image = cv2.imread(path)

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

    return image

# =====================================
# VISUALIZE PIPELINE
# =====================================

def visualize_pipeline(folder, filename):

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(16,8)
    )

    axes = axes.flatten()

    for i, step in enumerate(PIPELINE):

        image_path = os.path.join(
            ROOT,
            step["folder"],
            folder,
            f"{step['prefix']}{filename}"
        )

        image = load_image(
            image_path,
            grayscale=step["grayscale"]
        )

        if image is None:

            axes[i].text(
                0.5,
                0.5,
                "Missing",
                ha="center",
                va="center",
                fontsize=12
            )

            axes[i].set_title(
                step["title"]
            )

            axes[i].axis("off")

            continue

        if step["grayscale"]:

            axes[i].imshow(
                image,
                cmap="gray"
            )

        else:

            axes[i].imshow(image)

        axes[i].set_title(
            step["title"],
            fontsize=10
        )

        axes[i].axis("off")

    plt.suptitle(
        filename,
        fontsize=14
    )

    plt.tight_layout()

    output_dir = os.path.join(
        "outputs",
        "pipeline_visualization"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    save_path = os.path.join(
        output_dir,
        f"pipeline_{os.path.splitext(filename)[0]}.png"
    )

    plt.savefig(
        save_path,
        bbox_inches="tight"
    )

    plt.show()

    print(f"Saved visualization: {save_path}")

# =====================================
# MAIN
# =====================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--folder",
    required=True,
    type=str
)

parser.add_argument(
    "--image",
    required=True,
    type=str
)

args = parser.parse_args()

visualize_pipeline(
    args.folder,
    args.image
)