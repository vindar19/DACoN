# -*- coding: utf-8 -*-

"""
prepare_dacon_cut.py

Prepare animation CUT data for DACoN inference.

Input:

G:\\paint_assistant\\ref_prj\\DATA\\CUT_01
├── tga
│   ├── b0001.tga
│   ├── b0002.tga
│   ├── b0003.tga
│   └── ...
│
└── color_reference
    └── b0002.tga

Output:

G:\\paint_assistant\\ref_prj\\DATA\\CUT_01_DACON\\CUT_01
├── line
│   ├── b0001.png
│   ├── b0002.png
│   └── ...
│
└── ref
    ├── line
    │   └── b0002.png
    │
    └── gt
        └── b0002.png

IMPORTANT:

1. RGB TGA -> RGB PNG
2. No near-white -> transparent conversion
3. Existing DACoN generated data is cleaned before preparation:
       line/
       ref/
       seg/
       pred/
4. This prevents old segmentation / prediction results from being reused.
"""

import shutil
from pathlib import Path

from PIL import Image


# ============================================================
# PATH CONFIGURATION
# ============================================================

REF_PRJ = Path(
    r"G:\paint_assistant\ref_prj"
)

DATA_DIR = (
    REF_PRJ /
    "DATA"
)

# Default CUT
CUT_NAME = "CUT_01"

CUT_DIR = (
    DATA_DIR /
    CUT_NAME
)

TGA_DIR = (
    CUT_DIR /
    "tga"
)

REFERENCE_DIR = (
    CUT_DIR /
    "color_reference"
)

# DACoN output
OUTPUT_ROOT = (
    DATA_DIR /
    f"{CUT_NAME}_DACON"
)

OUTPUT_DIR = (
    OUTPUT_ROOT /
    CUT_NAME
)


# ============================================================
# UTIL
# ============================================================

def print_line():
    print("-" * 70)


def remove_directory(path):
    """
    Remove directory completely if it exists.
    """

    if path.exists():

        print(
            f"Removing old directory: {path}"
        )

        shutil.rmtree(path)

        print(
            f"Removed: {path}"
        )


def ensure_directory(path):
    """
    Create directory.
    """

    path.mkdir(
        parents=True,
        exist_ok=True
    )


def copy_tga_to_png(
    source,
    target
):
    """
    Convert TGA to PNG.

    IMPORTANT:
    RGB input stays RGB.

    We deliberately DO NOT:
        - create alpha
        - remove near-white
        - convert white to transparent
        - modify RGB values

    This is important for animation lineart.
    """

    with Image.open(source) as image:

        source_mode = image.mode
        source_size = image.size

        # ----------------------------------------------------
        # DACoN line input:
        #
        # Keep RGB as RGB.
        #
        # If the source happens to be another mode,
        # convert to RGB.
        # ----------------------------------------------------

        if image.mode != "RGB":

            image = image.convert(
                "RGB"
            )

        image.save(
            target,
            format="PNG"
        )

    return (
        source_mode,
        source_size,
        image.mode,
        image.size
    )


# ============================================================
# FIND CUT
# ============================================================

def find_tga_files():

    if not TGA_DIR.exists():

        raise FileNotFoundError(
            f"TGA directory not found:\n{TGA_DIR}"
        )

    files = sorted(
        [
            p
            for p in TGA_DIR.iterdir()
            if p.is_file()
            and p.suffix.lower() == ".tga"
        ]
    )

    if not files:

        raise RuntimeError(
            f"No TGA line frames found:\n{TGA_DIR}"
        )

    return files


# ============================================================
# FIND REFERENCE
# ============================================================

def find_reference_file(
    line_files
):
    """
    Find color reference.

    Current rule:

    - take the first TGA from color_reference
    - its filename stem is used to find the corresponding
      line frame in tga/

    Example:

        color_reference/b0002.tga

    ->

        tga/b0002.tga
    """

    if not REFERENCE_DIR.exists():

        raise FileNotFoundError(
            f"Reference directory not found:\n"
            f"{REFERENCE_DIR}"
        )

    reference_files = sorted(
        [
            p
            for p in REFERENCE_DIR.iterdir()
            if p.is_file()
            and p.suffix.lower() == ".tga"
        ]
    )

    if not reference_files:

        raise RuntimeError(
            f"No reference TGA found:\n"
            f"{REFERENCE_DIR}"
        )

    reference = reference_files[0]

    reference_stem = (
        reference.stem.lower()
    )

    reference_line = None

    for line_file in line_files:

        if (
            line_file.stem.lower()
            ==
            reference_stem
        ):

            reference_line = line_file

            break

    if reference_line is None:

        raise RuntimeError(
            "\n"
            "Reference line not found.\n\n"
            f"Reference image:\n"
            f"  {reference}\n\n"
            f"Expected corresponding line:\n"
            f"  {TGA_DIR / (reference.stem + '.tga')}\n\n"
            "Please make sure the reference image and "
            "line frame use the same filename."
        )

    return (
        reference,
        reference_line
    )


# ============================================================
# CLEAN OLD DACON DATA
# ============================================================

def clean_old_dacon_data():

    print()
    print(
        "Cleaning old DACoN generated data..."
    )

    print()

    # --------------------------------------------------------
    # IMPORTANT
    #
    # We only delete generated folders.
    #
    # We DO NOT delete:
    #
    #   CUT_01
    #   tga
    #   color_reference
    #
    # --------------------------------------------------------

    folders_to_remove = [

        OUTPUT_DIR / "line",

        OUTPUT_DIR / "ref",

        OUTPUT_DIR / "seg",

        OUTPUT_DIR / "pred",

    ]

    for folder in folders_to_remove:

        remove_directory(
            folder
        )

    print()

    print(
        "Old DACoN data cleaned."
    )


# ============================================================
# PREPARE LINE FRAMES
# ============================================================

def prepare_line_frames(
    line_files
):

    print()
    print(
        "[1/3] Preparing target line frames"
    )

    print()

    output_line_dir = (
        OUTPUT_DIR /
        "line"
    )

    ensure_directory(
        output_line_dir
    )

    for line_file in line_files:

        print(
            line_file.name
        )

        output_file = (
            output_line_dir /
            f"{line_file.stem}.png"
        )

        (
            source_mode,
            source_size,
            output_mode,
            output_size
        ) = copy_tga_to_png(
            line_file,
            output_file
        )

        print(
            f"  source mode : {source_mode}"
        )

        print(
            f"  source size : {source_size}"
        )

        print(
            f"  output      : {output_file}"
        )

        print(
            f"  output mode : {output_mode}"
        )

        print(
            f"  output size : {output_size}"
        )

        print()


# ============================================================
# PREPARE REFERENCE LINE
# ============================================================

def prepare_reference_line(
    reference_line
):

    print(
        "[2/3] Preparing reference line"
    )

    print()

    output_ref_line_dir = (
        OUTPUT_DIR /
        "ref" /
        "line"
    )

    ensure_directory(
        output_ref_line_dir
    )

    output_file = (
        output_ref_line_dir /
        f"{reference_line.stem}.png"
    )

    (
        source_mode,
        source_size,
        output_mode,
        output_size
    ) = copy_tga_to_png(
        reference_line,
        output_file
    )

    print(
        reference_line.name
    )

    print(
        f"  source mode : {source_mode}"
    )

    print(
        f"  source size : {source_size}"
    )

    print(
        f"  output      : {output_file}"
    )

    print(
        f"  output mode : {output_mode}"
    )

    print(
        f"  output size : {output_size}"
    )

    print()


# ============================================================
# PREPARE REFERENCE COLOR
# ============================================================

def prepare_reference_color(
    reference
):

    print(
        "[3/3] Preparing reference color image"
    )

    print()

    output_gt_dir = (
        OUTPUT_DIR /
        "ref" /
        "gt"
    )

    ensure_directory(
        output_gt_dir
    )

    output_file = (
        output_gt_dir /
        f"{reference.stem}.png"
    )

    (
        source_mode,
        source_size,
        output_mode,
        output_size
    ) = copy_tga_to_png(
        reference,
        output_file
    )

    print(
        reference.name
    )

    print(
        f"  source mode : {source_mode}"
    )

    print(
        f"  source size : {source_size}"
    )

    print(
        f"  output      : {output_file}"
    )

    print(
        f"  output mode : {output_mode}"
    )

    print(
        f"  output size : {output_size}"
    )

    print()


# ============================================================
# COUNT PNG
# ============================================================

def count_png_files():

    if not OUTPUT_ROOT.exists():

        return []

    return sorted(
        OUTPUT_ROOT.rglob(
            "*.png"
        )
    )


# ============================================================
# PRINT STRUCTURE
# ============================================================

def print_structure():

    print()
    print(
        "Expected structure"
    )

    print()

    print(
        OUTPUT_DIR
    )

    print(
        "├── line"
    )

    line_dir = (
        OUTPUT_DIR /
        "line"
    )

    line_files = sorted(
        line_dir.glob(
            "*.png"
        )
    )

    for index, file in enumerate(
        line_files
    ):

        if index == len(line_files) - 1:

            prefix = "│   └── "

        else:

            prefix = "│   ├── "

        print(
            prefix +
            file.name
        )

    print(
        "│"
    )

    print(
        "└── ref"
    )

    print(
        "    ├── line"
    )

    ref_line_dir = (
        OUTPUT_DIR /
        "ref" /
        "line"
    )

    ref_line_files = sorted(
        ref_line_dir.glob(
            "*.png"
        )
    )

    for index, file in enumerate(
        ref_line_files
    ):

        if index == len(ref_line_files) - 1:

            prefix = "    │   └── "

        else:

            prefix = "    │   ├── "

        print(
            prefix +
            file.name
        )

    print(
        "    │"
    )

    print(
        "    └── gt"
    )

    gt_dir = (
        OUTPUT_DIR /
        "ref" /
        "gt"
    )

    gt_files = sorted(
        gt_dir.glob(
            "*.png"
        )
    )

    for index, file in enumerate(
        gt_files
    ):

        if index == len(gt_files) - 1:

            prefix = "        └── "

        else:

            prefix = "        ├── "

        print(
            prefix +
            file.name
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "DACoN CUT Preparation"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"CUT_DIR       : {CUT_DIR}"
    )

    print(
        f"TGA_DIR       : {TGA_DIR}"
    )

    print(
        f"REFERENCE_DIR : {REFERENCE_DIR}"
    )

    print(
        f"OUTPUT_DIR    : {OUTPUT_DIR}"
    )

    print()

    # ========================================================
    # CHECK INPUT
    # ========================================================

    if not CUT_DIR.exists():

        raise FileNotFoundError(
            f"CUT directory not found:\n{CUT_DIR}"
        )

    line_files = (
        find_tga_files()
    )

    (
        reference,
        reference_line
    ) = find_reference_file(
        line_files
    )

    # ========================================================
    # INPUT INFORMATION
    # ========================================================

    print_line()

    print(
        "Input information"
    )

    print_line()

    print(
        f"Line frames      : {len(line_files)}"
    )

    for file in line_files:

        print(
            f"  - {file.name}"
        )

    print()

    print(
        f"Reference        : {reference.name}"
    )

    print(
        f"Reference line   : {reference_line.name}"
    )

    print()

    # ========================================================
    # CLEAN OLD DATA
    # ========================================================

    clean_old_dacon_data()

    # ========================================================
    # PREPARE
    # ========================================================

    prepare_line_frames(
        line_files
    )

    prepare_reference_line(
        reference_line
    )

    prepare_reference_color(
        reference
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print(
        "=" * 70
    )

    print(
        "PREPARATION COMPLETE"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "DACoN input directory:"
    )

    print(
        OUTPUT_DIR
    )

    # ========================================================
    # GENERATED FILES
    # ========================================================

    png_files = (
        count_png_files()
    )

    print()

    print(
        "Generated files:"
    )

    print()

    for file in png_files:

        print(
            file
        )

    print()

    print(
        f"Total PNG files: {len(png_files)}"
    )

    # ========================================================
    # STRUCTURE
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print_structure()

    # ========================================================
    # NEXT STEP
    # ========================================================

    print()

    print(
        "Next step:"
    )

    print(
        "python dacon\\inference.py "
        "--config configs\\inference.yaml "
        "--model checkpoints\\dacon_v1_1.pth "
        f"--data {OUTPUT_ROOT} "
        "--version 1_1"
    )

    print()


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as e:

        print()

        print(
            "=" * 70
        )

        print(
            "ERROR"
        )

        print(
            "=" * 70
        )

        print()

        print(
            str(e)
        )

        print()

        raise