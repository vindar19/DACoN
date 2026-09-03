# -*- coding: utf-8 -*-

"""
prepare_dacon_cut.py

Prepare animation CUT data for DACoN inference.

DACoN directory layout used by Paint Assistant:

    G:\paint_assistant\DATA\CUT_01_DACON\
    ├── tga\
    │   ├── b0001.tga
    │   ├── b0002.tga
    │   └── ...
    ├── color_reference\
    │   └── b0002.tga
    │
    └── CUT_01\
        ├── line\
        │   ├── b0001.png
        │   ├── b0002.png
        │   └── ...
        ├── ref\
        │   ├── line\
        │   │   └── b0002.png
        │   ├── gt\
        │   │   └── b0002.png
        │   └── seg\
        ├── seg\
        └── pred\

IMPORTANT:
1. The DACoN root is supplied with --cut-dir.
2. The input TGA files are inside <CUT>_DACON/tga.
3. The color reference is inside <CUT>_DACON/color_reference.
4. The generated DACoN data is inside <CUT>_DACON/CUT_XX.
5. No sibling DATA/CUT_XX directory is created by this script.
6. RGB TGA -> RGB PNG.
7. No near-white -> transparent conversion.
8. Existing generated line/ref/seg/pred data is cleaned before preparation.
"""

import argparse
import shutil
from pathlib import Path

from PIL import Image


# ============================================================
# PATH
# ============================================================

def normalize_dacon_root(path):
    """
    Accept either:

        DATA/CUT_01_DACON

    or, for convenience:

        DATA/CUT_01_DACON/CUT_01

    and always return:

        DATA/CUT_01_DACON
    """
    path = Path(path).expanduser().resolve()

    if path.name.upper().endswith("_DACON"):
        return path

    if path.parent.name.upper() == f"{path.name}_DACON":
        return path.parent

    # Do not silently create DATA/CUT_XX.
    # If a normal CUT path is supplied, derive its sibling DACoN root.
    return path.parent / f"{path.name}_DACON"


def get_cut_name(dacon_root):
    name = dacon_root.name
    if not name.upper().endswith("_DACON"):
        raise ValueError(
            f"Invalid DACoN root directory:\n{dacon_root}\n\n"
            "Expected a directory named like CUT_01_DACON."
        )
    return name[:-6]


# ============================================================
# UTIL
# ============================================================

def print_line():
    print("-" * 70)


def remove_directory(path):
    path = Path(path)

    if path.exists():
        print(f"Removing old directory: {path}")
        shutil.rmtree(path)
        print(f"Removed: {path}")


def ensure_directory(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def copy_tga_to_png(source, target):
    """
    Convert TGA to PNG without changing RGB values.

    We deliberately do NOT:
        - create alpha
        - remove near-white
        - convert white to transparent
        - alter RGB values
    """
    with Image.open(source) as image:
        source_mode = image.mode
        source_size = image.size

        if image.mode != "RGB":
            image = image.convert("RGB")

        output_mode = image.mode
        output_size = image.size

        image.save(target, format="PNG")

    return (
        source_mode,
        source_size,
        output_mode,
        output_size,
    )


# ============================================================
# INPUT
# ============================================================

def find_tga_files(tga_dir):
    tga_dir = Path(tga_dir)

    if not tga_dir.exists():
        raise FileNotFoundError(
            f"TGA directory not found:\n{tga_dir}"
        )

    files = sorted(
        [
            p
            for p in tga_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() == ".tga"
        ],
        key=lambda p: p.name.lower(),
    )

    if not files:
        raise RuntimeError(
            f"No TGA line frames found:\n{tga_dir}"
        )

    return files


def find_reference_file(reference_dir, line_files):
    """
    Current rule:

    - take the first TGA from color_reference
    - use its filename stem to find the corresponding
      line frame in tga/

    Example:
        color_reference/b0002.tga
            ->
        tga/b0002.tga
    """
    reference_dir = Path(reference_dir)

    if not reference_dir.exists():
        raise FileNotFoundError(
            f"Reference directory not found:\n{reference_dir}"
        )

    reference_files = sorted(
        [
            p
            for p in reference_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() == ".tga"
        ],
        key=lambda p: p.name.lower(),
    )

    if not reference_files:
        raise RuntimeError(
            f"No reference TGA found:\n{reference_dir}"
        )

    reference = reference_files[0]
    reference_stem = reference.stem.lower()

    reference_line = None

    for line_file in line_files:
        if line_file.stem.lower() == reference_stem:
            reference_line = line_file
            break

    if reference_line is None:
        raise RuntimeError(
            "\nReference line not found.\n\n"
            f"Reference image:\n"
            f"  {reference}\n\n"
            f"Expected corresponding line:\n"
            f"  {Path(line_files[0]).parent / (reference.stem + '.tga')}\n\n"
            "Please make sure the reference image and "
            "line frame use the same filename."
        )

    return reference, reference_line


# ============================================================
# CLEAN GENERATED DATA
# ============================================================

def clean_old_dacon_data(output_dir):
    """
    Only delete DACoN generated data.

    Never delete:
        <DACON_ROOT>/tga
        <DACON_ROOT>/color_reference
    """
    print()
    print("Cleaning old DACoN generated data...")
    print()

    folders_to_remove = [
        output_dir / "line",
        output_dir / "ref",
        output_dir / "seg",
        output_dir / "pred",
    ]

    for folder in folders_to_remove:
        remove_directory(folder)

    print()
    print("Old DACoN data cleaned.")


# ============================================================
# PREPARE LINE
# ============================================================

def prepare_line_frames(line_files, output_dir):
    print()
    print("[1/3] Preparing target line frames")
    print()

    output_line_dir = output_dir / "line"
    ensure_directory(output_line_dir)

    for line_file in line_files:
        print(line_file.name)

        output_file = output_line_dir / f"{line_file.stem}.png"

        (
            source_mode,
            source_size,
            output_mode,
            output_size,
        ) = copy_tga_to_png(line_file, output_file)

        print(f"  source mode : {source_mode}")
        print(f"  source size : {source_size}")
        print(f"  output      : {output_file}")
        print(f"  output mode : {output_mode}")
        print(f"  output size : {output_size}")
        print()


# ============================================================
# PREPARE REFERENCE LINE
# ============================================================

def prepare_reference_line(reference_line, output_dir):
    print("[2/3] Preparing reference line")
    print()

    output_ref_line_dir = output_dir / "ref" / "line"
    ensure_directory(output_ref_line_dir)

    output_file = (
        output_ref_line_dir /
        f"{reference_line.stem}.png"
    )

    (
        source_mode,
        source_size,
        output_mode,
        output_size,
    ) = copy_tga_to_png(reference_line, output_file)

    print(reference_line.name)
    print(f"  source mode : {source_mode}")
    print(f"  source size : {source_size}")
    print(f"  output      : {output_file}")
    print(f"  output mode : {output_mode}")
    print(f"  output size : {output_size}")
    print()


# ============================================================
# PREPARE REFERENCE COLOR
# ============================================================

def prepare_reference_color(reference, output_dir):
    print("[3/3] Preparing reference color image")
    print()

    output_gt_dir = output_dir / "ref" / "gt"
    ensure_directory(output_gt_dir)

    output_file = (
        output_gt_dir /
        f"{reference.stem}.png"
    )

    (
        source_mode,
        source_size,
        output_mode,
        output_size,
    ) = copy_tga_to_png(reference, output_file)

    print(reference.name)
    print(f"  source mode : {source_mode}")
    print(f"  source size : {source_size}")
    print(f"  output      : {output_file}")
    print(f"  output mode : {output_mode}")
    print(f"  output size : {output_size}")
    print()


# ============================================================
# COUNT / STRUCTURE
# ============================================================

def count_png_files(output_root):
    output_root = Path(output_root)

    if not output_root.exists():
        return []

    return sorted(
        output_root.rglob("*.png"),
        key=lambda p: str(p).lower(),
    )


def print_structure(output_dir):
    print()
    print("Expected structure")
    print()

    print(output_dir)
    print("├── line")

    line_dir = output_dir / "line"
    line_files = sorted(
        line_dir.glob("*.png"),
        key=lambda p: p.name.lower(),
    )

    for index, file in enumerate(line_files):
        prefix = (
            "│   └── "
            if index == len(line_files) - 1
            else "│   ├── "
        )
        print(prefix + file.name)

    print("│")
    print("└── ref")
    print("    ├── line")

    ref_line_dir = output_dir / "ref" / "line"
    ref_line_files = sorted(
        ref_line_dir.glob("*.png"),
        key=lambda p: p.name.lower(),
    )

    for index, file in enumerate(ref_line_files):
        prefix = (
            "    │   └── "
            if index == len(ref_line_files) - 1
            else "    │   ├── "
        )
        print(prefix + file.name)

    print("    │")
    print("    └── gt")

    gt_dir = output_dir / "ref" / "gt"
    gt_files = sorted(
        gt_dir.glob("*.png"),
        key=lambda p: p.name.lower(),
    )

    for index, file in enumerate(gt_files):
        prefix = (
            "        └── "
            if index == len(gt_files) - 1
            else "        ├── "
        )
        print(prefix + file.name)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare a DACoN CUT."
    )
    parser.add_argument(
        "--cut-dir",
        required=True,
        help=(
            "DACoN root directory, e.g. "
            r"G:\paint_assistant\DATA\CUT_01_DACON"
        ),
    )

    args = parser.parse_args()

    dacon_root = normalize_dacon_root(args.cut_dir)

    if dacon_root.name.upper().endswith("_DACON") is False:
        raise ValueError(
            f"Invalid DACoN root:\n{dacon_root}"
        )

    cut_name = get_cut_name(dacon_root)

    # IMPORTANT:
    # Input and output are both contained inside CUT_XX_DACON.
    tga_dir = dacon_root / "tga"
    reference_dir = dacon_root / "color_reference"
    output_root = dacon_root
    output_dir = dacon_root / cut_name

    print("=" * 70)
    print("DACoN CUT Preparation")
    print("=" * 70)
    print()

    print(f"DACON_ROOT    : {dacon_root}")
    print(f"TGA_DIR       : {tga_dir}")
    print(f"REFERENCE_DIR : {reference_dir}")
    print(f"OUTPUT_DIR    : {output_dir}")
    print()

    # Safety check: this script must not write to a sibling CUT_XX.
    sibling_basic_cut = dacon_root.parent / cut_name

    print(f"Sibling CUT check: {sibling_basic_cut}")

    if sibling_basic_cut.exists():
        print(
            "NOTE: A sibling CUT directory already exists. "
            "This script will NOT modify or delete it."
        )

    print()

    if not dacon_root.exists():
        raise FileNotFoundError(
            f"DACoN root directory not found:\n{dacon_root}\n\n"
            "Create the DACoN template first."
        )

    line_files = find_tga_files(tga_dir)

    reference, reference_line = find_reference_file(
        reference_dir,
        line_files,
    )

    print_line()
    print("Input information")
    print_line()

    print(f"Line frames      : {len(line_files)}")
    for file in line_files:
        print(f"  - {file.name}")

    print()
    print(f"Reference        : {reference.name}")
    print(f"Reference line   : {reference_line.name}")
    print()

    # Clean only generated DACoN data inside CUT_XX_DACON/CUT_XX.
    clean_old_dacon_data(output_dir)

    # Prepare
    prepare_line_frames(
        line_files,
        output_dir,
    )

    prepare_reference_line(
        reference_line,
        output_dir,
    )

    prepare_reference_color(
        reference,
        output_dir,
    )

    print("=" * 70)
    print("PREPARATION COMPLETE")
    print("=" * 70)
    print()

    print("DACoN input directory:")
    print(output_root)
    print()

    png_files = count_png_files(output_dir)

    print("Generated files:")
    print()

    for file in png_files:
        print(file)

    print()
    print(f"Total PNG files: {len(png_files)}")

    print()
    print("=" * 70)
    print_structure(output_dir)

    print()
    print("=" * 70)
    print("Next step:")
    print(
        "python dacon\\inference.py "
        "--config configs\\inference.yaml "
        "--model checkpoints\\dacon_v1_1.pth "
        f"--data {output_root} "
        "--version 1_1"
    )
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print()
        print("=" * 70)
        print("ERROR")
        print("=" * 70)
        print()
        print(str(e))
        print()
        raise
