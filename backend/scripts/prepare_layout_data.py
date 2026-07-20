"""Converts IAMonDo InkML documents into rasterized page images + YOLO bbox labels
for training the text/math layout detector. (Pranav)

Dataset: IAMonDo-database (Indermuehle et al., 2010) — online handwritten documents
with `Textblock` and `Formula` content zones (among others we don't use), stored as
InkML pen-stroke traces rather than images. See:
http://www.iapr-tc11.org/mediawiki/index.php/IAM_Online_Document_Database_(IAMonDo-database)

InkML encodes trace points with the W3C compact delta scheme (a value may be prefixed
with `!` = explicit/absolute, `'` = first difference, `"` = second difference; an
unprefixed value repeats the last prefix seen for that channel, starting at `!`).
We decode X/Y through that state machine but discard the other channels (T, F).

We render every stroke in a document to build a realistic-looking full page image,
but only emit YOLO labels for `Textblock` (class 0, "text") and `Formula` (class 1,
"math") zones — matching this project's Block.type of "text" | "math". Rendering uses
raw device coordinates directly rather than applying the file's `canvasTransform`
(which maps device space to a "Page Bounds" annotation we don't otherwise need): since
every trace in a document shares the same raw coordinate frame, staying in that frame
keeps strokes and the bounding boxes derived from them mutually consistent, which is
all a detector-training pipeline requires.
"""

import argparse
import os
import random
import re
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw

CLASSES = {"Textblock": 0, "Formula": 1}  # -> ("text", "math")

_XML_NS = "{http://www.w3.org/XML/1998/namespace}"
_TOKEN_RE = re.compile(r"(['\"!]?)(-?(?:\d+\.\d*|\.\d+|\d+))")


def _num_channels(root: ET.Element) -> int:
    trace_format = root.find(".//traceFormat")
    return len(trace_format.findall("channel")) if trace_format is not None else 4


def _parse_trace_xy(trace_text: str, num_channels: int) -> tuple[list[float], list[float]]:
    """Decode a trace's compact InkML point string into (x, y) coordinate lists."""
    xs: list[float] = []
    ys: list[float] = []
    abs_vals = [0.0] * num_channels
    last_deltas = [0.0] * num_channels
    last_prefix = ["!"] * num_channels

    for point_str in (trace_text or "").strip().split(","):
        point_str = point_str.strip()
        if not point_str:
            continue
        matches = _TOKEN_RE.findall(point_str)
        if len(matches) != num_channels:
            continue  # malformed/truncated point, skip it

        for i, (prefix, num_str) in enumerate(matches):
            value = float(num_str)
            effective_prefix = prefix or last_prefix[i]
            last_prefix[i] = effective_prefix

            if effective_prefix == "!":
                abs_vals[i] = value
                last_deltas[i] = 0.0
            elif effective_prefix == "'":
                abs_vals[i] += value
                last_deltas[i] = value
            elif effective_prefix == '"':
                last_deltas[i] += value
                abs_vals[i] += last_deltas[i]

        xs.append(abs_vals[0])
        ys.append(abs_vals[1])

    return xs, ys


def _collect_trace_refs(trace_view: ET.Element) -> list[str]:
    """Recursively collect every traceDataRef ("#tN") under a traceView, without the '#'."""
    refs = []
    for elem in trace_view.iter("traceView"):
        ref = elem.get("traceDataRef")
        if ref:
            refs.append(ref.lstrip("#"))
    return refs


def _find_zones(root: ET.Element) -> list[tuple[str, list[str]]]:
    """Return (zone_type, trace_ids) for each top-level content zone under the Document node."""
    zones = []
    for top_level in root.findall("traceView"):
        annotation = top_level.find("annotation")
        if annotation is None or annotation.text != "Document":
            continue
        for zone in top_level.findall("traceView"):
            zone_annotation = zone.find("annotation")
            zone_type = zone_annotation.text if zone_annotation is not None else None
            zones.append((zone_type, _collect_trace_refs(zone)))
    return zones


def _convert_document(inkml_path: str, images_dir: str, labels_dir: str, scale: float, padding: int) -> bool:
    """Render one InkML document and write its image + YOLO label file. Returns False if skipped."""
    try:
        root = ET.parse(inkml_path).getroot()
    except ET.ParseError:
        return False

    num_channels = _num_channels(root)
    trace_points = {}
    for trace in root.iter("trace"):
        trace_id = trace.get(f"{_XML_NS}id") or trace.get("id")
        if trace_id is None:
            continue
        trace_points[trace_id] = _parse_trace_xy(trace.text, num_channels)

    all_x = [x for xs, _ in trace_points.values() for x in xs]
    all_y = [y for _, ys in trace_points.values() for y in ys]
    if not all_x or not all_y:
        return False

    min_x, min_y = min(all_x), min(all_y)
    width = int((max(all_x) - min_x + 2 * padding) * scale)
    height = int((max(all_y) - min_y + 2 * padding) * scale)
    if width <= 0 or height <= 0:
        return False

    def to_pixel(x: float, y: float) -> tuple[float, float]:
        return (x - min_x + padding) * scale, (y - min_y + padding) * scale

    image = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(image)
    for xs, ys in trace_points.values():
        if len(xs) < 2:
            continue
        draw.line([to_pixel(x, y) for x, y in zip(xs, ys)], fill=0, width=2, joint="curve")

    labels = []
    for zone_type, trace_ids in _find_zones(root):
        class_id = CLASSES.get(zone_type)
        if class_id is None:
            continue

        zone_x = [x for tid in trace_ids for x in trace_points.get(tid, ([], []))[0]]
        zone_y = [y for tid in trace_ids for y in trace_points.get(tid, ([], []))[1]]
        if not zone_x or not zone_y:
            continue

        x1, y1 = to_pixel(min(zone_x), min(zone_y))
        x2, y2 = to_pixel(max(zone_x), max(zone_y))
        cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
        w, h = (x2 - x1) / width, (y2 - y1) / height
        labels.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    if not labels:
        return False

    doc_id = os.path.splitext(os.path.basename(inkml_path))[0]
    image.save(os.path.join(images_dir, f"{doc_id}.png"))
    with open(os.path.join(labels_dir, f"{doc_id}.txt"), "w") as f:
        f.write("\n".join(labels) + "\n")
    return True


def prepare_layout_data(
    raw_data_dir: str,
    output_dir: str,
    val_split: float = 0.15,
    scale: float = 1.5,
    padding: int = 20,
    seed: int = 42,
) -> None:
    """Convert every .inkml file in raw_data_dir into a train/val YOLO detection dataset."""
    inkml_paths = sorted(
        os.path.join(raw_data_dir, name) for name in os.listdir(raw_data_dir) if name.endswith(".inkml")
    )

    converted_ids = []
    for split in ("train", "val"):
        os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)

    # Convert into a staging area first since we don't know the train/val split until
    # we know which documents actually produced usable (image, label) pairs.
    staging_images = os.path.join(output_dir, "_staging", "images")
    staging_labels = os.path.join(output_dir, "_staging", "labels")
    os.makedirs(staging_images, exist_ok=True)
    os.makedirs(staging_labels, exist_ok=True)

    for path in inkml_paths:
        doc_id = os.path.splitext(os.path.basename(path))[0]
        if _convert_document(path, staging_images, staging_labels, scale, padding):
            converted_ids.append(doc_id)

    random.Random(seed).shuffle(converted_ids)
    num_val = max(1, int(len(converted_ids) * val_split)) if converted_ids else 0
    val_ids = set(converted_ids[:num_val])

    for doc_id in converted_ids:
        split = "val" if doc_id in val_ids else "train"
        for ext, subdir in ((".png", "images"), (".txt", "labels")):
            src = os.path.join(output_dir, "_staging", subdir, doc_id + ext)
            dst = os.path.join(output_dir, split, subdir, doc_id + ext)
            os.replace(src, dst)

    os.rmdir(staging_images)
    os.rmdir(staging_labels)
    os.rmdir(os.path.join(output_dir, "_staging"))

    with open(os.path.join(output_dir, "data.yaml"), "w") as f:
        f.write(
            f"path: {os.path.abspath(output_dir)}\n"
            "train: train/images\n"
            "val: val/images\n"
            "names:\n"
            "  0: text\n"
            "  1: math\n"
        )

    print(f"Converted {len(converted_ids)}/{len(inkml_paths)} documents "
          f"({len(converted_ids) - num_val} train / {num_val} val).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_data_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--scale", type=float, default=1.5)
    args = parser.parse_args()

    prepare_layout_data(args.raw_data_dir, args.output_dir, val_split=args.val_split, scale=args.scale)
