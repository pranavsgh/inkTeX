"""Trains a YOLOv8 layout detector on the prepared text/math zone dataset. (Pranav)"""

from ultralytics import YOLO


def train_layout(
    data_yaml: str,
    output_dir: str,
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    base_weights: str = "yolov8n.pt",
) -> str:
    """Fine-tune a YOLOv8 detector on the given data.yaml, returning the path to best.pt."""
    model = YOLO(base_weights)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=output_dir,
        name="layout",
        exist_ok=True,
    )
    return str(results.save_dir / "weights" / "best.pt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_yaml")
    parser.add_argument("output_dir")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    best_weights = train_layout(
        args.data_yaml, args.output_dir, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch
    )
    print(f"Best weights: {best_weights}")
