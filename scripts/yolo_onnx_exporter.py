"""
Provides a utility to export YOLO models to ONNX format with
configurable settings via command-line arguments.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_model_path = (script_dir / "../models/yolov11n-det.pt").resolve()

    parser = argparse.ArgumentParser(
        description="Export a YOLO model to ONNX with configurable settings."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(default_model_path),
        help="Path to .pt model or model name (default: ../models/yolov11n-det.pt).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs="+",
        default=None,
        metavar=("H", "W"),
        help=(
            "Input image size. Use one value for square (e.g. --imgsz 640) "
            "or two values for height/width (e.g. --imgsz 640 960)."
        ),
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Batch size for export. Use with --dynamic for dynamic batch support.",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic ONNX axes (including dynamic batch dimension).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=None,
        help="ONNX opset version (e.g. 12, 13, 17).",
    )
    parser.add_argument(
        "--simplify",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable ONNX graph simplification.",
    )

    args = parser.parse_args()
    if args.imgsz is not None and len(args.imgsz) not in (1, 2):
        parser.error("--imgsz expects 1 value (square) or 2 values (H W).")
    return args


def main() -> None:
    args = parse_args()

    model = YOLO(args.model)

    export_kwargs = {
        "format": "onnx",
        "dynamic": args.dynamic,
    }

    if args.imgsz is not None:
        export_kwargs["imgsz"] = args.imgsz
    if args.batch is not None:
        export_kwargs["batch"] = args.batch
    if args.opset is not None:
        export_kwargs["opset"] = args.opset
    if args.simplify is not None:
        export_kwargs["simplify"] = args.simplify

    export_kwargs["device"] = 0

    print("Export settings:", export_kwargs)
    model.export(**export_kwargs)


if __name__ == "__main__":
    main()