import sys
from pathlib import Path


for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "cats(1).csv"
RAW_IMAGES = PROJECT_ROOT / "raw_images"
RAW_LABELS = PROJECT_ROOT / "raw_labels"
DATASET = PROJECT_ROOT / "cats_dataset"
RUNS_ROOT = PROJECT_ROOT / "runs" / "cats_lab"
TEST_RUNS_ROOT = PROJECT_ROOT / "runs" / "cats_test"
MODEL_NAME = "yolov8n.pt"

EXPERIMENTS = [
    {"name": "exp_quick", "epochs": 5, "imgsz": 416, "batch": 4},
]

BEST_MODEL = RUNS_ROOT / EXPERIMENTS[0]["name"] / "weights" / "best.pt"
