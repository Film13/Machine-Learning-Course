import argparse
from pathlib import Path

from ultralytics import YOLO

from config import BEST_MODEL, PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description="ตรวจจับแมวในภาพ")
    parser.add_argument("image", type=Path, help="พาธของภาพที่ต้องการทดสอบ")
    args = parser.parse_args()

    if not BEST_MODEL.exists():
        raise FileNotFoundError(f"ไม่พบโมเดล {BEST_MODEL}; รัน 07_train.py ก่อน")
    if not args.image.is_file():
        raise FileNotFoundError(f"ไม่พบไฟล์ภาพ: {args.image}")

    YOLO(str(BEST_MODEL)).predict(
        source=str(args.image),
        conf=0.25,
        save=True,
        project=str(PROJECT_ROOT / "runs" / "detect"),
        name="image",
        exist_ok=True,
    )
    print(f"บันทึกผลไว้ที่ {PROJECT_ROOT / 'runs' / 'detect' / 'image'}")


if __name__ == "__main__":
    main()
