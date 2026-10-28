import argparse
from pathlib import Path

from ultralytics import YOLO

from config import BEST_MODEL, PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description="ตรวจจับแมวในวิดีโอ")
    parser.add_argument("video", type=Path, help="พาธของวิดีโอที่ต้องการทดสอบ")
    args = parser.parse_args()

    if not BEST_MODEL.exists():
        raise FileNotFoundError(f"ไม่พบโมเดล {BEST_MODEL}; รัน 07_train.py ก่อน")
    if not args.video.is_file():
        raise FileNotFoundError(f"ไม่พบไฟล์วิดีโอ: {args.video}")

    YOLO(str(BEST_MODEL)).predict(
        source=str(args.video),
        conf=0.25,
        save=True,
        show=False,
        project=str(PROJECT_ROOT / "runs" / "detect"),
        name="video",
        exist_ok=True,
    )
    print(f"บันทึกผลไว้ที่ {PROJECT_ROOT / 'runs' / 'detect' / 'video'}")


if __name__ == "__main__":
    main()
