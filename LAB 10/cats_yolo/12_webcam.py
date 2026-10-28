import argparse

import cv2
from ultralytics import YOLO

from config import BEST_MODEL


def main():
    parser = argparse.ArgumentParser(description="ตรวจจับแมวจากกล้องแบบ Real-time")
    parser.add_argument("--camera", type=int, default=0, help="หมายเลขกล้อง (ค่าเริ่มต้น: 0)")
    args = parser.parse_args()

    if not BEST_MODEL.exists():
        raise FileNotFoundError(f"ไม่พบโมเดล {BEST_MODEL}; รัน 07_train.py ก่อน")

    model = YOLO(str(BEST_MODEL))
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"เปิดกล้องหมายเลข {args.camera} ไม่ได้")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("อ่านภาพจากกล้องไม่ได้")
            result = model.predict(frame, conf=0.25, verbose=False)
            cv2.imshow("YOLO Cat Detection - press q to quit", result[0].plot())
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
