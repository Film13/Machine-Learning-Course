from ultralytics import YOLO

from config import DATASET, EXPERIMENTS, MODEL_NAME, RUNS_ROOT


def main():
    data_yaml = DATASET / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"ไม่พบ {data_yaml}; รัน 05_prepare_dataset.py และตรวจ Label ก่อน"
        )

    for experiment in EXPERIMENTS:
        print(f"\nเริ่มการทดลอง: {experiment}")
        model = YOLO(MODEL_NAME)
        model.train(
            data=str(data_yaml.resolve()),
            epochs=experiment["epochs"],
            imgsz=experiment["imgsz"],
            batch=experiment["batch"],
            seed=42,
            deterministic=True,
            workers=0,
            project=str(RUNS_ROOT),
            name=experiment["name"],
            exist_ok=True,
            plots=True,
        )
    print("\nฝึกเสร็จแล้ว")


if __name__ == "__main__":
    main()
