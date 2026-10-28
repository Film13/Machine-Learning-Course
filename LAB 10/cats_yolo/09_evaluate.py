import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO

from config import DATASET, EXPERIMENTS, RUNS_ROOT, TEST_RUNS_ROOT


def main():
    data_yaml = DATASET / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"ไม่พบ {data_yaml}; รัน 05_prepare_dataset.py ก่อน")

    metric_rows = []
    for experiment in EXPERIMENTS:
        weights = RUNS_ROOT / experiment["name"] / "weights" / "best.pt"
        if not weights.exists():
            print(f"ข้าม ไม่พบโมเดล: {weights}")
            continue

        model = YOLO(str(weights))
        metrics = model.val(
            data=str(data_yaml.resolve()),
            split="test",
            imgsz=experiment["imgsz"],
            batch=experiment["batch"],
            workers=0,
            plots=True,
            project=str(TEST_RUNS_ROOT),
            name=experiment["name"],
            exist_ok=True,
        )
        metric_rows.append(
            {
                "Experiment": experiment["name"],
                "Epochs": experiment["epochs"],
                "Image Size": experiment["imgsz"],
                "Batch": experiment["batch"],
                "Precision": float(metrics.box.mp),
                "Recall": float(metrics.box.mr),
                "mAP50": float(metrics.box.map50),
                "mAP50-95": float(metrics.box.map),
            }
        )

    if not metric_rows:
        raise FileNotFoundError("ไม่พบโมเดลที่ฝึกแล้ว; รัน 07_train.py ก่อน")

    results = pd.DataFrame(metric_rows)
    output_csv = RUNS_ROOT / "yolo_test_metrics.csv"
    results.to_csv(output_csv, index=False)
    print(results.round(4).to_string(index=False))
    print(f"บันทึกผลประเมิน: {output_csv}")

    axis = results.set_index("Experiment")[
        ["Precision", "Recall", "mAP50", "mAP50-95"]
    ].plot(kind="bar", figsize=(10, 5))
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("YOLO Test Metrics by Configuration")
    axis.grid(axis="y")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    chart_path = RUNS_ROOT / "yolo_test_metrics.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"บันทึกกราฟ: {chart_path}")


if __name__ == "__main__":
    main()
