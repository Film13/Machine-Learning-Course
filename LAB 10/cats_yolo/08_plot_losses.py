import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import EXPERIMENTS, RUNS_ROOT


def main():
    saved = 0
    for experiment in EXPERIMENTS:
        csv_path = RUNS_ROOT / experiment["name"] / "results.csv"
        if not csv_path.exists():
            print(f"ไม่พบผลฝึก: {csv_path}; รัน 07_train.py ก่อน")
            continue

        history = pd.read_csv(csv_path)
        history.columns = [column.strip() for column in history.columns]
        loss_columns = [column for column in history.columns if "loss" in column.lower()]
        if not loss_columns:
            print(f"ไม่พบคอลัมน์ loss ใน {csv_path}")
            continue

        figure, axis = plt.subplots(figsize=(10, 5))
        for column in loss_columns:
            axis.plot(history.index + 1, history[column], label=column)
        axis.set_title(f"Training / Validation Loss: {experiment['name']}")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.legend()
        axis.grid(True)
        figure.tight_layout()
        output_path = csv_path.parent / "loss_curve.png"
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
        print(f"บันทึกกราฟ: {output_path}")
        saved += 1

    if not saved:
        raise FileNotFoundError("ยังไม่มีกราฟ Loss ให้บันทึก; ตรวจว่าฝึกโมเดลสำเร็จแล้ว")


if __name__ == "__main__":
    main()
