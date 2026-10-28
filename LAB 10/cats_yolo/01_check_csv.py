import pandas as pd

from config import CSV_PATH
from csv_columns import COLUMNS


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ CSV: {CSV_PATH}")

    data = pd.read_csv(CSV_PATH, header=None, names=COLUMNS)
    print(f"ไฟล์: {CSV_PATH}")
    print(f"จำนวนแถว/คอลัมน์: {data.shape}")
    preview_columns = ["record_id", "animal_type", "age", "gender", "size", "breed"]
    print("\nตัวอย่างข้อมูล:")
    print(data[preview_columns].head().to_string(index=False))
    print("\nประเภทสัตว์:")
    print(data["animal_type"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
