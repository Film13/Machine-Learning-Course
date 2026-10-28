import shutil

from sklearn.model_selection import train_test_split

from config import DATASET, RAW_IMAGES, RAW_LABELS
from label_utils import validate_label


def main():
    if not RAW_IMAGES.exists():
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์ภาพ: {RAW_IMAGES}; รัน 02_download_images.py ก่อน")
    if DATASET.exists():
        raise FileExistsError(
            f"พบ Dataset เดิมที่ {DATASET}; สำรอง/เปลี่ยนชื่อโฟลเดอร์ก่อนสร้างใหม่"
        )

    images = sorted(
        path for path in RAW_IMAGES.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    valid_pairs = [
        (image, RAW_LABELS / f"{image.stem}.txt")
        for image in images
        if (RAW_LABELS / f"{image.stem}.txt").exists()
        and not validate_label(RAW_LABELS / f"{image.stem}.txt")
    ]
    if len(valid_pairs) < 10:
        raise ValueError(
            f"มีภาพที่ Label ถูกต้องเพียง {len(valid_pairs)} ภาพ; "
            "ทำ Label เพิ่มอย่างน้อย 10 ภาพก่อน"
        )

    train_pairs, temp_pairs = train_test_split(
        valid_pairs, test_size=0.30, random_state=42
    )
    val_pairs, test_pairs = train_test_split(
        temp_pairs, test_size=1 / 3, random_state=42
    )
    split_pairs = {"train": train_pairs, "val": val_pairs, "test": test_pairs}

    for split, pairs in split_pairs.items():
        image_dir = DATASET / "images" / split
        label_dir = DATASET / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for image, label in pairs:
            shutil.copy2(image, image_dir / image.name)
            shutil.copy2(label, label_dir / label.name)

    yaml_text = (
        f"path: {DATASET.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        "  0: cat\n"
    )
    (DATASET / "data.yaml").write_text(yaml_text, encoding="utf-8")
    print("จำนวนภาพในแต่ละชุด:", {name: len(pairs) for name, pairs in split_pairs.items()})
    print(f"สร้างไฟล์ตั้งค่าแล้ว: {DATASET / 'data.yaml'}")


if __name__ == "__main__":
    main()
