from config import RAW_IMAGES, RAW_LABELS
from label_utils import validate_label


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

def main():
    if not RAW_IMAGES.exists():
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์ภาพ: {RAW_IMAGES}; รัน 02_download_images.py ก่อน")
    RAW_LABELS.mkdir(parents=True, exist_ok=True)
    images = sorted(
        path for path in RAW_IMAGES.iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    )
    labeled = [image for image in images if (RAW_LABELS / f"{image.stem}.txt").exists()]
    missing = [image for image in images if not (RAW_LABELS / f"{image.stem}.txt").exists()]
    invalid = {
        image.name: validate_label(RAW_LABELS / f"{image.stem}.txt")
        for image in labeled
        if validate_label(RAW_LABELS / f"{image.stem}.txt")
    }

    print(f"ภาพทั้งหมด: {len(images)}")
    print(f"มี Label: {len(labeled)} | ไม่มี Label: {len(missing)} | Label ผิดรูปแบบ: {len(invalid)}")
    print("ตัวอย่างภาพที่ยังไม่มี Label:", [image.name for image in missing[:10]])
    print("ตัวอย่างข้อผิดพลาด:", list(invalid.items())[:5])


if __name__ == "__main__":
    main()
