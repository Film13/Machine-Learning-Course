import ast
from io import BytesIO

import pandas as pd
import requests
from PIL import Image, UnidentifiedImageError

from config import CSV_PATH, RAW_IMAGES
from csv_columns import COLUMNS


def parse_urls(value):
    if not isinstance(value, str):
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(parsed, list):
        return []
    return [url for url in parsed if isinstance(url, str) and url.startswith("http")]


def image_urls(row):
    urls = parse_urls(row["photo_urls"])
    if urls:
        return urls
    try:
        photos = ast.literal_eval(row["photos"]) if isinstance(row["photos"], str) else []
    except (ValueError, SyntaxError):
        return []
    if not isinstance(photos, list):
        return []
    return [
        photo.get("medium") or photo.get("large") or photo.get("full")
        for photo in photos
        if isinstance(photo, dict)
        and any(photo.get(key) for key in ("medium", "large", "full"))
        and str(photo.get("medium") or photo.get("large") or photo.get("full")).startswith("http")
    ]


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ CSV: {CSV_PATH}")
    RAW_IMAGES.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(CSV_PATH, header=None, names=COLUMNS)
    results = []

    with requests.Session() as session:
        session.headers.update({"User-Agent": "Mozilla/5.0 educational object-detection lab"})
        for index, row in data.iterrows():
            pet_id = str(row["pet_id"])
            urls = image_urls(row)
            output_path = RAW_IMAGES / f"cat_{pet_id}.jpg"
            status = "no_url"
            detail = ""

            if output_path.exists():
                status = "already_exists"
            elif urls:
                try:
                    response = session.get(urls[0], timeout=(5, 10))
                    response.raise_for_status()
                    with Image.open(BytesIO(response.content)) as image:
                        image.convert("RGB").save(output_path, quality=95)
                    status = "downloaded"
                except (requests.RequestException, UnidentifiedImageError, OSError) as error:
                    status = "failed"
                    detail = f"{type(error).__name__}: {error}"

            results.append({"pet_id": pet_id, "status": status, "detail": detail})
            if (index + 1) % 10 == 0 or index + 1 == len(data):
                print(f"ดำเนินการแล้ว {index + 1}/{len(data)} รายการ")

    log = pd.DataFrame(results)
    log_path = RAW_IMAGES / "download_log.csv"
    log.to_csv(log_path, index=False, encoding="utf-8-sig")
    print("\nสรุป:")
    print(log["status"].value_counts().to_string())
    print(f"จำนวนภาพ: {len(list(RAW_IMAGES.glob('*.jpg')))}")
    print(f"บันทึกรายงาน: {log_path}")
    failed = log[log["status"] == "failed"]
    if not failed.empty:
        print("\nรายการที่ดาวน์โหลดไม่สำเร็จ:")
        print(failed.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
