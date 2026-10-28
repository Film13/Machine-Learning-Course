import matplotlib.pyplot as plt
from PIL import Image

from config import RAW_IMAGES


def main():
    if not RAW_IMAGES.exists():
        raise FileNotFoundError(f"ไม่พบโฟลเดอร์ภาพ: {RAW_IMAGES}")

    images = sorted(
        path for path in RAW_IMAGES.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )[:9]
    if not images:
        raise FileNotFoundError(f"ไม่พบภาพใน {RAW_IMAGES}; รัน 02_download_images.py ก่อน")

    figure, axes = plt.subplots(3, 3, figsize=(10, 9))
    for axis, image_path in zip(axes.ravel(), images):
        with Image.open(image_path) as image:
            axis.imshow(image)
        axis.set_title(image_path.name, fontsize=8)
        axis.axis("off")
    for axis in axes.ravel()[len(images):]:
        axis.axis("off")
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
