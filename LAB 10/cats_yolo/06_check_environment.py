import platform

import torch
import ultralytics

from config import MODEL_NAME


def main():
    print("Python:", platform.python_version())
    print("PyTorch:", torch.__version__)
    print("Ultralytics:", ultralytics.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print(f"Pretrained model: {MODEL_NAME} (จะดาวน์โหลดอัตโนมัติเมื่อฝึกครั้งแรก)")


if __name__ == "__main__":
    main()
