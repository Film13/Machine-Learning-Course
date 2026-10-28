# YOLO ตรวจจับแมว (สคริปต์ Python)

โปรเจกต์นี้แยกขั้นตอนออกเป็นไฟล์ `.py` ไม่ต้องใช้ Jupyter Notebook

## 1. ติดตั้ง

เปิด Terminal ที่โฟลเดอร์โปรเจกต์หลัก สร้าง Virtual Environment แยกเพื่อไม่ให้แพ็กเกจเดิมในเครื่องชนกัน แล้วติดตั้งแพ็กเกจ:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r cats_yolo\requirements.txt
```

วางไฟล์ `cats(1).csv` ไว้ในโฟลเดอร์หลักเดียวกับโฟลเดอร์ `cats_yolo` (ไฟล์นี้มีอยู่แล้ว) จากนั้นตรวจ CSV:

```powershell
python cats_yolo\01_check_csv.py
```

## 2. ดาวน์โหลดและตรวจภาพ

```powershell
python cats_yolo\02_download_images.py
python cats_yolo\03_preview_images.py
```

ภาพจะอยู่ใน `raw_images` และรายงานการดาวน์โหลดอยู่ใน `raw_images\download_log.csv`

## 3. วาด Label ให้ภาพ

ไฟล์ CSV มี URL ภาพแต่ไม่มีพิกัดกรอบวัตถุ จึงต้องวาดกรอบแมวเองก่อนฝึก:

1. ติดตั้งและเปิด LabelImg ด้วย `python -m pip install labelImg` และ `labelImg`
2. เลือก **Open Dir** เป็นโฟลเดอร์ `raw_images`
3. เลือก **Change Save Dir** เป็นโฟลเดอร์ `raw_labels`
4. เลือกรูปแบบ **YOLO** แล้ววาดกรอบแมวโดยใช้คลาส `cat`
5. ตรวจ Label และเตรียมชุดข้อมูล:

```powershell
python cats_yolo\04_check_labels.py
python cats_yolo\05_prepare_dataset.py
```

ต้องมีภาพที่ติด Label ถูกต้องอย่างน้อย 10 ภาพ ชุดข้อมูลจะแบ่งเป็น Train / Validation / Test
สคริปต์จะไม่เขียนทับ `cats_dataset` ที่มีอยู่แล้ว; เปลี่ยนชื่อหรือย้ายชุดเดิมก่อนแบ่งใหม่

## 4. ฝึกและประเมินโมเดล

```powershell
python cats_yolo\06_check_environment.py
python cats_yolo\07_train.py
python cats_yolo\08_plot_losses.py
python cats_yolo\09_evaluate.py
```

การฝึกเริ่มด้วยการทดลองเร็วหนึ่งรอบ (`5 epochs`, `416 px`, `batch 4`) ปรับชุดทดลองได้ที่ `cats_yolo\config.py`
ผลฝึกและกราฟจะบันทึกใต้ `runs\cats_lab`; ผลทดสอบอยู่ใต้ `runs\cats_test`

## 5. ทดสอบภาพ วิดีโอ หรือกล้อง

```powershell
python cats_yolo\10_predict_image.py "พาธ\ไป\ภาพ.jpg"
python cats_yolo\11_predict_video.py "พาธ\ไป\วิดีโอ.mp4"
python cats_yolo\12_webcam.py
```

ถ้าต้องเลือกกล้องอื่น ใช้ `python cats_yolo\12_webcam.py --camera 1` กด `q` เพื่อออกจากกล้อง
ผลภาพและวิดีโอจะบันทึกใน `runs\detect`
