import os
import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


# =========================================================
# 1. ตั้งค่าให้ Terminal แสดงภาษาไทยได้
# =========================================================

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# =========================================================
# 2. หาโฟลเดอร์ที่ไฟล์ preprocessing.py อยู่
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ตำแหน่งไฟล์ cats(1).csv
CSV_PATH = os.path.join(BASE_DIR, "cats(1).csv")


# =========================================================
# 3. ตรวจสอบว่าไฟล์ CSV มีอยู่จริงหรือไม่
# =========================================================

print("========================================")
print("        NEURAL NETWORK LAB 6")
print("========================================")

print("\nCSV file location:")
print(CSV_PATH)

if not os.path.exists(CSV_PATH):
    print("\nERROR: cats(1).csv was not found.")
    print("Please check that cats(1).csv is in the LAB 6 folder.")
    sys.exit()


# =========================================================
# 4. โหลด Dataset
# =========================================================

df = pd.read_csv(
    CSV_PATH,
    header=None
)

print("\nDataset loaded successfully!")


# =========================================================
# 5. แสดงข้อมูล Dataset
# =========================================================

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nNumber of rows:", df.shape[0])
print("Number of columns:", df.shape[1])


# =========================================================
# 6. ตั้งชื่อ Column
# =========================================================

df.columns = [
    "ID",
    "PetID",
    "URL",
    "Animal",
    "Age",
    "Sex",
    "Size",
    "Coat",
    "Breed",
    "Images",
    "ImageURLs"
]

print("\nColumn names:")
print(df.columns.tolist())


# =========================================================
# 7. ตรวจสอบข้อมูลที่หายไป
# =========================================================

print("\nMissing values:")
print(df.isnull().sum())


# =========================================================
# 8. แสดงข้อมูล Target
# =========================================================

print("\nSex distribution:")
print(df["Sex"].value_counts())


# =========================================================
# 9. เลือก Features
# =========================================================

features = [
    "Age",
    "Size",
    "Coat",
    "Breed"
]

target = "Sex"

print("\nSelected features:")
print(features)

print("\nTarget:")
print(target)


# =========================================================
# 10. เตรียมข้อมูล X และ y
# =========================================================

X = df[features].copy()
y = df[target].copy()


# =========================================================
# 11. แปลงข้อมูลประเภทข้อความเป็นตัวเลข
# =========================================================

X = pd.get_dummies(
    X,
    columns=features
)

print("\nFeatures converted to numerical values.")

print("Number of input features:", X.shape[1])


# =========================================================
# 12. Encode Target
# =========================================================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)

print("\nTarget classes:")

for i, class_name in enumerate(label_encoder.classes_):
    print(i, "=", class_name)


# =========================================================
# 13. แบ่ง Training และ Testing
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTrain/Test Split:")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================================================
# 14. Standardization
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


print("\nStandardization completed.")


# =========================================================
# 15. แสดงผลสุดท้าย
# =========================================================

print("\n========================================")
print("        PREPROCESSING COMPLETED")
print("========================================")

print("\nTraining data shape:")
print(X_train_scaled.shape)

print("\nTesting data shape:")
print(X_test_scaled.shape)

print("\nTarget classes:")
print(label_encoder.classes_)

print("\nReady for Neural Network training!")

print("\n========================================")