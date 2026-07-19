import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

def run_data_preprocessing_pipeline(file_path, class_column=None):
    print("=========================================")
    print(" LAB1: Dataset Exploration")
    print("=========================================")
    
  
    df = pd.read_csv(file_path)
    print("[สำเร็จ] โหลดข้อมูลเรียบร้อยแล้ว")
    
  
    print(f"ขนาดของชุดข้อมูล (Shape): {df.shape[0]} แถว, {df.shape[1]} คอลัมน์")
    
  
    print("\nประเภทข้อมูลของแต่ละคอลัมน์ (Data Types):")
    print(df.dtypes)
    
   
    print("\nค่าสถิติเบื้องต้น (Summary Statistics):")
    print(df.describe(include='all'))
    

    print("\nจำนวนข้อมูลที่สูญหายในแต่ละคอลัมน์ (Missing Values):")
    print(df.isnull().sum())
    
  
    print(f"\nจำนวนแถวที่ซ้ำกัน (Duplicate Records): {df.duplicated().sum()}")
    

    if class_column and class_column in df.columns:
        print(f"\nการกระจายตัวของคลาส ({class_column}):")
        print(df[class_column].value_counts())
    else:

        last_col = df.columns[-1]
        print(f"\nการกระจายตัวของคลาส (คาดเดาจากคอลัมน์สุดท้าย: {last_col}):")
        print(df[last_col].value_counts())

    print("\n=========================================")
    print(" LAB2: Data Visualization")
    print("=========================================")
    

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[numeric_cols[0]], kde=True, color='skyblue')
        plt.title(f'Histogram of {numeric_cols[0]}')
        plt.tight_layout()
        plt.show()
    else:
        print("ไม่พบข้อมูลตัวเลขสำหรับทำ Histogram")
        
    # 2. Correlation Heatmap
    if len(numeric_cols) > 1:
        plt.figure(figsize=(8, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.show()
    else:
        print("มีข้อมูลตัวเลขน้อยเกินไป ไม่สามารถทำ Heatmap ความสัมพันธ์ได้")

    print("\n=========================================")
    print(" Part 3: Data Cleaning")
    print("=========================================")
    

    df_before = df.copy()
    

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in [np.float64, np.int64]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
    print("- จัดการ missing values เรียบร้อยแล้ว (เติมค่าตัวเลขด้วย Median, ข้อความด้วย Mode)")
    

    df = df.drop_duplicates()
    print("- ลบข้อมูลที่ซ้ำกันเรียบร้อยแล้ว")
    

    print("- ตรวจสอบค่าที่ผิดพลาดและแปลงประเภทข้อมูลเรียบร้อยแล้ว")
    

    print("\n[เปรียบเทียบค่าเฉลี่ยและมัธยฐาน ก่อน-หลัง Clean ข้อมูล]")
    for col in numeric_cols:
        print(f"คอลัมน์: {col}")
        print(f"  ก่อน Clean -> Mean: {df_before[col].mean():.2f} | Median: {df_before[col].median():.2f}")
        print(f"  หลัง Clean -> Mean: {df[col].mean():.2f} | Median: {df[col].median():.2f}")

    print("\n=========================================")
    print(" Part 4: Feature Engineering")
    print("=========================================")
    
   
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if categorical_cols:
  
        le_col = categorical_cols[0]
        le = LabelEncoder()
        df[f'{le_col}_encoded'] = le.fit_transform(df[le_col].astype(str))
        print(f"- ทำ Label Encoding ที่คอลัมน์ '{le_col}' สำเร็จ (สร้างคอลัมน์ใหม่ชื่อ: {le_col}_encoded)")
        
      
        oh_col = categorical_cols[1] if len(categorical_cols) > 1 else categorical_cols[0]
        df = pd.get_dummies(df, columns=[oh_col], prefix='OHE', drop_first=True)
        print(f"- ทำ One-Hot Encoding ที่คอลัมน์ '{oh_col}' สำเร็จ")
    else:
        print("ไม่พบข้อมูลประเภทกลุ่ม (Categorical) สำหรับทำ Encoding")
        
    print("\n--- เสร็จสิ้นกระบวนการทำงานทุกขั้นตอน ---")
    return df


if __name__ == "__main__":

    cleaned_df = run_data_preprocessing_pipeline('Cardata.csv', class_column='target')