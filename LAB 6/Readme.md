แหล่งที่มาของ Dataset : https://www.kaggle.com/code/root31415/breed-classification/notebook
คำอธิบาย
# LAB 6: Neural Network (NN)
1. load_data.py
ใช้สำหรับโหลด Dataset `cats(1).csv` และตรวจสอบข้อมูลเบื้องต้น เช่น จำนวนข้อมูล คอลัมน์ และข้อมูลที่ขาดหาย

2. preprocessing.py
ใช้สำหรับเตรียมข้อมูลก่อนสร้างโมเดล โดยเลือก Features และ Target แปลงข้อมูลเป็นตัวเลข แบ่งข้อมูลเป็น Training 80% และ Testing 20% และทำ Standardization

3. epoch_comparison.py
ใช้เปรียบเทียบประสิทธิภาพของ Neural Network เมื่อใช้จำนวน Epochs แตกต่างกัน ได้แก่ 10, 25, 50, 100 และ 200 Epochs โดยวัดผลด้วย Accuracy และสร้างกราฟเปรียบเทียบ

4. nn_configuration.py
ใช้เปรียบเทียบโครงสร้าง Neural Network ที่มีจำนวน Hidden Layers และ Neurons แตกต่างกัน เพื่อหาการตั้งค่าที่ให้ Accuracy สูงที่สุด

5. training_validation.py
ใช้แสดงผลการฝึก Neural Network โดยเปรียบเทียบ Training Accuracy, Validation Accuracy, Training Loss และ Validation Loss ในแต่ละ Epoch

6. prediction.py
ใช้ Neural Network ที่ฝึกแล้วในการทำนายข้อมูล และเปรียบเทียบผล Prediction กับค่าจริง พร้อมคำนวณ Accuracy

7. cats(1).csv
เป็น Dataset ที่ใช้ในการทดลอง โดยใช้ข้อมูลเกี่ยวกับแมวสำหรับนำมาใช้ฝึกและทดสอบ Neural Network

ผลลัพธ์
โปรแกรมจะสร้างไฟล์ผลลัพธ์ `.txt` และกราฟ `.png` เพื่อใช้เปรียบเทียบ Accuracy, Neural Network Configuration, Training/Validation และผลการ Prediction