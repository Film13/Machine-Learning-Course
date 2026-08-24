แหล่งที่มาของ Dataset : https://www.kaggle.com/code/root31415/breed-classification/notebook
คำอธิบาย

1. LAB: SVM on Cats Dataset
LAB นี้เป็นการนำเทคนิค Machine Learning แบบ Support Vector Machine (SVM) มาใช้กับชุดข้อมูลแมว เพื่อจำแนกข้อมูลและเปรียบเทียบประสิทธิภาพของ SVM แต่ละรูปแบบ โดยใช้ Kernel ได้แก่ Linear, Polynomial และ RBF

2. Explore the Dataset
เริ่มต้นด้วยการโหลดและสำรวจข้อมูล cats(1).csv เพื่อศึกษาจำนวนข้อมูล ค่าที่หายไป และการกระจายของข้อมูล เช่น Age, Gender, Size และ Breed เพื่อทำความเข้าใจลักษณะของ Dataset ก่อนนำไปวิเคราะห์

3. Select Features and Preprocess the Data
เลือกข้อมูล Age, Size และ Breed เป็น Input Features ส่วน Gender เป็น Target ที่ต้องการทำนาย จากนั้นทำการเตรียมข้อมูลและลบข้อมูลที่ขาดหาย เพื่อให้ข้อมูลมีความพร้อมสำหรับการสร้างโมเดล

4. Standardization and Data Preparation
ข้อมูลประเภทข้อความจะถูกแปลงเป็นตัวเลขด้วย One-Hot Encoding และทำ Standardization เพื่อปรับข้อมูลให้อยู่ในรูปแบบที่เหมาะสมสำหรับการนำไปใช้กับโมเดล SVM

5. Clustering
นำข้อมูลมาทดลองจัดกลุ่มด้วย K-Means Clustering โดยแบ่งข้อมูลออกเป็น 2 กลุ่ม และใช้ PCA เพื่อลดมิติของข้อมูลให้สามารถแสดงผลการจัดกลุ่มในรูปแบบกราฟได้ง่ายขึ้น

6. Split Data for Classification
ข้อมูลถูกแบ่งออกเป็น 2 ส่วน คือ Training Data 75% สำหรับใช้ฝึกโมเดล และ Testing Data 25% สำหรับใช้ทดสอบประสิทธิภาพของโมเดลกับข้อมูลที่ไม่เคยเห็นมาก่อน

7. Train SVM Models with Different Kernels
สร้างและฝึกโมเดล SVM โดยใช้ Kernel ทั้ง 3 รูปแบบ ได้แก่ Linear, Polynomial และ RBF เพื่อเปรียบเทียบว่าแต่ละรูปแบบสามารถจำแนกข้อมูล Gender ของแมวได้ดีเพียงใด

8. Compare Accuracy
หลังจากฝึกโมเดลแล้ว จะนำผลการทำนายมาเปรียบเทียบกับข้อมูลจริง และคำนวณค่า Accuracy ของแต่ละ Kernel เพื่อดูว่าโมเดลใดมีประสิทธิภาพในการจำแนกข้อมูลมากที่สุด

9. Predictions
สุดท้าย นำโมเดลที่ผ่านการฝึกมาใช้ทำนายข้อมูลใหม่จาก Age, Size และ Breed เพื่อให้โมเดลคาดการณ์ Gender ของแมว และสรุปผลการทำงานของ SVM ใน Dataset นี้ได้