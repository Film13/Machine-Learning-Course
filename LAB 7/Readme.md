แหล่งที่มาของ Dataset : https://www.kaggle.com/code/root31415/breed-classification/notebook

คำอธิบาย

LAB: CNN on a Dataset of Your Choice

โปรเจกต์นี้ใช้ CNN ในการจำแนกสายพันธุ์แมวจาก Dataset cats(1).csv โดยนำข้อมูลมาเตรียมและแบ่งเป็นข้อมูลสำหรับ Training และ Testing จากนั้นทำการ Standardize ข้อมูลและสร้างโมเดล CNN หลายรูปแบบเพื่อเปรียบเทียบประสิทธิภาพ

ภายในโปรเจกต์มีการทดลอง จำนวน Layer, Neurons และจำนวน Epoch ที่แตกต่างกัน พร้อมวัดผลด้วย Accuracy และแสดงกราฟ Training/Validation Accuracy และ Loss เพื่อดูประสิทธิภาพของโมเดล

สุดท้ายโมเดลจะนำข้อมูล Test มาทำ Prediction เพื่อเปรียบเทียบผลที่โมเดลทำนายกับค่าจริง และหาว่าโมเดลและจำนวน Epoch แบบใดให้ผลดีที่สุด