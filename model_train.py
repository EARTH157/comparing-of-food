import pandas as pd
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

X = []
y = []

# กำหนด path ไปยังโฟลเดอร์ที่เก็บรูป
IMAGE_FOLDER = "Questionair Images"

# โหลดไฟล์ CSV
file_path = "data_from_questionaire.csv"
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"❌ ไม่พบไฟล์ CSV: {file_path}")
    exit(1)

IMG_SIZE = (224, 224)  # ขนาดภาพที่ใช้กับโมเดล

def load_and_preprocess_image(image_name):
    """ โหลดภาพจากโฟลเดอร์, Resize และ Normalize """
    img_path = os.path.join(IMAGE_FOLDER, image_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ ไม่พบรูปภาพ: {img_path}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # แปลงเป็น RGB
    img = cv2.resize(img, IMG_SIZE)  # Resize เป็นขนาดที่กำหนด
    img = img / 255.0  # Normalize ค่าให้อยู่ในช่วง [0,1]
    return img

for _, row in df.iterrows():
    img1 = load_and_preprocess_image(row["Image 1"])
    img2 = load_and_preprocess_image(row["Image 2"])
    
    if img1 is None or img2 is None:
        continue  # ข้ามข้อมูลที่โหลดไม่ได้

    X.append(img1)
    X.append(img2)

    # Winner = 1 หมายถึง Image 1 ชนะ, Winner = 2 หมายถึง Image 2 ชนะ
    if row["Winner"] == 1:
        y.append(0)  # Image 1 ดีกว่า
        y.append(1)  # Image 2 แย่กว่า
    else:
        y.append(1)  # Image 1 แย่กว่า
        y.append(0)  # Image 2 ดีกว่า

X = np.array(X)
y = np.array(y)

print(f"📌 ขนาดของ X: {X.shape}")
print(f"📌 ขนาดของ y: {y.shape}")

# สร้างโมเดล CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")  # Output เป็น 0 หรือ 1
])

# แสดงตัวอย่างข้อมูล
print(df.info(), df.head())

# ทดสอบโหลดภาพตัวอย่าง
sample_image = load_and_preprocess_image(df.iloc[1]["Image 2"])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# ฝึกโมเดล
model.fit(X, y, epochs=15, batch_size=32, validation_split=0.2)
model.save("food_comparison_model_v8.h5")
print("✅ โมเดลถูกบันทึกเรียบร้อย")

plt.imshow(sample_image)
plt.axis("off")
plt.show()