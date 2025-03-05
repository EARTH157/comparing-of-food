import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import csv
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import re

# ตั้งค่าพื้นฐาน
IMAGE_FOLDER = "Questionair Images"
IMG_SIZE = (224, 224)
model = load_model("food_comparison_model_v8.h5")
CSV_FILE = "test.csv"

# ฟังก์ชันดึงตัวเลขจากชื่อไฟล์เพื่อเรียงลำดับให้ถูกต้อง
def extract_number(filename):
    # ค้นหาหมายเลขในชื่อไฟล์
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else float('inf')

# โหลดและเรียงลำดับไฟล์ภาพ
image_files = sorted(
    [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.jpg', '.png'))],
    key=extract_number
)

# ฟังก์ชันโหลดและแปลงภาพ
def load_and_preprocess_image(image_name):
    img_path = os.path.join(IMAGE_FOLDER, image_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ ไม่พบรูปภาพ: {img_path}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    return img

# ฟังก์ชันทำนายผล
def predict_winner(image1, image2):
    img1 = load_and_preprocess_image(image1)
    img2 = load_and_preprocess_image(image2)
    
    if img1 is None or img2 is None:
        return None, None, None

    pred1 = model.predict(np.expand_dims(img1, axis=0))[0][0]
    pred2 = model.predict(np.expand_dims(img2, axis=0))[0][0]
    winner = 1 if pred1 > pred2 else 2

    return winner, pred1, pred2, img1, img2

# ฟังก์ชันบันทึกผลลง CSV (บันทึกแค่ชื่อภาพและผู้ชนะ)
def save_result_to_csv(image1, image2, winner):
    with open(CSV_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([image1, image2, winner])

# ฟังก์ชันการประมวลผลภาพและบันทึกผลลง CSV
def process_images_and_save_results():
    for i in range(0, len(image_files) - 1, 2):
        image1 = image_files[i]
        image2 = image_files[i + 1]
        winner, _, _, _, _ = predict_winner(image1, image2)

        if winner is not None:
            save_result_to_csv(image1, image2, winner)  # บันทึกผลลง CSV

# 📌 **ดำเนินการเปรียบเทียบภาพทั้งหมดก่อนแสดง GUI**
process_images_and_save_results()

# 📌 **เริ่ม GUI หลังจากบันทึกผลแล้ว**
def update_images():
    global current_index
    if current_index >= len(image_files) - 1:
        return

    image1_name = image_files[current_index]
    image2_name = image_files[current_index + 1]

    winner, score1, score2, img1, img2 = predict_winner(image1_name, image2_name)

    if img1 is None or img2 is None:
        return

    img1_tk = ImageTk.PhotoImage(Image.fromarray((img1 * 255).astype(np.uint8)))
    img2_tk = ImageTk.PhotoImage(Image.fromarray((img2 * 255).astype(np.uint8)))

    label_img1.config(image=img1_tk)
    label_img1.image = img1_tk
    label_img2.config(image=img2_tk)
    label_img2.image = img2_tk
    label_image1_name.config(text=f"Image 1: {image1_name}")
    label_image2_name.config(text=f"Image 2: {image2_name}")
    label_winner.config(text=f"🏆 The Best Picture is: {winner}")

    # แสดงคะแนน
    label_score1.config(text=f"Score: {score1:.4f}")
    label_score2.config(text=f"Score: {score2:.4f}")

# ฟังก์ชันเลื่อนภาพ
def next_image():
    global current_index
    if current_index < len(image_files) - 2:
        current_index += 2
        update_images()

def prev_image():
    global current_index
    if current_index > 0:
        current_index -= 2
        update_images()

# ฟังก์ชันหลักของ GUI
def __main__():
    global label_img1, label_img2, label_image1_name, label_image2_name, label_winner, label_score1, label_score2, current_index
    current_index = 0

    root = tk.Tk()
    root.title("Food Image Comparison")
    root.geometry("900x600")

    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=1)
    root.rowconfigure(0, weight=4)
    root.rowconfigure(1, weight=1)
    root.rowconfigure(2, weight=1)
    root.rowconfigure(3, weight=1)
    root.rowconfigure(4, weight=1)

    label_img1 = Label(root)
    label_img1.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

    label_img2 = Label(root)
    label_img2.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

    label_image1_name = Label(root, text="Image 1: ", font=("Arial", 10))
    label_image1_name.grid(row=4, column=0, pady=10, sticky="nsew")

    label_image2_name = Label(root, text="Image 2: ", font=("Arial", 10))
    label_image2_name.grid(row=4, column=1, pady=10, sticky="nsew")

    label_score1 = Label(root, text="Score: ", font=("Arial", 12))
    label_score1.grid(row=1, column=0, sticky="nsew")

    label_score2 = Label(root, text="Score: ", font=("Arial", 12))
    label_score2.grid(row=1, column=1, sticky="nsew")

    label_winner = Label(root, text="The Best Picture is: ", font=("Arial", 14, "bold"), fg="red")
    label_winner.grid(row=2, column=0, columnspan=2, pady=10, sticky="nsew")

    button_prev = Button(root, text="⬅ Previous", font=("Arial", 12), command=prev_image)
    button_prev.grid(row=3, column=0, pady=10, sticky="nsew")

    button_next = Button(root, text="Next ➡", font=("Arial", 12), command=next_image)
    button_next.grid(row=3, column=1, pady=10, sticky="nsew")

    update_images()
    root.mainloop()

if __name__ == "__main__":
    __main__()