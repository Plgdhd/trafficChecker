import tkinter as tk
from tkinter import messagebox
import tensorflow as tf
import pickle
import numpy as np
#загружаем модель
model = tf.keras.models.load_model("http_model.h5")
with open("http_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

def classify_request():
    raw_data = input_text.get("1.0", tk.END).strip()
    if not raw_data:
        messagebox.showwarning("Херня какая то", "Введите HTTP-запрос")
        return
    
    x = tokenizer.texts_to_matrix([raw_data], mode='binary')
    prediction = model.predict(x)[0]
    label = np.argmax(prediction)
    prob = prediction[label]

    if label == 1:
        result = f"✅ Нормальный запрос ({prob:.2f})"
    else:
        result = f"❌ Вредоносный запрос ({prob:.2f})"
    
    result_label.config(text=result)

root = tk.Tk()
root.title("HTTP-checker")
tk.Label(root, text="Введите свой запрос:").pack()
input_text = tk.Text(root, height=10, width=60)
input_text.pack()
tk.Button(root, text="Проверить", command=classify_request).pack(pady=5)
result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.pack(pady=5)
root.mainloop()