import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array

# Load model
model = tf.keras.models.load_model('fine_tuned.keras')
img_size = (224, 224)
class_labels = [chr(i) for i in range(65, 91)]  # A-Z

def preprocess_user_image(path):
    img = load_img(path, target_size=img_size)
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def classify_image(path):
    x = preprocess_user_image(path)
    preds = model.predict(x)
    pred_class = class_labels[np.argmax(preds)]
    return pred_class

def open_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        return
    pred = classify_image(file_path)

    img = Image.open(file_path).convert('RGB')
    img.thumbnail((300, 300))
    img_tk = ImageTk.PhotoImage(img)

    panel.config(image=img_tk)
    panel.image = img_tk
    label_var.set(f"Predicted letter: {pred}")

# GUI setup
root = tk.Tk()
root.title("Letter Classifier")

btn = tk.Button(root, text="Choose Image", command=open_file)
btn.pack(pady=10)

panel = tk.Label(root)
panel.pack()

label_var = tk.StringVar()
label = tk.Label(root, textvariable=label_var, font=("Arial", 16))
label.pack(pady=10)

root.mainloop()
