import os
import shutil
import random
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
import numpy as np

random.seed(42)

data_dir = 'data'
output_train_dir = 'train'
output_test_dir = 'test'
img_size = (224, 224)
test_ratio = 0.1

# Create output dirs
for letter in os.listdir(data_dir):
    if os.path.isdir(os.path.join(data_dir, letter)):
        os.makedirs(os.path.join(output_train_dir, letter), exist_ok=True)
        os.makedirs(os.path.join(output_test_dir, letter), exist_ok=True)

def preprocess_and_save(input_path, output_path):
    img = load_img(input_path, target_size=img_size)
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)  # BGR, zero-center
    # Convert back to uint8 for saving (approximate)
    img_array = (img_array + [103.939, 116.779, 123.68])  # undo zero-centering
    img_array = img_array[..., ::-1]  # BGR to RGB
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    save_img(output_path, img_array)

for letter in os.listdir(data_dir):
    letter_dir = os.path.join(data_dir, letter)
    if not os.path.isdir(letter_dir):
        continue

    images = [f for f in os.listdir(letter_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)
    split_index = int(len(images) * (1 - test_ratio))

    train_imgs = images[:split_index]
    test_imgs = images[split_index:]

    for fname in train_imgs:
        src = os.path.join(letter_dir, fname)
        dst = os.path.join(output_train_dir, letter, fname)
        preprocess_and_save(src, dst)

    for fname in test_imgs:
        src = os.path.join(letter_dir, fname)
        dst = os.path.join(output_test_dir, letter, fname)
        preprocess_and_save(src, dst)

print("Preprocessing done. Processed images saved in 'train/' and 'test/' folders.")
