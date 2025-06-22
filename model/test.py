import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Parameters
test_dir = 'test'
img_size = (224, 224)
batch_size = 32
num_classes = 26

# Load model
model = tf.keras.models.load_model('fine_tuned.keras')

# Prepare test data generator (no shuffle)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Predict
pred_probs = model.predict(test_generator)
pred_labels = np.argmax(pred_probs, axis=1)
true_labels = test_generator.classes
class_labels = list(test_generator.class_indices.keys())
file_paths = test_generator.filepaths

# Metrics
print("Classification Report:\n")
print(classification_report(true_labels, pred_labels, target_names=class_labels))
print("Confusion Matrix:\n")
print(confusion_matrix(true_labels, pred_labels))

# Find misclassified indices
misclassified_idx = np.where(pred_labels != true_labels)[0]
print(f"\nTotal misclassified: {len(misclassified_idx)}\n")

def show_misclassified_images(indices):
    for idx in indices:
        img_path = file_paths[idx]
        true_class = class_labels[true_labels[idx]]
        pred_class = class_labels[pred_labels[idx]]

        print(f"File: {img_path}")
        print(f"True class: {true_class} | Predicted class: {pred_class}")
        
        # Load and show image
        img = load_img(img_path, target_size=img_size)
        plt.imshow(img)
        plt.title(f"True: {true_class}  Predicted: {pred_class}")
        plt.axis('off')
        plt.show()

        input("Press Enter for next...")

if len(misclassified_idx) > 0:
    show_misclassified_images(misclassified_idx)
else:
    print("No misclassified images found.")
