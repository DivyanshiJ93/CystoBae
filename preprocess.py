import os
import cv2
import numpy as np
import pickle

IMG_SIZE = 128

# Paths to train and test folders
TRAIN_DIR = os.path.join("clean_data", "train")
TEST_DIR = os.path.join("clean_data", "test")

def load_images_from_folder(folder_path, label_value):
    data = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append((img, label_value))
        except Exception as e:
            print(f"Failed to load {img_path}: {e}")
    return data

def load_data():
    data = []

    # Train set
    data += load_images_from_folder(os.path.join(TRAIN_DIR, "infected"), 1)
    data += load_images_from_folder(os.path.join(TRAIN_DIR, "notinfected"), 0)

    # Test set
    data += load_images_from_folder(os.path.join(TEST_DIR, "infected"), 1)
    data += load_images_from_folder(os.path.join(TEST_DIR, "notinfected"), 0)

    return data

# Load and preprocess
data = load_data()
print(f"✅ Total images loaded: {len(data)}")

# Shuffle and separate
np.random.shuffle(data)
X, y = zip(*data)
X = np.array(X) / 255.0
y = np.array(y)

# Grayscale to RGB for ML
X_rgb = np.stack([cv2.cvtColor((img * 255).astype("uint8"), cv2.COLOR_GRAY2RGB) for img in X])

# Add channel dim for DL
X_dl = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Split train-test (you already have train/test folders, so just for model eval we do this)
from sklearn.model_selection import train_test_split
X_train_rgb, X_test_rgb, y_train_rgb, y_test_rgb = train_test_split(X_rgb, y, test_size=0.2, random_state=42)
X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(X_dl, y, test_size=0.2, random_state=42)

# Save to disk
with open("preprocessed_ml1.pkl", "wb") as f:
    pickle.dump((X_train_rgb, X_test_rgb, y_train_rgb, y_test_rgb), f)

with open("preprocessed_dl1.pkl", "wb") as f:
    pickle.dump((X_train_dl, X_test_dl, y_train_dl, y_test_dl), f)

print("🎉 Preprocessing complete. Data saved as preprocessed_ml1.pkl and preprocessed_dl1.pkl.")

