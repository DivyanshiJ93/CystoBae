import os
import cv2
import numpy as np
import pickle
import random

IMG_SIZE = 100  # Resize to 100x100

def augment_image(img):
    # Random flip
    if random.random() > 0.5:
        img = cv2.flip(img, 1)  # horizontal
    if random.random() > 0.5:
        img = cv2.flip(img, 0)  # vertical
    
    # Random rotation
    angle = random.choice([0, 90, 180, 270])
    M = cv2.getRotationMatrix2D((IMG_SIZE // 2, IMG_SIZE // 2), angle, 1)
    img = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE))
    
    return img

def add_noise(img):
    noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
    return cv2.add(img, noise)

def load_images(folder_path, is_test=False):
    data = []
    labels = []
    categories = ['infected', 'notinfected']

    for label, category in enumerate(categories):
        folder = os.path.join(folder_path, category)
        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"⚠️ Failed to load {path}")
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            if is_test:
                img = add_noise(img)  # Add noise to test images only
            else:
                img = augment_image(img)  # Augment train images

            data.append(img.flatten())  # Flatten image
            labels.append(label)
    
    return np.array(data), np.array(labels)

print("📦 Loading and preprocessing training data...")
X_train, y_train = load_images("clean_data/train", is_test=False)

print("🧪 Loading and preprocessing testing data...")
X_test, y_test = load_images("clean_data/test", is_test=True)

print(f"✅ Loaded {len(X_train)} training and {len(X_test)} test samples")

# Save for later
with open("preprocessed_ml2.pkl", "wb") as f:
    pickle.dump((X_train, X_test, y_train, y_test), f)

print("💾 Data saved as preprocessed_ml2.pkl")
