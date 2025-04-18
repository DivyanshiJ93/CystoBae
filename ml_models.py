import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model

# Load preprocessed data
with open("preprocessed_ml1.pkl", "rb") as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# Step 1: Feature Extraction using VGG16
def extract_features(X):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    model = Model(inputs=base_model.input, outputs=base_model.output)
    X_prep = preprocess_input(X * 255.0)
    features = model.predict(X_prep, verbose=1)
    return features.reshape(features.shape[0], -1)

print("🔄 Extracting features with VGG16...")
X_train_feat = extract_features(X_train)
X_test_feat = extract_features(X_test)
print(f"✅ Feature shape: {X_train_feat.shape}")

# Step 2: Train ML models
models = {
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# Step 3: Train and evaluate
for name, model in models.items():
    print(f"\n🔍 Training {name}...")
    model.fit(X_train_feat, y_train)
    y_pred = model.predict(X_test_feat)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy of {name}: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['NotInfected', 'Infected'], yticklabels=['NotInfected', 'Infected'])
    plt.title(f'{name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()
