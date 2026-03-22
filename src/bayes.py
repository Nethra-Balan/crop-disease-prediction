import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Dataset path
DATASET_PATH = "../dataset/"

# Class names
classes = ["healthy", "diseased"]

data = []
labels = []

print("Loading images...")

# Load ALL images
for label, category in enumerate(classes):
    folder_path = os.path.join(DATASET_PATH, category)
    
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            continue
        
        # Resize image
        img = cv2.resize(img, (200, 200))
        
        # Keep color (better visualization)
        data.append(img)
        labels.append(label)

print("Total images loaded:", len(data))

# Convert to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Flatten images for model
data_flat = data.reshape(len(data), -1)

# Train-test split
X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(
    data_flat, labels, data, test_size=0.3, random_state=42
)

# Train model
model = GaussianNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Results
print("\n===== NAIVE BAYES RESULTS =====")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Label mapping
label_map = {0: "Healthy", 1: "Diseased"}

# Show sample predictions
print("\nDisplaying sample predictions...")

for i in range(5):
    img = img_test[i]
    
    # Convert BGR to RGB for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.imshow(img_rgb)
    plt.title(f"Actual: {label_map[y_test[i]]} | Predicted: {label_map[y_pred[i]]}")
    plt.axis('off')
    plt.show()