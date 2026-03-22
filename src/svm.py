import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
DATASET_PATH = "../dataset/"
classes = ["healthy", "diseased"]
data = []
labels = []
print("Loading images...")
for label, category in enumerate(classes):
    folder_path = os.path.join(DATASET_PATH, category)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (200, 200))
        data.append(img)
        labels.append(label)
print("Total images loaded:", len(data))
data = np.array(data)
labels = np.array(labels)
data_flat = data.reshape(len(data), -1)
X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(
    data_flat, labels, data, test_size=0.3, random_state=42
)
model = SVC(kernel='linear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\n===== SVM RESULTS =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
label_map = {0: "Healthy", 1: "Diseased"}
print("\nDisplaying sample predictions...")
for i in range(5):
    img = img_test[i]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6,6))
    plt.imshow(img_rgb)
    plt.title(f"Actual: {label_map[y_test[i]]} | Predicted: {label_map[y_pred[i]]}")
    plt.axis('off')
    plt.show()