import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

DATASET_PATH = "../dataset/"
classes = ["healthy", "diseased"]

data = []
img_names = []
print("Loading images for K-Means...")
for label, category in enumerate(classes):
    folder_path = os.path.join(DATASET_PATH, category)
    for img_name in os.listdir(folder_path)[:50]:  # sample 50 per class
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (100, 100))
        data.append(img.flatten())
        img_names.append(img_path)

data = np.array(data)
print("Total images loaded:", len(data))

# K-Means
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(data)

# PCA for 2D visualization
pca = PCA(n_components=2)
data_2d = pca.fit_transform(data)

plt.figure(figsize=(6,5))
plt.scatter(data_2d[:,0], data_2d[:,1], c=labels, cmap='viridis', s=50)
plt.title("K-Means Clustering (2D PCA)")
plt.show()

import random

for cluster in range(2):
    idxs = np.where(labels == cluster)[0]
    sample_idxs = random.sample(list(idxs), min(5, len(idxs)))  # pick 5 randomly
    plt.figure(figsize=(12,3))  # smaller figure for 5 images
    for i, idx in enumerate(sample_idxs):
        img = cv2.imread(img_names[idx])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1,5,i+1)  # 1 row x 5 cols
        plt.imshow(img_rgb)
        plt.axis('off')
    plt.suptitle(f"K-Means Cluster {cluster} samples (random 5)")
    plt.show()