import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import random

# Dataset path
DATASET_PATH = "../dataset/"
classes = ["healthy", "diseased"]

data = []
img_names = []

# Load 50 images per class
print("Loading images for Agglomerative Clustering...")
for label, category in enumerate(classes):
    folder_path = os.path.join(DATASET_PATH, category)
    imgs_loaded = 0
    for img_name in os.listdir(folder_path):
        if imgs_loaded >= 50:
            break
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (100, 100))
        data.append(img.flatten())
        img_names.append(img_path)
        imgs_loaded += 1

data = np.array(data)
print("Total images loaded:", len(data))

# Agglomerative Clustering
agglo = AgglomerativeClustering(n_clusters=2, linkage='ward')
agg_labels = agglo.fit_predict(data)

# PCA 2D plot for visualization
pca = PCA(n_components=2)
data_2d = pca.fit_transform(data)

plt.figure(figsize=(6,5))
plt.scatter(data_2d[:,0], data_2d[:,1], c=agg_labels, cmap='coolwarm', s=50)
plt.title("Agglomerative Clustering (2D PCA)")
plt.show()

# Display 5 random images per cluster
for cluster in range(2):
    idxs = np.where(agg_labels == cluster)[0]
    sample_idxs = random.sample(list(idxs), min(5, len(idxs)))
    plt.figure(figsize=(12,3))
    for i, idx in enumerate(sample_idxs):
        img = cv2.imread(img_names[idx])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1,5,i+1)
        plt.imshow(img_rgb)
        plt.axis('off')
    plt.suptitle(f"Agglomerative Cluster {cluster} samples (random 5)")
    plt.show()