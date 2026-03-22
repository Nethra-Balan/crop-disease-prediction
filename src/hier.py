import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
import random

# Dataset path
DATASET_PATH = "../dataset/"
classes = ["healthy", "diseased"]

data = []
img_names = []

# Load 50 images per class
print("Loading images for Hierarchical Clustering...")
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

# Hierarchical clustering (Ward method)
linked = linkage(data, 'ward')

# Dendrogram
plt.figure(figsize=(10,5))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title("Hierarchical Clustering Dendrogram")
plt.show()

from scipy.cluster.hierarchy import fcluster

# Form 2 clusters (like K-Means)
cluster_labels = fcluster(linked, t=2, criterion='maxclust')

# Display 5 random images per cluster
for cluster in [1,2]:  # fcluster clusters start from 1
    idxs = np.where(cluster_labels == cluster)[0]
    sample_idxs = random.sample(list(idxs), min(5, len(idxs)))
    plt.figure(figsize=(12,3))
    for i, idx in enumerate(sample_idxs):
        img = cv2.imread(img_names[idx])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1,5,i+1)
        plt.imshow(img_rgb)
        plt.axis('off')
    plt.suptitle(f"Hierarchical Cluster {cluster} samples")
    plt.show()