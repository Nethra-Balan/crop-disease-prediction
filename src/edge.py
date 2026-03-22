import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops  # ✅ updated spelling

# -------------------------------
# Load sample image (grayscale)
img_path = r"E:\college works\sem6\IP\proj\crop-disease-prediction\dataset\healthy\da93ba82-4070-4373-b818-a6fea8c91351___RS_HL 4724.JPG"
image = cv2.imread(img_path, 0)
image = cv2.resize(image, (256,256))

print("Original Image matrix (first 5x5):\n", image[:5,:5])

# -------------------------------
# 1️⃣ Edge Detection using Canny
edges = cv2.Canny(image, 100, 200)
print("\nEdge matrix (first 5x5):\n", edges[:5,:5])

# -------------------------------
# 2️⃣ Texture Analysis using GLCM (Gray Level Co-occurrence Matrix)
# Quantize image to 8 levels to simplify GLCM
quantized = (image / 32).astype(np.uint8)
glcm = graycomatrix(quantized, distances=[1], angles=[0], levels=8, symmetric=True, normed=True)
contrast = graycoprops(glcm, prop='contrast')[0,0]
dissimilarity = graycoprops(glcm, prop='dissimilarity')[0,0]
homogeneity = graycoprops(glcm, prop='homogeneity')[0,0]
energy = graycoprops(glcm, prop='energy')[0,0]
correlation = graycoprops(glcm, prop='correlation')[0,0]

print("\nTexture Features from GLCM:")
print(f"Contrast: {contrast:.4f}, Dissimilarity: {dissimilarity:.4f}")
print(f"Homogeneity: {homogeneity:.4f}, Energy: {energy:.4f}, Correlation: {correlation:.4f}")

# -------------------------------
# 3️⃣ Feature Description: ORB keypoints
orb = cv2.ORB_create(nfeatures=50)
keypoints, descriptors = orb.detectAndCompute(image, None)
image_features = cv2.drawKeypoints(image, keypoints, None, color=(0,255,0))

print("\nNumber of ORB keypoints detected:", len(keypoints))
if descriptors is not None:
    print("Descriptor matrix shape:", descriptors.shape)

# -------------------------------
# DISPLAY
plt.figure(figsize=(12,6))

plt.subplot(1,4,1)
plt.title("Original")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1,4,2)
plt.title("Edges (Canny)")
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.subplot(1,4,3)
plt.title("Texture (GLCM quantized)")
plt.imshow(quantized, cmap='gray')
plt.axis('off')

plt.subplot(1,4,4)
plt.title("ORB Features")
plt.imshow(image_features)
plt.axis('off')

plt.show()