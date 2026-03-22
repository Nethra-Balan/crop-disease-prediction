import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Load image
img_path = r"E:\college works\sem6\IP\proj\crop-disease-prediction\dataset\healthy\6e8627f0-d19e-46c2-876a-d36c5c57c25a___RS_HL 4503.JPG"
image = cv2.imread(img_path, 0)
image = cv2.resize(image, (256,256))

print("Original Matrix:\n", image)

# -------------------------------
# 1️⃣ Mean Filter
mean = cv2.blur(image, (5,5))

# -------------------------------
# 2️⃣ Gaussian Filter
gaussian = cv2.GaussianBlur(image, (5,5), 0)

# -------------------------------
# 3️⃣ Median Filter
median = cv2.medianBlur(image, 5)

# -------------------------------
# 4️⃣ Bilateral Filter
bilateral = cv2.bilateralFilter(image, 9, 75, 75)

# -------------------------------
# PRINT MATRICES

print("\nMean Filter Matrix:\n", mean)
print("\nGaussian Filter Matrix:\n", gaussian)
print("\nMedian Filter Matrix:\n", median)
print("\nBilateral Filter Matrix:\n", bilateral)

# -------------------------------
# DISPLAY

titles = ["Original", "Mean", "Gaussian", "Median", "Bilateral"]
images = [image, mean, gaussian, median, bilateral]

plt.figure(figsize=(12,5))
for i in range(len(images)):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()