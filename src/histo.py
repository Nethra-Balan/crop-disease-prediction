import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Load image
img_path = r"E:\college works\sem6\IP\proj\crop-disease-prediction\dataset\healthy\866c2369-534c-4963-8703-b5e0257fea81___RS_HL 4703.JPG"
image = cv2.imread(img_path, 0)
image = cv2.resize(image, (256,256))

print("Original Image Matrix:\n", image)

# -------------------------------
# Histogram (Before)
hist_before = cv2.calcHist([image],[0],None,[256],[0,256])

# -------------------------------
# Histogram Equalization
equalized = cv2.equalizeHist(image)

print("\nEqualized Image Matrix:\n", equalized)

# -------------------------------
# Histogram (After)
hist_after = cv2.calcHist([equalized],[0],None,[256],[0,256])

# -------------------------------
# DISPLAY IMAGES

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Equalized Image")
plt.imshow(equalized, cmap='gray')
plt.axis('off')

plt.show()

# -------------------------------
# DISPLAY HISTOGRAMS

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.title("Histogram Before")
plt.plot(hist_before)
plt.xlabel("Intensity")
plt.ylabel("Frequency")

plt.subplot(1,2,2)
plt.title("Histogram After")
plt.plot(hist_after)
plt.xlabel("Intensity")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()