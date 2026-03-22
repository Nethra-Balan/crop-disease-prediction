import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Load image
img_path = r"E:\college works\sem6\IP\proj\crop-disease-prediction\dataset\diseased\4a516057-a035-4bfe-b6a2-a6978efe0837___RS_L.Scorch 0865.JPG"
image = cv2.imread(img_path, 0)
image = cv2.resize(image, (256,256))

print("Original Matrix:\n", image)

# -------------------------------
# 1️⃣ Laplacian
laplacian = cv2.Laplacian(image, cv2.CV_64F)
laplacian = np.uint8(np.absolute(laplacian))

# -------------------------------
# 2️⃣ Sobel X
sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0)
sobelx = np.uint8(np.absolute(sobelx))

# -------------------------------
# 3️⃣ Sobel Y
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1)
sobely = np.uint8(np.absolute(sobely))

# -------------------------------
# 4️⃣ Unsharp Masking
blur = cv2.GaussianBlur(image, (5,5), 0)
unsharp = cv2.addWeighted(image, 1.5, blur, -0.5, 0)

# -------------------------------
# PRINT MATRICES

print("\nLaplacian Matrix:\n", laplacian)
print("\nSobel X Matrix:\n", sobelx)
print("\nSobel Y Matrix:\n", sobely)
print("\nUnsharp Mask Matrix:\n", unsharp)

# -------------------------------
# DISPLAY

titles = ["Original", "Laplacian", "Sobel X", "Sobel Y", "Unsharp"]
images = [image, laplacian, sobelx, sobely, unsharp]

plt.figure(figsize=(12,5))
for i in range(len(images)):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()