import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hadamard
import pywt

# -------------------------------
# Load image
img_path = r"E:\college works\sem6\IP\proj\crop-disease-prediction\dataset\diseased\85b88dd2-5e69-4de4-9f41-e675c21427ac___RS_L.Scorch 1579.JPG"
image = cv2.imread(img_path, 0)
image = cv2.resize(image, (256,256))

print("Original Image Matrix:")
print(image)

# -------------------------------
# 1️⃣ DCT
image_float = np.float32(image)
dct = cv2.dct(image_float)
dct_log = np.log1p(np.abs(dct))
dct_norm = (255 * dct_log / np.max(dct_log)).astype(np.uint8)

print("\nDCT Matrix:")
print(dct)

# -------------------------------
# 2️⃣ WHT (Walsh-Hadamard)
H = hadamard(256)
wht = H @ image @ H
wht_log = np.log1p(np.abs(wht))
wht_norm = (255 * wht_log / np.max(wht_log)).astype(np.uint8)

print("\nWHT Matrix:")
print(wht)

# -------------------------------
# 3️⃣ Haar Transform
def haar_transform(img):
    img = np.float32(img)
    rows, cols = img.shape
    temp = np.zeros_like(img)

    # Row transform
    for i in range(rows):
        for j in range(0, cols, 2):
            avg = (img[i,j] + img[i,j+1]) / 2
            diff = (img[i,j] - img[i,j+1]) / 2
            temp[i, j//2] = avg
            temp[i, j//2 + cols//2] = diff

    result = np.zeros_like(img)

    # Column transform
    for j in range(cols):
        for i in range(0, rows, 2):
            avg = (temp[i,j] + temp[i+1,j]) / 2
            diff = (temp[i,j] - temp[i+1,j]) / 2
            result[i//2, j] = avg
            result[i//2 + rows//2, j] = diff

    return result

haar = haar_transform(image)
haar_log = np.log1p(np.abs(haar))
haar_norm = (255 * haar_log / np.max(haar_log)).astype(np.uint8)

print("\nHaar Matrix:")
print(haar)

# -------------------------------
# 4️⃣ Wavelet Transform
coeffs2 = pywt.dwt2(image, 'db1')
LL, (LH, HL, HH) = coeffs2

# Combine for display
top = np.hstack((LL, LH))
bottom = np.hstack((HL, HH))
wavelet_img = np.vstack((top, bottom))

wavelet_log = np.log1p(np.abs(wavelet_img))
wavelet_norm = (255 * wavelet_log / np.max(wavelet_log)).astype(np.uint8)

print("\nWavelet LL Matrix:")
print(LL)

# -------------------------------
# DISPLAY ALL RESULTS

plt.figure(figsize=(12,8))

plt.subplot(2,3,1)
plt.title("Original")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(2,3,2)
plt.title("DCT")
plt.imshow(dct_norm, cmap='jet')
plt.axis('off')

plt.subplot(2,3,3)
plt.title("WHT")
plt.imshow(wht_norm, cmap='jet')
plt.axis('off')

plt.subplot(2,3,4)
plt.title("Haar")
plt.imshow(haar_norm, cmap='jet')
plt.axis('off')

plt.subplot(2,3,5)
plt.title("Wavelet (LL LH HL HH)")
plt.imshow(wavelet_norm, cmap='jet')
plt.axis('off')

plt.tight_layout()
plt.show()