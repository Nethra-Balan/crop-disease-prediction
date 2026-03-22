import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Load sample image
img_path = r"E:\college works\sem6\IP\proj\crop-disease-prediction\dataset\healthy\da93ba82-4070-4373-b818-a6fea8c91351___RS_HL 4724.JPG"
image = cv2.imread(img_path)
image = cv2.resize(image, (256,256))

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print("Original Grayscale Image matrix:")
print(gray)

# -------------------------------
# Show original
plt.figure(figsize=(5,5))
plt.title("Original Grayscale Image")
plt.imshow(gray, cmap='gray')
plt.axis('off')
plt.show()

# =========================================================
# -------------------- DFT -------------------------------
# (Using OpenCV DFT for distinction)

dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# Magnitude spectrum
magnitude_dft = cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1])
magnitude_dft_log = np.log1p(magnitude_dft)
magnitude_dft_norm = 255 * magnitude_dft_log / np.max(magnitude_dft_log)
magnitude_dft_norm = magnitude_dft_norm.astype(np.uint8)

plt.figure(figsize=(5,5))
plt.title("DFT Magnitude Spectrum")
plt.imshow(magnitude_dft_norm, cmap='jet')
plt.axis('off')
plt.show()

print("DFT Magnitude Matrix:")
print(magnitude_dft_norm)

# -------------------------------
# -------------------------------
# IDFT Reconstruction (FIXED)

dft_ishift = np.fft.ifftshift(dft_shift)

img_back_dft = cv2.idft(dft_ishift, flags=cv2.DFT_SCALE)
img_back_dft = cv2.magnitude(img_back_dft[:,:,0], img_back_dft[:,:,1])

# Convert properly to uint8
img_back_dft = np.clip(img_back_dft, 0, 255)
img_back_dft = img_back_dft.astype(np.uint8)

plt.figure(figsize=(5,5))
plt.title("Reconstructed Image (DFT)")
plt.imshow(img_back_dft, cmap='gray')
plt.axis('off')
plt.show()

print("Reconstructed Image Matrix (DFT - Corrected):")
print(img_back_dft)

# =========================================================
# -------------------- FFT -------------------------------

fft = np.fft.fft2(gray)
fft_shift = np.fft.fftshift(fft)

magnitude_fft = np.abs(fft_shift)
magnitude_fft_log = np.log1p(magnitude_fft)
magnitude_fft_norm = 255 * magnitude_fft_log / np.max(magnitude_fft_log)
magnitude_fft_norm = magnitude_fft_norm.astype(np.uint8)

plt.figure(figsize=(5,5))
plt.title("FFT Magnitude Spectrum")
plt.imshow(magnitude_fft_norm, cmap='jet')
plt.axis('off')
plt.show()

print("FFT Magnitude Matrix:")
print(magnitude_fft_norm)

# -------------------------------
# IFFT Reconstruction

ifft_shift = np.fft.ifftshift(fft_shift)
img_back_fft = np.fft.ifft2(ifft_shift)
img_back_fft = np.abs(img_back_fft)
img_back_fft = np.round(img_back_fft).astype(np.uint8)

plt.figure(figsize=(5,5))
plt.title("Reconstructed Image (FFT)")
plt.imshow(img_back_fft, cmap='gray')
plt.axis('off')
plt.show()

print("Reconstructed Image Matrix (FFT):")
print(img_back_fft)