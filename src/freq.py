import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Load image
img_path = r"E:\college works\sem6\IP\proj\crop-disease-prediction\dataset\healthy\998f4c67-2e7e-4881-a865-4ce2dc940a43___RS_HL 1851.JPG"
image = cv2.imread(img_path, 0)
image = cv2.resize(image, (256,256))

print("Original Image Matrix:\n", image)

# -------------------------------
# FFT
f = np.fft.fft2(image)
f_shift = np.fft.fftshift(f)

magnitude = np.log1p(np.abs(f_shift))
magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
magnitude = magnitude.astype(np.uint8)

print("\nFFT Magnitude Matrix:\n", magnitude)

# -------------------------------
# Create Masks
rows, cols = image.shape
crow, ccol = rows//2, cols//2

# Low-pass mask (center square)
low_pass_mask = np.zeros((rows, cols), np.uint8)
low_pass_mask[crow-30:crow+30, ccol-30:ccol+30] = 1

# High-pass mask (inverse)
high_pass_mask = 1 - low_pass_mask

print("\nLow Pass Mask:\n", low_pass_mask)
print("\nHigh Pass Mask:\n", high_pass_mask)

# -------------------------------
# Apply Filters

# Low-pass filtering
f_low = f_shift * low_pass_mask
img_low = np.fft.ifft2(np.fft.ifftshift(f_low))
img_low = np.abs(img_low)
img_low = np.uint8(img_low)

# High-pass filtering
f_high = f_shift * high_pass_mask
img_high = np.fft.ifft2(np.fft.ifftshift(f_high))
img_high = np.abs(img_high)
img_high = np.uint8(img_high)

# -------------------------------
# DISPLAY

plt.figure(figsize=(12,8))

plt.subplot(2,3,1)
plt.title("Original")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(2,3,2)
plt.title("FFT Spectrum")
plt.imshow(magnitude, cmap='jet')
plt.axis('off')

plt.subplot(2,3,3)
plt.title("Low Pass Mask")
plt.imshow(low_pass_mask, cmap='gray')
plt.axis('off')

plt.subplot(2,3,4)
plt.title("High Pass Mask")
plt.imshow(high_pass_mask, cmap='gray')
plt.axis('off')

plt.subplot(2,3,5)
plt.title("Low Pass Result")
plt.imshow(img_low, cmap='gray')
plt.axis('off')

plt.subplot(2,3,6)
plt.title("High Pass Result")
plt.imshow(img_high, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()