import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Load image
img_path = r"E:\college works\sem6\IP\proj\crop-disease-prediction\dataset\diseased\1ba17895-c6a5-404a-967c-ef33d16a5b65___RS_L.Scorch 0108.JPG"
image = cv2.imread(img_path, 0)
image = cv2.resize(image, (256,256))

print("Original Image Matrix:\n", image)

# Normalize (0–1)
img_norm = image / 255.0

# -------------------------------
# 1️⃣ Identity
identity = image.copy()

# -------------------------------
# 2️⃣ Negative
negative = 255 - image

# -------------------------------
# 3️⃣ Log Transform (FIXED)
log_transform = np.log1p(image.astype(np.float32))
log_transform = cv2.normalize(log_transform, None, 0, 255, cv2.NORM_MINMAX)
log_transform = log_transform.astype(np.uint8)

# -------------------------------
# 4️⃣ Inverse Log
inv_log = np.exp(img_norm) - 1
inv_log = cv2.normalize(inv_log, None, 0, 255, cv2.NORM_MINMAX)
inv_log = inv_log.astype(np.uint8)

# -------------------------------
# 5️⃣ Gamma > 1
gamma_high = np.power(img_norm, 2)
gamma_high = (gamma_high * 255).astype(np.uint8)

# -------------------------------
# 6️⃣ Gamma < 1
gamma_low = np.power(img_norm, 0.5)
gamma_low = (gamma_low * 255).astype(np.uint8)

# -------------------------------
# 7️⃣ Contrast Stretching
min_val = np.min(image)
max_val = np.max(image)
contrast = (image - min_val) * (255 / (max_val - min_val))
contrast = contrast.astype(np.uint8)

# -------------------------------
# 8️⃣ Thresholding
_, threshold = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# -------------------------------
# 9️⃣ Gray Level Slicing
slice_img = np.where((image > 100) & (image < 180), 255, 0).astype(np.uint8)

# -------------------------------
# PRINT MATRICES (ALL)

print("\nIdentity Matrix:\n", identity)
print("\nNegative Matrix:\n", negative)
print("\nLog Transform Matrix:\n", log_transform)
print("\nInverse Log Matrix:\n", inv_log)
print("\nGamma > 1 Matrix:\n", gamma_high)
print("\nGamma < 1 Matrix:\n", gamma_low)
print("\nContrast Matrix:\n", contrast)
print("\nThreshold Matrix:\n", threshold)
print("\nGray Level Slicing Matrix:\n", slice_img)

# -------------------------------
# DISPLAY IMAGES

titles = ["Original", "Identity", "Negative",
          "Log", "Inverse Log",
          "Gamma>1", "Gamma<1",
          "Contrast", "Threshold", "Slicing"]

images = [image, identity, negative,
          log_transform, inv_log,
          gamma_high, gamma_low,
          contrast, threshold, slice_img]

plt.figure(figsize=(15,10))
for i in range(len(images)):
    plt.subplot(4,3,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()

# -------------------------------
# GRAPH (Intensity Mapping)

x = np.linspace(0,1,100)

plt.figure(figsize=(8,6))
plt.plot(x, x, label="Identity")
plt.plot(x, 1-x, label="Negative")
plt.plot(x, np.log1p(x), label="Log")
plt.plot(x, (np.exp(x)-1)/np.max(np.exp(x)-1), label="Inverse Log")
plt.plot(x, x**2, label="Gamma>1")
plt.plot(x, x**0.5, label="Gamma<1")

plt.legend()
plt.title("Gray Level Transformations")
plt.xlabel("Input Intensity")
plt.ylabel("Output Intensity")
plt.grid()
plt.show()