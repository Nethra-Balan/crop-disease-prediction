import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Load base and overlay images
base_path = r"E:\college works\sem6\IP\proj\crop-disease-prediction\dataset\healthy\da93ba82-4070-4373-b818-a6fea8c91351___RS_HL 4724.JPG"
overlay_path = r"E:\college works\sem6\IP\proj\crop-disease-prediction\dataset\healthy\1faaac0a-ea12-4085-af35-ab8b70bbf60a___RS_HL 4725.JPG"

base = cv2.imread(base_path)
overlay = cv2.imread(overlay_path)

# Resize both images to same size
base = cv2.resize(base, (256,256))
overlay = cv2.resize(overlay, (256,256))

print("Base Image Matrix (first 5x5 pixels):\n", base[:5,:5])
print("\nOverlay Image Matrix (first 5x5 pixels):\n", overlay[:5,:5])

# -------------------------------
# 1️⃣ Alpha Blending
alpha = 0.5
blended = cv2.addWeighted(base, alpha, overlay, 1-alpha, 0)
print("\nBlended Image Matrix (first 5x5 pixels):\n", blended[:5,:5])

# -------------------------------
# 2️⃣ Masking
# Convert overlay to grayscale and create a binary mask
gray_overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray_overlay, 128, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

# Extract regions
base_bg = cv2.bitwise_and(base, base, mask=mask_inv)
overlay_fg = cv2.bitwise_and(overlay, overlay, mask=mask)

# Combine
composited = cv2.add(base_bg, overlay_fg)
print("\nComposited Image Matrix (first 5x5 pixels):\n", composited[:5,:5])

# -------------------------------
# 3️⃣ Color Tint Effect
tint = np.zeros_like(base)
tint[:,:,1] = 100  # add green tint
tinted_image = cv2.addWeighted(base, 0.7, tint, 0.3, 0)
print("\nTinted Image Matrix (first 5x5 pixels):\n", tinted_image[:5,:5])

# -------------------------------
# DISPLAY
plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.title("Blended (Alpha)")
plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2,2,2)
plt.title("Composited (Masking)")
plt.imshow(cv2.cvtColor(composited, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2,2,3)
plt.title("Tinted Image")
plt.imshow(cv2.cvtColor(tinted_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2,2,4)
plt.title("Original Base Image")
plt.imshow(cv2.cvtColor(base, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()