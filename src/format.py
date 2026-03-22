import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Load image
img_path = r"E:\college works\sem6\IP\proj\crop-disease-prediction\dataset\healthy\da93ba82-4070-4373-b818-a6fea8c91351___RS_HL 4724.JPG"
image = cv2.imread(img_path)

# Output folder
output_folder = "output_formats"
os.makedirs(output_folder, exist_ok=True)

# -------------------------------
# Save in different formats

cv2.imwrite(os.path.join(output_folder, "image.bmp"), image)

cv2.imwrite(os.path.join(output_folder, "image.jpg"), image,
            [cv2.IMWRITE_JPEG_QUALITY, 50])  # lossy

cv2.imwrite(os.path.join(output_folder, "image.png"), image)

cv2.imwrite(os.path.join(output_folder, "image.tiff"), image)

cv2.imwrite(os.path.join(output_folder, "image.gif"), image)

# RAW (dump bytes)
image.tofile(os.path.join(output_folder, "image.raw"))

print("Images saved.")

# -------------------------------
# File sizes

print("\nFile Sizes:")
for file in os.listdir(output_folder):
    size = os.path.getsize(os.path.join(output_folder, file)) / 1024
    print(f"{file} : {size:.2f} KB")

# -------------------------------
# DISPLAY IMAGES

formats = ["bmp", "jpg", "png", "tiff", "gif"]
images = []

for fmt in formats:
    path = os.path.join(output_folder, f"image.{fmt}")
    img = cv2.imread(path)
    
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    else:
        images.append(np.zeros((256,256,3), dtype=np.uint8))

# RAW (reconstruct manually)
raw_path = os.path.join(output_folder, "image.raw")
raw_data = np.fromfile(raw_path, dtype=np.uint8)

try:
    raw_img = raw_data.reshape(image.shape)
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
except:
    raw_img = np.zeros((256,256,3), dtype=np.uint8)

images.append(raw_img)
formats.append("raw")

# -------------------------------
# Plot

plt.figure(figsize=(12,6))

for i in range(len(images)):
    plt.subplot(2,3,i+1)
    plt.imshow(images[i])
    plt.title(formats[i].upper())
    plt.axis('off')

plt.tight_layout()
plt.show()