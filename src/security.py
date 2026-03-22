import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# Dataset path
DATASET_PATH = "../dataset/"
classes = ["healthy", "diseased"]

# Load all image paths
all_images = []
for category in classes:
    folder_path = os.path.join(DATASET_PATH, category)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        all_images.append(img_path)

# Pick a random image
img_path = random.choice(all_images)
image = cv2.imread(img_path)  # color image
image = cv2.resize(image, (200, 200))  # resize for display

print("Selected image:", img_path)

# -------------------------------
# LSB Steganography (on red channel)
secret_message = "Hi"
binary_message = ''.join(format(ord(c), '08b') for c in secret_message)
stego = image.copy()

# Flatten red channel for LSB
red_channel = stego[:,:,2].flatten()
for i in range(len(binary_message)):
    red_channel[i] = (red_channel[i] & 254) | int(binary_message[i])  # safe LSB
stego[:,:,2] = red_channel.reshape(stego[:,:,2].shape)
# Show first 20 pixels in red channel
print("Original red channel (first 20):", image[:,:,2].flatten()[:20])
print("Stego red channel (first 20):   ", stego[:,:,2].flatten()[:20])
# -------------------------------
# Watermarking using text
watermarked = stego.copy()
text = "PLANT"
font = cv2.FONT_HERSHEY_SIMPLEX
scale = 0.6
color = (0, 127, 0)  # Red text
thickness = 2

# Get text size
(text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)

# Position at bottom-right corner
x = watermarked.shape[1] - text_width - 10
y = watermarked.shape[0] - 10

# Overlay text watermark
overlay = watermarked.copy()
cv2.putText(overlay, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
alpha = 0.5
watermarked = cv2.addWeighted(overlay, alpha, watermarked, 1 - alpha, 0)

# -------------------------------
# Difference image for proof
diff_stego = cv2.absdiff(image, stego)
diff_watermark = cv2.absdiff(stego, watermarked)

# -------------------------------
# Display all images
plt.figure(figsize=(16,5))

plt.subplot(1,4,1)
plt.title("Original")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1,4,2)
plt.title("Stego (LSB)")
plt.imshow(cv2.cvtColor(stego, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1,4,3)
plt.title("Watermarked")
plt.imshow(cv2.cvtColor(watermarked, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()