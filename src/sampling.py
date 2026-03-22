import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Load image (GRAYSCALE)
img_path = r"E:\college works\sem6\IP\proj\crop-disease-prediction\dataset\diseased\4a516057-a035-4bfe-b6a2-a6978efe0837___RS_L.Scorch 0865.JPG"
image = cv2.imread(img_path, 0)
image = cv2.resize(image, (256,256))

# Print matrix (clean 5x5)
print("Original Matrix (first 5x5):\n", image[:5, :5])

# -------------------------------
# 🔹 SAMPLING

sizes = [128, 64, 32, 16]
sampled_images = []
sampled_titles = []

for s in sizes:
    temp = cv2.resize(image, (s,s))
    temp_up = cv2.resize(temp, (256,256), interpolation=cv2.INTER_NEAREST)
    sampled_images.append(temp_up)
    sampled_titles.append(f"{s}x{s}")
    
    print(f"\nSampled {s}x{s} Matrix:\n", temp[:5, :5])

# -------------------------------
# 🔹 QUANTIZATION

def quantize(img, levels):
    step = 256 // levels
    return (img // step) * step

levels_list = [128, 64, 32, 16, 8, 4]
quant_images = []
quant_titles = []

for lvl in levels_list:
    q = quantize(image, lvl)
    quant_images.append(q)
    quant_titles.append(f"{lvl} levels")
    
    print(f"\nQuantized ({lvl} levels) Matrix:\n", q[:5, :5])

# -------------------------------
# DISPLAY

plt.figure(figsize=(14,10))

# Original
plt.subplot(4,4,1)
plt.title("Original")
plt.imshow(image, cmap='gray')
plt.axis('off')

# Sampling
for i in range(len(sampled_images)):
    plt.subplot(4,4,i+2)
    plt.title(sampled_titles[i])
    plt.imshow(sampled_images[i], cmap='gray')
    plt.axis('off')

# Quantization
start = len(sampled_images) + 2
for i in range(len(quant_images)):
    plt.subplot(4,4,start+i)
    plt.title(quant_titles[i])
    plt.imshow(quant_images[i], cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.show()