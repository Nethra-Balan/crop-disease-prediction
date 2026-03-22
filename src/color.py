import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Load image (COLOR)
img_path = r"E:\college works\sem6\IP\proj\crop-disease-prediction\dataset\healthy\da93ba82-4070-4373-b818-a6fea8c91351___RS_HL 4724.JPG"
image = cv2.imread(img_path)
image = cv2.resize(image, (256,256))

# Convert BGR → RGB
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("RGB Matrix (5x5):\n", rgb[:5,:5])

# -------------------------------
# HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
print("\nHSV Matrix (5x5):\n", hsv[:5,:5])

# -------------------------------
# HSL (called HLS in OpenCV)
hsl = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
print("\nHSL Matrix (5x5):\n", hsl[:5,:5])

# -------------------------------
# CMY (manual)
cmy = 255 - rgb
print("\nCMY Matrix (5x5):\n", cmy[:5,:5])

# -------------------------------
# CMYK (manual)
rgb_norm = rgb / 255.0
K = 1 - np.max(rgb_norm, axis=2)

C = (1 - rgb_norm[:,:,0] - K) / (1 - K + 1e-5)
M = (1 - rgb_norm[:,:,1] - K) / (1 - K + 1e-5)
Y = (1 - rgb_norm[:,:,2] - K) / (1 - K + 1e-5)

cmyk = np.dstack((C,M,Y,K))
print("\nCMYK Matrix (5x5):\n", cmyk[:5,:5])

# -------------------------------
# YCbCr (OpenCV uses YCrCb)
ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
print("\nYCbCr Matrix (5x5):\n", ycbcr[:5,:5])

# -------------------------------
# CIELAB
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
print("\nCIELAB Matrix (5x5):\n", lab[:5,:5])

# -------------------------------
# DISPLAY

plt.figure(figsize=(12,8))

titles = ["RGB", "HSV", "HSL", "CMY", "YCbCr", "CIELAB"]
images = [rgb, hsv, hsl, cmy, ycbcr, lab]

for i in range(len(images)):
    plt.subplot(2,3,i+1)
    plt.title(titles[i])
    
    if titles[i] == "RGB" or titles[i] == "CMY":
        plt.imshow(images[i])
    else:
        plt.imshow(images[i], cmap='jet')  # for visualization
    
    plt.axis('off')

plt.tight_layout()
plt.show()