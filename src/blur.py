import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Load grayscale image
img_path = r"E:\college works\sem6\IP\proj\crop-disease-prediction\dataset\healthy\da93ba82-4070-4373-b818-a6fea8c91351___RS_HL 4724.JPG"
image = cv2.imread(img_path, 0)  # grayscale
image = cv2.resize(image, (256,256))

print("Original Image Matrix (first 5x5 pixels):")
print(image[:5,:5])

# -------------------------------
# 🔹 BLUR (Average Blur)
blur = cv2.blur(image, (5,5))
print("\nBlurred Image Matrix (first 5x5 pixels):")
print(blur[:5,:5])

# Display Original → Blur
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Blurred")
plt.imshow(blur, cmap='gray')
plt.axis('off')
plt.show()

# -------------------------------
# 🔹 IMAGE RESTORATION (Wiener Filter)
def wiener_filter(img, kernel_size=5, K=0.01):
    img_float = np.float32(img)
    local_mean = cv2.blur(img_float, (kernel_size,kernel_size))
    local_var = cv2.blur(img_float**2, (kernel_size,kernel_size)) - local_mean**2
    result = local_mean + (local_var/(local_var + K)) * (img_float - local_mean)
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

restored = wiener_filter(blur, kernel_size=5, K=0.01)
print("\nRestored Image Matrix (first 5x5 pixels):")
print(restored[:5,:5])

# Display Original → Restored
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Restored")
plt.imshow(restored, cmap='gray')
plt.axis('off')
plt.show()