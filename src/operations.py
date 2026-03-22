import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Load image
img_path = r"E:\college works\sem6\IP\proj\crop-disease-prediction\dataset\healthy\da93ba82-4070-4373-b818-a6fea8c91351___RS_HL 4724.JPG"
image = cv2.imread(img_path)
image = cv2.resize(image, (256,256))

# Convert to RGB for display
img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# -------------------------------
# GEOMETRIC OPERATIONS

resized = cv2.resize(image, (128,128))

M = cv2.getRotationMatrix2D((128,128), 45, 1)
rotated = cv2.warpAffine(image, M, (256,256))

flipped = cv2.flip(image, 1)

# -------------------------------
# ARITHMETIC OPERATIONS

image2 = cv2.GaussianBlur(image, (15,15), 0)

# ADDITION
add = cv2.add(image, image2)

# SUBTRACTION (enhanced)
sub = cv2.subtract(image, image2)
sub = cv2.normalize(sub, None, 0, 255, cv2.NORM_MINMAX)

# MULTIPLICATION (normalized)
img1_norm = image / 255.0
img2_norm = image2 / 255.0
mul = img1_norm * img2_norm
mul = (mul * 255).astype(np.uint8)
mul = cv2.normalize(mul, None, 0, 255, cv2.NORM_MINMAX)

# -------------------------------
# GRAYSCALE MATRICES (for clean display)

add_gray = cv2.cvtColor(add, cv2.COLOR_BGR2GRAY)
sub_gray = cv2.cvtColor(sub.astype(np.uint8), cv2.COLOR_BGR2GRAY)
mul_gray = cv2.cvtColor(mul, cv2.COLOR_BGR2GRAY)

print("Addition Matrix (5x5):\n", add_gray[:5,:5])
print("\nSubtraction Matrix (5x5):\n", sub_gray[:5,:5])
print("\nMultiplication Matrix (5x5):\n", mul_gray[:5,:5])

# -------------------------------
# LOGICAL OPERATIONS

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

gray2 = cv2.GaussianBlur(gray, (15,15), 0)
_, binary2 = cv2.threshold(gray2, 127, 255, cv2.THRESH_BINARY)

and_img = cv2.bitwise_and(binary1, binary2)
or_img = cv2.bitwise_or(binary1, binary2)
xor_img = cv2.bitwise_xor(binary1, binary2)
not_img = cv2.bitwise_not(binary1)

# PRINT LOGICAL MATRICES
print("\nBinary Matrix (5x5):\n", binary1[:5,:5])
print("\nAND Matrix (5x5):\n", and_img[:5,:5])
print("\nOR Matrix (5x5):\n", or_img[:5,:5])
print("\nXOR Matrix (5x5):\n", xor_img[:5,:5])
print("\nNOT Matrix (5x5):\n", not_img[:5,:5])

# -------------------------------
# DISPLAY

plt.figure(figsize=(14,10))

# Geometric
plt.subplot(3,4,1)
plt.title("Original")
plt.imshow(img_rgb)
plt.axis('off')

plt.subplot(3,4,2)
plt.title("Resized")
plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(3,4,3)
plt.title("Rotated")
plt.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(3,4,4)
plt.title("Flipped")
plt.imshow(cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Arithmetic
plt.subplot(3,4,5)
plt.title("Addition")
plt.imshow(cv2.cvtColor(add, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(3,4,6)
plt.title("Subtraction")
plt.imshow(cv2.cvtColor(sub.astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(3,4,7)
plt.title("Multiplication")
plt.imshow(cv2.cvtColor(mul, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Logical
plt.subplot(3,4,9)
plt.title("AND")
plt.imshow(and_img, cmap='gray')
plt.axis('off')

plt.subplot(3,4,10)
plt.title("OR")
plt.imshow(or_img, cmap='gray')
plt.axis('off')

plt.subplot(3,4,11)
plt.title("XOR")
plt.imshow(xor_img, cmap='gray')
plt.axis('off')

plt.subplot(3,4,12)
plt.title("NOT")
plt.imshow(not_img, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()