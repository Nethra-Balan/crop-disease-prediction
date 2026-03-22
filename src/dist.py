import numpy as np
import cv2
import matplotlib.pyplot as plt

# Create blank grid
img = np.ones((200,200,3), dtype=np.uint8) * 255

# Define two points
p1 = (50, 50)
p2 = (150, 120)

x1, y1 = p1
x2, y2 = p2

# -------------------------------
# DISTANCE CALCULATIONS

euclidean = np.sqrt((x1-x2)**2 + (y1-y2)**2)
cityblock = abs(x1-x2) + abs(y1-y2)
chessboard = max(abs(x1-x2), abs(y1-y2))

print("Point 1:", p1)
print("Point 2:", p2)

print("\nEuclidean Distance:", euclidean)
print("City Block Distance:", cityblock)
print("Chessboard Distance:", chessboard)

# -------------------------------
# DRAW POINTS

cv2.circle(img, (y1,x1), 5, (0,0,255), -1)  # Red
cv2.circle(img, (y2,x2), 5, (255,0,0), -1)  # Blue

# -------------------------------
# Draw Euclidean (straight line)
img_eu = img.copy()
cv2.line(img_eu, (y1,x1), (y2,x2), (0,0,0), 2)

# -------------------------------
# Draw City Block (L path)
img_city = img.copy()
cv2.line(img_city, (y1,x1), (y1,x2), (0,255,0), 2)
cv2.line(img_city, (y1,x2), (y2,x2), (0,255,0), 2)

# -------------------------------
# Draw Chessboard (diagonal max steps)
img_chess = img.copy()

dx = x2 - x1
dy = y2 - y1

steps = max(abs(dx), abs(dy))
for i in range(steps+1):
    xi = int(x1 + i * dx / steps)
    yi = int(y1 + i * dy / steps)
    img_chess[xi, yi] = [0,0,0]

# -------------------------------
# DISPLAY

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Euclidean (Straight Line)")
plt.imshow(cv2.cvtColor(img_eu, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1,3,2)
plt.title("City Block (Manhattan)")
plt.imshow(cv2.cvtColor(img_city, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1,3,3)
plt.title("Chessboard Distance")
plt.imshow(cv2.cvtColor(img_chess, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()