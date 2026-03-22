import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Load grayscale image
img_path = r"E:\college works\sem6\IP\proj\crop-disease-prediction\dataset\healthy\1faaac0a-ea12-4085-af35-ab8b70bbf60a___RS_HL 4725.JPG"
image = cv2.imread(img_path, 0)
image = cv2.resize(image, (256,256))

# Convert to binary
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Select pixel
x, y = 128, 128

print("Selected pixel value:", binary[x, y])
print("\n5x5 Matrix around pixel:\n", binary[x-2:x+3, y-2:y+3])

# -------------------------------
# Convert to color for marking
img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

# Define neighbors
neighbors_4 = [(x-1,y), (x+1,y), (x,y-1), (x,y+1)]
neighbors_8 = neighbors_4 + [(x-1,y-1),(x-1,y+1),(x+1,y-1),(x+1,y+1)]
neighbors_diag = [(x-1,y-1),(x-1,y+1),(x+1,y-1),(x+1,y+1)]

# -------------------------------
# Function to mark and zoom
def show_zoom(neighbors, title, color):
    temp = img.copy()

    # Mark center
    cv2.circle(temp, (y,x), 5, (0,0,255), -1)

    # Mark neighbors
    for (i,j) in neighbors:
        cv2.circle(temp, (j,i), 5, color, -1)

    # ZOOM REGION (important 🔥)
    zoom = temp[x-20:x+20, y-20:y+20]

    # Enlarge zoom for visibility
    zoom = cv2.resize(zoom, (200,200), interpolation=cv2.INTER_NEAREST)

    plt.imshow(cv2.cvtColor(zoom, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')

# -------------------------------
# DISPLAY

plt.figure(figsize=(10,3))

plt.subplot(1,3,1)
show_zoom(neighbors_4, "4-Connectivity", (255,0,0))

plt.subplot(1,3,2)
show_zoom(neighbors_8, "8-Connectivity", (0,255,0))

plt.subplot(1,3,3)
show_zoom(neighbors_diag, "Diagonal", (0,255,255))

plt.show()