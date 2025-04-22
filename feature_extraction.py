import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import color
 
# Đọc ảnh
image = cv2.imread('car.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 1. Histogram màu
def extract_color_histogram(image, bins=(8, 8, 8)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins,
                        [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# 2. Trích chọn cạnh bằng Canny
def extract_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

# 3. Trích chọn HOG
def extract_hog(image):
    gray = color.rgb2gray(image)
    hog_features, hog_image = hog(gray, pixels_per_cell=(8, 8),
                                   cells_per_block=(2, 2),
                                   visualize=True)
    return hog_features, hog_image

# Gọi các hàm trích chọn
hist = extract_color_histogram(image)
edges = extract_edges(image)
hog_features, hog_image = extract_hog(image_rgb)

# Hiển thị ảnh gốc và các đặc trưng
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title("Ảnh gốc")

plt.subplot(1, 3, 2)
plt.imshow(edges, cmap='gray')
plt.title("Canny Edges")

plt.subplot(1, 3, 3)
plt.imshow(hog_image, cmap='gray')
plt.title("HOG Visualization")

plt.tight_layout()
plt.show()
