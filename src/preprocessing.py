
import cv2
import os

# Membaca gambar
image = cv2.imread("dataset/train/denah1.png")

if image is None:
    print("Gambar tidak ditemukan!")
    exit()

# 1. Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. Gaussian Blur
blur = cv2.GaussianBlur(gray, (3, 3), 0)

# 3. Thresholding
_, threshold = cv2.threshold(
    blur, 200, 255, cv2.THRESH_BINARY_INV
)

# Menyimpan hasil
os.makedirs("results", exist_ok=True)

cv2.imwrite("results/grayscale.png", gray)
cv2.imwrite("results/threshold.png", threshold)

print("Preprocessing berhasil!")
print("Hasil tersimpan di folder results")
