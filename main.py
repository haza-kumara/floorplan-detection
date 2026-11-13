
import cv2
import os

# Lokasi dataset
image_path = "dataset/train/denah1.png"

# Membaca gambar
image = cv2.imread(image_path)

# Memeriksa apakah gambar berhasil dibaca
if image is None:
    print("Gambar gagal dibaca!")
else:
    print("Gambar berhasil dibaca!")

    # Menampilkan informasi gambar
    height, width, channels = image.shape

    print("Lebar gambar:", width)
    print("Tinggi gambar:", height)
    print("Jumlah channel:", channels)

    # Menampilkan gambar
    cv2.imshow("Denah Rumah", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
