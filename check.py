from solution import *
from obpng import read_png, write_png
import cv2

print("- Ocena dostateczna")
renew_pictures()


print("- Ocena dobra")
img = cv2.imread('figures/crushed2.png')
erosion_img = own_simple_erosion(img)
cv2.imwrite("figures/own_simple_erosion.png", erosion_img)


print("- Ocena bardzo dobra")
image = cv2.imread("figures/crushed3.png")
kernel = np.array([[0, 1, 1, 1, 0],
                   [0, 1, 1, 1, 0],
                   [1, 1, 1, 1, 1],
                   [0, 1, 1, 1, 0],
                   [0, 1, 1, 1, 0]], np.uint8)
erosion_img = erosion(image, kernel)
cv2.imwrite("figures/erosion.png", erosion_img)

