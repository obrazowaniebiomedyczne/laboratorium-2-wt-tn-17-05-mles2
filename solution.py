import numpy as np
import cv2
import math

def close_open(img,kernel):
    closing_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    result = opening = cv2.morphologyEx(closing_img, cv2.MORPH_OPEN, kernel)
    return result

# Zadanie na ocenę dostateczną
def renew_pictures():
    kernel1 = np.ones((3,3),np.uint8)
    kernel2 = np.ones((5,5),np.uint8)

    img_1 = cv2.imread('figures/crushed.png')
    img_1 = close_open(img_1, kernel1)
    cv2.imwrite("figures/renewed.png", img_1)

    img_2 = cv2.imread('figures/crushed2.png')
    img_2 = close_open(img_2, kernel1)
    cv2.imwrite("figures/renewed2.png", img_2)

    img_3 = cv2.imread('figures/crushed3.png')
    img_3 = close_open(img_3, kernel2)
    cv2.imwrite("figures/renewed3.png", img_3)

    img_4 = cv2.imread('figures/crushed4.png')
    img_4 = close_open(img_4, kernel2)
    cv2.imwrite("figures/renewed4.png", img_4)

def prepare_image_for_erosion(image, kernel):
    new_shape = ((image.shape[0] + kernel[0].size), (image.shape[1] + kernel[1].size), 3)
    copied_image = np.ones(new_shape, dtype=image.dtype)
    copied_image = np.multiply(copied_image,255)
    
    x_offset = int(kernel[0].size / 2)
    y_offset = int(kernel[1].size / 2)

    for k in range(image.shape[2]):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                copied_image[i+x_offset, j + y_offset ,k] = image[i,j,k]
    return copied_image



def implication(p, q):
    return (not p) or q


def is_kernel_matched(kernel, img_under_kernel):
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            if not implication(kernel[i,j], img_under_kernel[i,j]):
                return True
    return False


def get_image_under_mask(img, kernel_shape, i, j):
    x_offset = int(kernel_shape[0]/2)
    y_offset = int(kernel_shape[1]/2)

    i_begin = i - x_offset
    j_begin = j - y_offset

    i_end = i + x_offset + 1
    j_end = j + y_offset + 1

    return img[i_begin:i_end, j_begin:j_end]

def erosion(image, kernel):
    prepared_image = prepare_image_for_erosion(image, kernel)
    x_offset = int(kernel[0].size / 2)
    y_offset = int(kernel[1].size / 2)

    for k in range(image.shape[2]):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if is_kernel_matched(kernel, get_image_under_mask(prepared_image[:,:,k], kernel.shape, i + x_offset, j + y_offset)):
                    image[i,j,k] = 0

    return image


# Zadanie na ocenę dobrą
def own_simple_erosion(image):
    #binaryzacja obrazu:
    thresh, image = cv2.threshold(image,127,255,cv2.THRESH_BINARY)

    kernel = np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]])
    
    result = erosion(image, kernel)
    return result


# Zadanie na ocenę bardzo dobrą
def own_erosion(image, kernel=None):
    if kernel is None:
        kernel = np.array([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]])
    #binaryzacja obrazu:
    thresh, image = cv2.threshold(image,127,255,cv2.THRESH_BINARY)

    result = erosion(image, kernel)
    return result
