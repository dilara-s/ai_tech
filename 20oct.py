import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image


def svertka(image, kernel):
    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape

    result_h = image_h - kernel_h + 1
    result_w = image_w - kernel_w + 1
    result = np.zeros((result_h, result_w))

    for i in range(result_h):
        for j in range(result_w):
            piece = image[i:i + kernel_h, j:j + kernel_w]
            result[i, j] = np.sum(piece * kernel)
    return result

image = cv2.imread("image.jpeg", cv2.IMREAD_GRAYSCALE)

#plt.imshow(image, cmap="gray")
#plt.show()

matrix = np.array(image)
#matrix_s = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).T
#matrix_s = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).T
#matrix_s = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])/16
#matrix_s = np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]])
#matrix_s = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
matrix_s = np.random.random((45, 45))

result = svertka(matrix, matrix_s)

plt.imshow(result, cmap="gray")
plt.show()
