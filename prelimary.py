import numpy as np
import matplotlib.pyplot as plt

def color2gray(img):
    b = [.3, .6, .1]
    return np.dot(img[...,:3], b)

image = plt.imread("houndog1.png")
gray_image = color2gray(image)
plt.imshow(gray_image, cmap="gray")
plt.show()
plt.imsave("gray_hounddog1.png", gray_image, cmap="gray")