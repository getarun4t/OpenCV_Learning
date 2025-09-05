import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_canny(image):
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(grey, (5,5), 0)
    return cv2.Canny(blur, 50, 150)
    

image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)
canny = get_canny(lane_image)
plt.imshow(canny)
plt.show()