import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_canny(image):
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(grey, (5,5), 0)
    return cv2.Canny(blur, 50, 150)
    
def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(200, height), (1100, height), (550, 200)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    return mask

image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)
canny = get_canny(lane_image)
cv2.imshow("result", region_of_interest(canny))
cv2.waitKey(0)