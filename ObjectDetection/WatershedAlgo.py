import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def plot(img, title='', cmap='gray'):
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.show()


sep_coins = cv.imread('../DATA/pennies.jpg')
plot(sep_coins, 'coins')

""" TRY NON WATERSHED APPROACH {ENDS IN FAILING} """
"""# Median Blur ( To remove useless info )
sep_blur = cv.medianBlur(sep_coins, 25)
# Convert to Grayscale
gray_sep_coins = cv.cvtColor(sep_blur, cv.COLOR_BGR2GRAY)
plot(gray_sep_coins, 'gray coins')
# Binary Threshold ( to make it black and white)
ret, sep_thresh = cv.threshold(gray_sep_coins, 160, 255, cv.THRESH_BINARY_INV)
plot(sep_thresh, 'binary coins')
# Find Contours
contours, hierarchy = cv.findContours(sep_thresh.copy(), cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:  # if external
        cv.drawContours(sep_coins, contours, i, (255, 0, 0), 10)
plot(sep_coins, 'False contour draw')"""

""" USING WATERSHED """
