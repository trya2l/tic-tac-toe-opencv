
import cv2
import numpy as np
from matplotlib import pyplot as plt

# reading image
tic = cv2.imread("img/ima.png")

gray = cv2.cvtColor(tic, cv2.COLOR_BGR2GRAY)

img = cv2.medianBlur(gray, 5)

cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT,
                           1,120, param1=100, param2=30, minRadius=0, maxRadius=0)

circles = np.uint16(np.around(circles))

for circle in circles[0, :]:

    # circle
    cv2.circle(tic, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)

    # center
    cv2.circle(tic, (circle[0], circle[1]), 2, (0, 255, 0), 3)


cv2.imshow("cicrles", tic)
cv2.waitKey()
cv2.destroyAllWindows()
