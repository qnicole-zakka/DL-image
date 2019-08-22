import numpy as np

# openvino-compatible opencv was installed before
# to use another version of opencv-python:
import sys
sys.path=[' /home/xi/.virtualenvs/tf1/lib/python3.6/site-packages/']+sys.path
import cv2

print(cv2.__version__)

img = cv2.imread('baboon.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(gray, kp)
img = cv2.drawKeypoints(gray, kp)
cv2.imwrite('sift_keypoints.jpg', img)


