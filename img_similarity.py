import numpy as np
import scipy
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
# openvino-compatible opencv was installed before
# to use another version of opencv-python:
import sys
sys.path=[' /home/xi/.virtualenvs/tf1/lib/python3.6/site-packages/']+sys.path
import cv2

print(cv2.__version__)

img = cv2.imread('baboon.jpg')
img2 = cv2.imread('perturbed_baboon.jpg')
rgb = ('b', 'g', 'r')
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#sift = cv2.xfeatures2d.SIFT_create()
#kp, des = sift.detectAndCompute(gray, kp)
#img = cv2.drawKeypoints(gray, kp)
#cv2.imwrite('sift_keypoints.jpg', img)

# plot rgb channels histogram
cleanHist = []
badHist = []
plt.figure()
for i, channel in enumerate(rgb):
    hist = cv2.calcHist([img], [i], None, [256], [0,256])
    hist2 = cv2.calcHist([img2], [i], None, [256], [0,256])
    euc = dist.euclidean(hist, hist2)
    cleanHist.append(hist)
    badHist.append(hist2)
    plt.subplot(3, 1, 1)
    plt.plot(hist, color=channel)
    plt.subplot(3, 1, 2)
    plt.plot(hist2, color=channel)
    plt.subplot(3, 1, 3)
    plt.bar(i, euc, color=channel)
    print(euc)
plt.savefig('hist-compare.jpg')
#plt.show()

# calculate distance between histograms of clean and perturbed
DIST_METHODS = (
    ('Euclidean', dist.euclidean),
    ('Manhattan', dist.cityblock), 
    ('Chebysev', dist.chebyshev)
)


print('cleanHist shape = {}, badHist shape = {}'.format(len(cleanHist), len(badHist)))