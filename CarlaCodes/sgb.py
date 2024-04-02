import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

left_img = cv2.imread('outputs/output_synchronized/020453_left.png')
right_img = cv2.imread('outputs/output_synchronized/020453_right.png')

stereo_bm = cv2.StereoBM_create(numDisparities=16, blockSize=15)
dispmap_bm = stereo_bm.compute(cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY))

stereo_sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=16, blockSize=15)
dispmap_sgbm = stereo_sgbm.compute(left_img, right_img)

plt.figure(figsize=(12, 10))
plt.subplot(2, 1, 1)
plt.title('Global Matching')
plt.imshow(dispmap_bm)

plt.subplot(2, 1, 2)
plt.title('Semi-Global Matching')
plt.imshow(dispmap_sgbm)
plt.show()
