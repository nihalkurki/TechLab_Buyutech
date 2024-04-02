import cv2, os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import tqdm

# root_dir = "/Users/evanwyf/Desktop/carla_data/data/test"
# bm_dir = "non_learning_output/bm"
# sgbm_dir = "non_learning_output/sgbm"

# left_imgs = sorted([osp.join(root_dir, "left", l) for l in os.listdir(osp.join(root_dir,"left")) if "left" in l])
# right_imgs = sorted([osp.join(root_dir, "right", r) for r in os.listdir(osp.join(root_dir,"right")) if "right" in r])
# gt_depth = sorted([depth for depth in os.listdir(osp.join(root_dir,"gt_depth"))])

# print(len(left_imgs), len(right_imgs), len(gt_depth))

# for i in tqdm.tqdm(range(len(left_imgs))):
    
#     left, right = cv2.imread(left_imgs[i], cv2.IMREAD_GRAYSCALE), cv2.imread(right_imgs[i], cv2.IMREAD_GRAYSCALE)
#     # Global Block Matching
#     stereo_bm = cv2.StereoBM_create(numDisparities=16, blockSize=15)
#     dispmap_bm = stereo_bm.compute(left, right)
#     # Semi-Global Block Matching
#     stereo_sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=16, blockSize=15)
#     dispmap_sgbm = stereo_sgbm.compute(left, right)

#     # output
#     cv2.imwrite(osp.join(bm_dir, gt_depth[i][:-7]+"bm.jpg"), dispmap_bm)
#     cv2.imwrite(osp.join(sgbm_dir, gt_depth[i][:-7]+"sgbm.jpg"), dispmap_sgbm)

# gt = cv2.imread("gt.png", cv2.IMREAD_GRAYSCALE)

left, right = cv2.imread("data/mini_test/001751_left.jpg", cv2.IMREAD_GRAYSCALE), cv2.imread("data/mini_test/001751_right.jpg", cv2.IMREAD_GRAYSCALE)

# Global Block Matching
stereo_bm = cv2.StereoBM_create(numDisparities=32, blockSize=7)
dispmap_bm = stereo_bm.compute(left, right)
# Semi-Global Block Matching
stereo_sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=32, blockSize=7)
dispmap_sgbm = stereo_sgbm.compute(left, right)

print("left/right img size: ", left.shape)
print("bm/sgbm img size: ", dispmap_sgbm.shape)
# output
cv2.imwrite("data/mini_test/001751_bm.jpg", dispmap_bm)
cv2.imwrite("data/mini_test/001751_sgbm.jpg", dispmap_sgbm)
