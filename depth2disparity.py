import cv2
import numpy as np
import matplotlib.pyplot as plt

def inverse_log_depth(log_depth_image):
    # Retrieve the depth from its logarithmic form
    depth_image = np.exp(log_depth_image / 1.0) - 1
    return depth_image

def depth_to_disparity(normalized_depth_map, B, f):
    # inverse log_depth into actual depth
    depth_map = inverse_log_depth(normalized_depth_map)
    # Handle division by zero
    eps = 1e-6
    depth_map[depth_map == 0] = eps

    disparity_map = (B * f) / (depth_map)
    
    return disparity_map



depth_map = cv2.imread('/Users/evanwyf/Desktop/techlab/data/mini_test/001751_dep.jpg')[:,:,0]
normalized_depth_map = depth_map / 255.0

B = 0.12         # Set the baseline distance in  meter !!!
f = 184.752086  # Set the focal length pixel !!!

disparity_map = depth_to_disparity(normalized_depth_map, B, f)
cv2.imwrite('/Users/evanwyf/Desktop/techlab/data/mini_test/001751_disp_converted.png', disparity_map)


