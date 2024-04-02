import cv2
import numpy as np
import matplotlib.pyplot as plt

point_cloud = cv2.imread('data/mini_test/2018-07-09-16-11-56_2018-07-09-16-11-56-502.png')

print(point_cloud)

img_greyscale = cv2.imread("data/mini_test/001751_sgbm.jpg", cv2.IMREAD_GRAYSCALE)
# Normalize the image to range [0, 1] if it's not already

print(img_greyscale)
print(np.min(img_greyscale), np.max(img_greyscale))

img_normalized = img_greyscale.astype('float32') / 255.0



# Apply a colormap (e.g., 'jet') using Matplotlib
img_rgb = plt.cm.jet(img_normalized)[:, :, :3]  # Ignore alpha channel
img_rgb = 255-(img_rgb * 255).astype('uint8')
img_rgb = cv2.applyColorMap(img_rgb, cv2.COLORMAP_MAGMA)
# Save or display the image as needed
cv2.imwrite('data/mini_test/001751_sgbm_magma.png', img_rgb)
cv2.imshow('RGB Image', img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()