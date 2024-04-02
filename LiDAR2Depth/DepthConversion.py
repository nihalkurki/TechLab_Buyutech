import numpy as np
import plyfile
import matplotlib.pyplot as plt
import cv2


# Load .ply file
f_name = "005323.ply"
plydata = plyfile.PlyData.read(f_name)

# Extract vertex data
vertices = plydata['vertex']

# Extract x, y, z coordinates from vertices
z = vertices['x']
x = vertices['y']
y = vertices['z']
# x = vertices['x']
# y = vertices['y']
# z = vertices['z']

# remove everything on the back of the camera
z[z<0] = 0
x = -x
y = -y

# Calculate the resolution of the depth map
min_x = np.min(x)
max_x = np.max(x)
min_y = np.min(y)
max_y = np.max(y)
min_z = np.min(z)
max_z = np.max(z)
resolution = 0.1 # adjust this value to change the resolution of the depth map

# Calculate the width and height of the depth map
width = int(np.ceil((max_x - min_x) / resolution))
height = int(np.ceil((max_y - min_y) / resolution))

# Create an empty depth map
depth_map = np.zeros((height, width), dtype=np.float32)

# Convert the point cloud to the depth map
for i in range(len(x)):
    col = int(np.round((x[i] - min_x) / resolution))
    row = int(np.round((y[i] - min_y) / resolution))
    if col >= 0 and col < width and row >= 0 and row < height:
        depth = z[i] - min_z
        if depth_map[row, col] == 0 or depth < depth_map[row, col]:
            depth_map[row, col] = depth

# Save the depth map as an image
depth_map = (depth_map / np.max(depth_map)) * 255
depth_map = np.uint8(depth_map)
cv2.imwrite("depth_map.png", depth_map)
