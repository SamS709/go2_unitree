#!/usr/bin/env python3
"""
Utility functions for processing LiDAR data and creating height maps.
"""

import numpy as np
from unitree_sdk2_python.unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
import struct
import sys
np.set_printoptions(precision=2, threshold=sys.maxsize, linewidth=np.inf, edgeitems=100, suppress=True)

# Lidar inclination correction (15 degrees)
LIDAR_PITCH_DEG = -15.09
LIDAR_PITCH_RAD = np.deg2rad(LIDAR_PITCH_DEG)
COS_PITCH = np.cos(LIDAR_PITCH_RAD)
SIN_PITCH = np.sin(LIDAR_PITCH_RAD)

def process_height_map(height_map: np.array, lidar_msg: PointCloud2_, height_map_dims: list, delete_count: int = 100, min_x = 0, max_x = 0, min_z = 0, max_z = 0):
    grid_size_x = height_map.shape[0]
    grid_size_y = height_map.shape[1]
    map_range_x = 2.0 * height_map_dims[0]
    map_range_y = 2.0 * height_map_dims[1]
    cell_size_x = map_range_x / grid_size_x  # meters per cell
    cell_size_y = map_range_y / grid_size_y  # meters per cell
    # Parse pointcloud
    num_points = lidar_msg.width * lidar_msg.height
    point_step = lidar_msg.point_step
    data_bytes = bytes(lidar_msg.data)

    # Clear old data (cells not updated in delete_count frames)
    old_cells = height_map[:, :, 1] > delete_count
    height_map[old_cells, 0] = 0.0 # Reset height
    height_map[old_cells, 1] = 0    # Reset age
    
    # Increment age for all cells (vectorized)
    height_map[:, :, 1] += 1
    x_max = 0.0
    x_min = 0.0
    z_max = 0.0
    z_min = 0.0
    max_x_z = 0.0
    min_x_z = 0.0
    
    # Project points onto grid
    for i in range(num_points):
        offset = i * point_step
        x = struct.unpack_from('f', data_bytes, offset)[0]
        y = struct.unpack_from('f', data_bytes, offset + 4)[0]
        z = struct.unpack_from('f', data_bytes, offset + 8)[0]
        
        # Filter invalid points
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
            continue
        
        # Correct for lidar pitch (rotation around Y-axis)
        x = -x
        y = -y
        x_corrected = x * COS_PITCH - z * SIN_PITCH
        z_corrected = x * SIN_PITCH + z * COS_PITCH
        x, y, z = x_corrected, y, z_corrected
        
        # Convert to grid coordinates (robot at center)
        # Flip x-axis: high x → row 0 (top), low x → row grid_size-1 (bottom)
        grid_x = grid_size_x - 1 - int((x + height_map_dims[0]) / cell_size_x)
        grid_y = int((y + height_map_dims[1]) / cell_size_y)
        if x > x_max:
            x_max = x
            max_x_z = z
        elif x < x_min:
            x_min = x
            min_x_z = z
        if z > z_max:
            z_max = z
        elif z < z_min:
            z_min = z
        if 0 <= grid_x < grid_size_x and 0 <= grid_y < grid_size_y:
            if z > height_map[grid_x, grid_y, 0]:  # Update if this point is higher
                height_map[grid_x, grid_y, 0] = z
                height_map[grid_x, grid_y, 1] = 0  # Reset age for updated cell
    np.set_printoptions(precision=2, threshold=sys.maxsize, linewidth=np.inf, edgeitems=100, suppress=True)
    print(height_map[:, :, 0])
    return x_max, x_min, z_max, z_min, max_x_z, min_x_z
    
    
    