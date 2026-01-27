#!/usr/bin/env python3
"""
Utility functions for processing LiDAR data and creating height maps.
"""

import numpy as np
from unitree_sdk2_python.unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
import struct
import sys
np.set_printoptions(precision=2, threshold=sys.maxsize, linewidth=2000)


def pointcloud_to_heightmap(lidar_msg, grid_size=80, map_range=4.0):
    """
    Convert PointCloud2 message to a height map grid.
    
    Args:
        lidar_msg: PointCloud2_ message from robot
        grid_size: Size of the grid (default 80x80)
        map_range: Total range in meters (default 4m = ±2m)
    
    Returns:
        height_map: 2D numpy array of shape (grid_size, grid_size) with heights
        info: Dictionary with metadata about the height map
    """
    resolution = map_range / grid_size
    
    # Initialize outputs
    height_map = np.zeros((grid_size, grid_size), dtype=np.float32)
    min_height_map = np.full((grid_size, grid_size), np.inf, dtype=np.float32)
    max_height_map = np.full((grid_size, grid_size), -np.inf, dtype=np.float32)
    point_count = np.zeros((grid_size, grid_size), dtype=np.int32)
    
    # Parse pointcloud
    num_points = lidar_msg.width * lidar_msg.height
    point_step = lidar_msg.point_step
    
    # Convert to bytes if needed
    if isinstance(lidar_msg.data, list):
        data_bytes = bytes(lidar_msg.data)
    else:
        data_bytes = lidar_msg.data
    
    # Project points onto grid
    for i in range(num_points):
        offset = i * point_step
        x = struct.unpack_from('f', data_bytes, offset)[0]
        y = struct.unpack_from('f', data_bytes, offset + 4)[0]
        z = struct.unpack_from('f', data_bytes, offset + 8)[0]
        
        # Filter invalid points
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
            continue
        
        # Convert to grid coordinates (robot at center)
        grid_x = int((x + map_range/2) / resolution)
        grid_y = int((y + map_range/2) / resolution)
        
        if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
            max_height_map[grid_x, grid_y] = max(max_height_map[grid_x, grid_y], z)
            min_height_map[grid_x, grid_y] = min(min_height_map[grid_x, grid_y], z)
            point_count[grid_x, grid_y] += 1
    
    # Use max height for occupied cells, zero for empty cells
    occupied_mask = point_count > 0
    height_map[occupied_mask] = max_height_map[occupied_mask]
    height_map[~occupied_mask] = 0.0
    
    # Fix inf values in min_height_map
    min_height_map[~occupied_mask] = 0.0
    
    # Calculate terrain metrics
    roughness = max_height_map - min_height_map
    roughness[~occupied_mask] = 0.0
    roughness[roughness == np.inf] = 0.0
    roughness[roughness == -np.inf] = 0.0
    
    info = {
        'grid_size': grid_size,
        'resolution': resolution,
        'map_range': map_range,
        'num_points': num_points,
        'occupied_cells': np.sum(occupied_mask),
        'total_cells': grid_size * grid_size,
        'min_height_map': min_height_map,
        'max_height_map': max_height_map,
        'roughness_map': roughness,
        'point_count': point_count,
    }
    
    return height_map, info

def process_height_map(height_map: np.array, lidar_msg: PointCloud2_, max_dist: float, delete_count: int = 100):
    grid_size = height_map.shape[0]
    map_range = 2.0 * max_dist
    cell_size = map_range / grid_size  # meters per cell
    
    # Parse pointcloud
    num_points = lidar_msg.width * lidar_msg.height
    point_step = lidar_msg.point_step
    data_bytes = bytes(lidar_msg.data)

    # Clear old data (cells not updated in delete_count frames)
    # old_cells = height_map[:, :, 1] > delete_count
    # height_map[old_cells, 0] = 0.0  # Reset height
    # height_map[old_cells, 1] = 0    # Reset age
    
    # Increment age for all cells (vectorized)
    height_map[:, :, 1] += 1
    
    # Project points onto grid
    for i in range(num_points):
        offset = i * point_step
        x = struct.unpack_from('f', data_bytes, offset)[0]
        y = struct.unpack_from('f', data_bytes, offset + 4)[0]
        z = struct.unpack_from('f', data_bytes, offset + 8)[0]
        
        # Filter invalid points
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
            continue
        
        # Convert to grid coordinates (robot at center)
        grid_x = int((x + max_dist) / cell_size)
        grid_y = int((y + max_dist) / cell_size)
        
        if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
            height_map[grid_x, grid_y, 0] = z
            height_map[grid_x, grid_y, 1] = 0
    
    
    

def visualize_heightmap(height_map, info, show_full_stats=True):
    """
    Print a visualization of the height map.
    
    Args:
        height_map: 2D numpy array with heights
        info: Dictionary with metadata
    """
    grid_size = info['grid_size']
    resolution = info['resolution']
    point_count = info['point_count']
    
    print("\n" + "="*60)
    print("HEIGHT MAP INFO:")
    print("="*60)
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Map range: ±{info['map_range']/2:.1f}m")
    print(f"Resolution: {resolution*100:.1f} cm/cell")
    print(f"Points processed: {info['num_points']}")
    print(f"Occupied cells: {info['occupied_cells']}/{info['total_cells']} "
          f"({100*info['occupied_cells']/info['total_cells']:.1f}%)")
    occupied_mask = point_count > 0
    if np.any(occupied_mask):
        valid_heights = height_map[occupied_mask]
        print(f"Height range: {np.min(valid_heights):.3f}m to {np.max(valid_heights):.3f}m")
        print(f"Mean height: {np.mean(valid_heights):.3f}m ± {np.std(valid_heights):.3f}m")
        
        if show_full_stats:
            roughness = info['roughness_map'][occupied_mask]
            print(f"Terrain roughness: mean={np.mean(roughness):.3f}m, max={np.max(roughness):.3f}m")
    
    # Center slices
    center_row = grid_size // 2
    center_col = grid_size // 2
    
    print(f"\n--- Forward view (x-axis, col {center_col}) ---")
    sample_rows = np.linspace(0, grid_size-1, min(10, grid_size), dtype=int)
    for row in sample_rows:
        h = height_map[row, center_col]
        cnt = point_count[row, center_col]
        x_pos = (row - grid_size/2) * resolution
        status = f"{int(h*100):3d}cm" if cnt > 0 else " empty"
        print(f"  x={x_pos:+.2f}m: {status} ({cnt:2d} pts)")
    
    # ASCII top-down view
    print(f"\n--- Top-down view (heights in cm) ---")
    print("       Front")
    row_step = max(1, grid_size // 16)
    col_step = max(1, grid_size // 16)
    
    for row in range(grid_size-1, -1, -row_step):
        label = "F" if row == grid_size-1 else ("M" if row == center_row else ("B" if row == 0 else " "))
        print(f"{row:2d}{label}| ", end="")
        
        for col in range(0, grid_size, col_step):
            cnt = point_count[row, col]
            if cnt > 0:
                h_cm = int(height_map[row, col] * 100)
                if h_cm >= 40:
                    print(f"\033[91m{h_cm:2d}\033[0m", end=" ")  # Red for high
                elif h_cm >= 15:
                    print(f"\033[93m{h_cm:2d}\033[0m", end=" ")  # Yellow for medium
                elif h_cm >= 5:
                    print(f"{h_cm:2d}", end=" ")  # Normal for low
                else:
                    print(" .", end=" ")  # Dot for ground
            else:
                print("--", end=" ")
        print()
    
    print("       Back")
    print("   (F=Front, M=Middle/Robot, B=Back)")
    print("="*60 + "\n")


def visualize_obstacle_map(height_map, info, obstacle_threshold=0.15, display_range=2.0):
    """
    Display a simple binary obstacle map (X = obstacle, O = clear).
    
    Args:
        height_map: 2D numpy array with heights
        info: Dictionary with metadata
        obstacle_threshold: Height threshold in meters to consider as obstacle (default 0.15m = 15cm)
        display_range: Display range in meters (default ±2.0m)
    """
    grid_size = info['grid_size']
    resolution = info['resolution']
    point_count = info['point_count']
    
    # Calculate display bounds
    center = grid_size // 2
    cells_per_meter = int(1.0 / resolution)
    display_cells = int(display_range / resolution)
    
    start_idx = max(0, center - display_cells)
    end_idx = min(grid_size, center + display_cells)
    
    print("\n" + "="*60)
    print(f"OBSTACLE MAP (threshold={obstacle_threshold*100:.0f}cm, range=±{display_range}m)")
    print("="*60)
    print("X = Obstacle detected")
    print("O = Clear / Ground")
    print("- = No data")
    print()
    
    # Print column header (Y-axis labels)
    print("     ", end="")
    for col in range(start_idx, end_idx, max(1, (end_idx-start_idx)//20)):
        y_pos = (col - center) * resolution
        print(f"{y_pos:+.1f}".center(2), end=" ")
    print()
    print("     " + "-" * ((end_idx - start_idx) // max(1, (end_idx-start_idx)//40) * 3))
    
    # Print rows (X-axis from front to back)
    for row in range(end_idx-1, start_idx-1, -1):
        x_pos = (row - center) * resolution
        
        # Row label
        if abs(x_pos) < 0.05:
            print(f"{x_pos:+.1f}R|", end=" ")  # R for robot position
        else:
            print(f"{x_pos:+.1f} |", end=" ")
        
        # Print obstacle indicators
        for col in range(start_idx, end_idx):
            cnt = point_count[row, col]
            h = height_map[row, col]
            
            if cnt == 0:
                print("-", end=" ")  # No data
            elif h >= obstacle_threshold:
                print("X", end=" ")  # Obstacle
            else:
                print("O", end=" ")  # Clear
        
        print()
    
    print("     " + "-" * ((end_idx - start_idx) // max(1, (end_idx-start_idx)//40) * 3))
    print(f"     Left <------ Y-axis ------> Right")
    print(f"     (X-axis: Front at top, Back at bottom)")
    print("="*60 + "\n")

