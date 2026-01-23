#!/usr/bin/env python3
"""
Quick test to check if lidar channel is publishing data
"""
import sys
import time
sys.path.insert(0, '../unitree_sdk2_python')

from unitree_sdk2_python.unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2_python.unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
from unitree_sdk2_python.unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2_python.unitree_sdk2py.idl.default import std_msgs_msg_dds__String_
from unitree_sdk2_python.unitree_sdk2py.core.channel import ChannelPublisher

class ChannelChecker:
    def __init__(self):
        self.received_lidar = False
        self.message_count = 0
        
    def lidar_callback(self, msg: PointCloud2_):
        self.received_lidar = True
        self.message_count += 1
        print(f"✓ Lidar message #{self.message_count} received:")
        print(f"  - Dimensions: {msg.width}x{msg.height} points")
        print(f"  - Data size: {len(msg.data)} bytes")
        print(f"  - Point step: {msg.point_step} bytes")
        print(f"  - Row step: {msg.row_step} bytes")
        print()

if __name__ == '__main__':
    print("Checking lidar channel availability...")
    print("-" * 60)
    
    # Initialize SDK
    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)
    
    checker = ChannelChecker()
    
    # Try to enable lidar first
    print("1. Sending lidar enable command...")
    lidar_switch_pub = ChannelPublisher("rt/utlidar/switch", String_)
    lidar_switch_pub.Init()
    
    lidar_cmd = std_msgs_msg_dds__String_()
    lidar_cmd.data = "ON"
    lidar_switch_pub.Write(lidar_cmd)
    print("   Enable command sent\n")
    
    # Subscribe to lidar channel
    print("2. Subscribing to 'rt/utlidar/cloud' channel...")
    lidar_sub = ChannelSubscriber("rt/utlidar/cloud", PointCloud2_)
    lidar_sub.Init(checker.lidar_callback, 10)
    print("   Subscription active\n")
    
    # Wait for messages
    print("3. Waiting for lidar data (15 second timeout)...")
    timeout = 15.0
    start = time.time()
    
    while (time.time() - start) < timeout:
        if checker.received_lidar:
            print(f"\n{'='*60}")
            print(f"SUCCESS! Lidar is publishing on 'rt/utlidar/cloud'")
            print(f"Received {checker.message_count} messages in {time.time() - start:.1f} seconds")
            print(f"{'='*60}\n")
            
            # Keep running for a few more seconds to see message rate
            print("Monitoring for 5 more seconds to check message rate...")
            time.sleep(5)
            print(f"Total messages: {checker.message_count}")
            print(f"Message rate: ~{checker.message_count/5:.1f} Hz")
            break
        
        elapsed = time.time() - start
        print(f"   Waiting... ({elapsed:.1f}s elapsed)", end='\r')
        time.sleep(0.1)
    
    if not checker.received_lidar:
        print(f"\n\n{'='*60}")
        print("⚠ WARNING: No lidar data received after 15 seconds")
        print("{'='*60}")
        print("\nPossible reasons:")
        print("  1. Robot doesn't have utlidar sensor installed")
        print("  2. Lidar sensor is disabled or not responding")
        print("  3. Wrong channel name (try checking DDS topics)")
        print("  4. Permissions issue accessing lidar data")
        print(f"{'='*60}\n")
