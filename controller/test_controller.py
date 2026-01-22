#!/usr/bin/env python3
"""
Test subscriber for controller data.
Subscribes to controller_input topic and displays received data.
"""

import time
import sys
import os

# Add SDK and parent directory to path
sys.path.append('/home/techlab/dev/robotics/unitree/unitree_sdk2_python')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from controller import ControllerMsg


class ControllerSubscriber:
    def __init__(self, topic_name="controller_input"):
        """
        Initialize controller subscriber.
        
        Args:
            topic_name: Name of the topic to subscribe to
        """
        self.start_time = time.perf_counter()
        self.topic_name = topic_name
        self.last_msg = None
        self.msg_count = 0
        self.last_print_time = time.time()
        
        # Initialize channel factory
        ChannelFactoryInitialize()
        
        # Create subscriber
        self.sub = ChannelSubscriber(self.topic_name, ControllerMsg)
        self.sub.Init(self.callback, 10)
        
        print(f"Subscribed to topic: {self.topic_name}")
        print("Waiting for controller data...\n")
    
    def callback(self, msg: ControllerMsg):
        """Callback function called when new controller data is received."""
        self.last_msg = msg
        self.msg_count += 1
    
    def print_detailed(self, msg):
        """Print detailed controller state."""
        print("\n" + "="*60)
        print(f"Controller State (Message #{self.msg_count})")
        print("="*60)
        print(f"Timestamp: {msg.timestamp:.3f}")
        print("\nAnalog Sticks:")
        print(f"  Left Stick  - X: {msg.lx:7.3f}  Y: {msg.ly:7.3f}")
        print(f"  Right Stick - X: {msg.rx:7.3f}  Y: {msg.ry:7.3f}")
        print("\nFace Buttons:")
        print(f"  A: {msg.A}  B: {msg.B}  X: {msg.X}  Y: {msg.Y}")
        print("\nShoulder Buttons:")
        print(f"  LB: {msg.LB}  RB: {msg.RB}  LT: {msg.LT}  RT: {msg.RT}")
        print("\nD-Pad:")
        print(f"  Up: {msg.up}  Down: {msg.down}  Left: {msg.left}  Right: {msg.right}")
        print("\nSystem Buttons:")
        print(f"  Back: {msg.back}  Start: {msg.start}")
        # print(self.start_time - time.perf_counter())
        print("="*60)
    
    def print_compact(self, msg):
        """Print compact one-line status."""
        # Axes
        axes = f"LX:{msg.lx:6.2f} LY:{msg.ly:6.2f} RX:{msg.rx:6.2f} RY:{msg.ry:6.2f}"
        
        # Buttons (only show pressed ones)
        pressed = []
        if msg.A: pressed.append("A")
        if msg.B: pressed.append("B")
        if msg.X: pressed.append("X")
        if msg.Y: pressed.append("Y")
        if msg.LB: pressed.append("LB")
        if msg.RB: pressed.append("RB")
        if msg.LT: pressed.append("LT")
        if msg.RT: pressed.append("RT")
        if msg.up: pressed.append("UP")
        if msg.down: pressed.append("DN")
        if msg.left: pressed.append("LF")
        if msg.right: pressed.append("RG")
        if msg.back: pressed.append("BK")
        if msg.start: pressed.append("ST")
        
        buttons = f"[{', '.join(pressed)}]" if pressed else "[---]"
        
        print(f"\r{axes} | {buttons:30s} | Msgs: {self.msg_count}", end="", flush=True)
    
    def run(self, mode="detailed", print_rate=10):
        """
        Main loop to display received controller data.
        
        Args:
            mode: Display mode - "compact" for single line, "detailed" for full info
            print_rate: How often to print (Hz) in compact mode
        """
        print(f"Display mode: {mode}")
        print("Press Ctrl+C to stop\n")
        
        try:
            if mode == "detailed":
                # Detailed mode: print full info when data changes
                while True:
                    if self.last_msg is not None:
                        self.print_detailed(self.last_msg)
                        self.last_msg = None  # Clear to wait for next
                    time.sleep(0.1)
            
            else:
                # Compact mode: continuous single-line updates
                print_period = 1.0 / print_rate
                
                while True:
                    current_time = time.time()
                    
                    if self.last_msg is not None and (current_time - self.last_print_time >= print_period):
                        self.print_compact(self.last_msg)
                        self.last_print_time = current_time
                    
                    time.sleep(0.01)
        
        except KeyboardInterrupt:
            print("\n\nShutting down subscriber...")
        finally:
            print(f"\nTotal messages received: {self.msg_count}")


def main():
    """Main entry point."""
    # Parse command line arguments
    topic_name = "controller_input"
    mode = "detailed"
    
    if len(sys.argv) > 1:
        topic_name = sys.argv[1]
    if len(sys.argv) > 2:
        mode = sys.argv[2]
    
    # Create and run subscriber
    subscriber = ControllerSubscriber(topic_name=topic_name)
    subscriber.run(mode=mode, print_rate=10)


if __name__ == "__main__":
    print("="*60)
    print("Controller Data Subscriber Test")
    print("="*60)
    print("\nUsage: python test_controller.py [topic_name] [mode]")
    print("  topic_name: Topic to subscribe to (default: controller_input)")
    print("  mode: 'compact' or 'detailed' (default: compact)")
    print()
    
    main()
