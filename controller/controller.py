#!/usr/bin/env python3
"""
Controller publisher for Unitree robots.
Publishes joystick/gamepad input to a topic that can be subscribed to by other nodes.
"""

import time
import sys
from dataclasses import dataclass
from cyclonedds.idl import IdlStruct

# Add SDK to path if needed
sys.path.append('/home/techlab/dev/robotics/unitree/unitree_sdk2_python')

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.utils.joystick import Joystick

# Define the controller message structure
@dataclass
class ControllerMsg(IdlStruct, typename="ControllerMsg"):
    # Timestamp
    timestamp: float
    
    # Axis values (analog sticks)
    lx: float  # Left stick X axis
    ly: float  # Left stick Y axis
    rx: float  # Right stick X axis
    ry: float  # Right stick Y axis
    
    # Button states (0 or 1)
    A: int
    B: int
    X: int
    Y: int
    LB: int
    RB: int
    LT: int
    RT: int
    back: int
    start: int
    up: int
    down: int
    left: int
    right: int


class ControllerPublisher:
    def __init__(self, topic_name="controller_input", publish_rate=50):
        """
        Initialize controller publisher.
        
        Args:
            topic_name: Name of the topic to publish to
            publish_rate: Publishing frequency in Hz
        """
        self.topic_name = topic_name
        self.publish_rate = publish_rate
        self.period = 1.0 / publish_rate
        
        # Initialize joystick
        self.joystick = Joystick()
        
        # Initialize channel factory
        ChannelFactoryInitialize()
        
        # Create publisher
        self.pub = ChannelPublisher(self.topic_name, ControllerMsg)
        self.pub.Init()
        
        print(f"Controller publisher initialized on topic: {self.topic_name}")
        print(f"Publishing at {self.publish_rate} Hz")
        
    def create_message(self):
        """Create a ControllerMsg from current joystick state."""
        msg = ControllerMsg(
            timestamp=time.time(),
            # Axes
            lx=self.joystick.lx.data,
            ly=self.joystick.ly.data,
            rx=self.joystick.rx.data,
            ry=self.joystick.ry.data,
            # Buttons
            A=1 if self.joystick.A.pressed else 0,
            B=1 if self.joystick.B.pressed else 0,
            X=1 if self.joystick.X.pressed else 0,
            Y=1 if self.joystick.Y.pressed else 0,
            LB=1 if self.joystick.LB.pressed else 0,
            RB=1 if self.joystick.RB.pressed else 0,
            LT=1 if self.joystick.LT.pressed else 0,
            RT=1 if self.joystick.RT.pressed else 0,
            back=1 if self.joystick.back.pressed else 0,
            start=1 if self.joystick.start.pressed else 0,
            up=1 if self.joystick.up.pressed else 0,
            down=1 if self.joystick.down.pressed else 0,
            left=1 if self.joystick.left.pressed else 0,
            right=1 if self.joystick.right.pressed else 0
        )
        return msg
    
    def print_status(self, msg):
        """Print controller status to terminal."""
        print(f"\rLX:{msg.lx:6.3f} LY:{msg.ly:6.3f} RX:{msg.rx:6.3f} RY:{msg.ry:6.3f} | "
              f"A:{msg.A} B:{msg.B} X:{msg.X} Y:{msg.Y} | "
              f"LB:{msg.LB} RB:{msg.RB} LT:{msg.LT} RT:{msg.RT}", end="")
    
    def run(self, verbose=True):
        """
        Main loop to read joystick and publish data.
        
        Args:
            verbose: Print controller values to terminal
        """
        print("Controller publisher running... (Press Ctrl+C to stop)")
        
        try:
            import pygame
            pygame.init()
            
            # Wait for controller connection
            while pygame.joystick.get_count() == 0:
                print("Waiting for controller connection...")
                time.sleep(1)
                pygame.joystick.quit()
                pygame.joystick.init()
            
            # Initialize the first joystick
            gamepad = pygame.joystick.Joystick(0)
            gamepad.init()
            print(f"Connected to: {gamepad.get_name()}")
            
            last_publish_time = time.time()
            
            while True:
                # Process pygame events to update joystick state
                pygame.event.pump()
                
                # Map pygame joystick to our Joystick class
                # Axes
                if gamepad.get_numaxes() >= 4:
                    self.joystick.lx(gamepad.get_axis(0))  # Left stick X
                    self.joystick.ly(-gamepad.get_axis(1))  # Left stick Y (inverted)
                    self.joystick.rx(gamepad.get_axis(2))  # Right stick X
                    self.joystick.ry(-gamepad.get_axis(3))  # Right stick Y (inverted)
                
                # Buttons (mapping may vary by controller)
                if gamepad.get_numbuttons() >= 10:
                    self.joystick.A(gamepad.get_button(0))
                    self.joystick.B(gamepad.get_button(1))
                    self.joystick.X(gamepad.get_button(2))
                    self.joystick.Y(gamepad.get_button(3))
                    self.joystick.LB(gamepad.get_button(4))
                    self.joystick.RB(gamepad.get_button(5))
                    self.joystick.back(gamepad.get_button(6))
                    self.joystick.start(gamepad.get_button(7))
                    
                # D-pad (hat)
                if gamepad.get_numhats() >= 1:
                    hat = gamepad.get_hat(0)
                    self.joystick.left(1 if hat[0] < 0 else 0)
                    self.joystick.right(1 if hat[0] > 0 else 0)
                    self.joystick.down(1 if hat[1] < 0 else 0)
                    self.joystick.up(1 if hat[1] > 0 else 0)
                
                # Triggers (if available as axes)
                if gamepad.get_numaxes() >= 6:
                    lt_value = (gamepad.get_axis(4) + 1) / 2  # Convert from [-1,1] to [0,1]
                    rt_value = (gamepad.get_axis(5) + 1) / 2
                    self.joystick.LT(1 if lt_value > 0.5 else 0)
                    self.joystick.RT(1 if rt_value > 0.5 else 0)
                
                # Publish at fixed rate
                current_time = time.time()
                if current_time - last_publish_time >= self.period:
                    msg = self.create_message()
                    
                    if self.pub.Write(msg):
                        if verbose:
                            self.print_status(msg)
                    else:
                        if verbose:
                            print("\rWaiting for subscribers...", end="")
                    
                    last_publish_time = current_time
                
                # Small sleep to prevent CPU overload
                time.sleep(0.001)
                
        except KeyboardInterrupt:
            print("\n\nShutting down controller publisher...")
        finally:
            self.pub.Close()
            pygame.quit()
            print("Controller publisher stopped.")


def main():
    """Main entry point."""
    # Parse command line arguments
    topic_name = "controller_input"
    rate = 50
    
    if len(sys.argv) > 1:
        topic_name = sys.argv[1]
    if len(sys.argv) > 2:
        rate = int(sys.argv[2])
    
    # Create and run publisher
    controller = ControllerPublisher(topic_name=topic_name, publish_rate=rate)
    controller.run(verbose=False)


if __name__ == "__main__":
    main()
