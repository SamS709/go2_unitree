#!/usr/bin/env python3

"""
RL Policy Controller for Unitree Go2 Robot
Loads a PyTorch policy and controls the robot at 50Hz
"""

"""
TO RUN:

ros2 launch huro go2_rviz.launch.py
ros2 run huro spacemouse_publisher.py
ros2 run huro sim_go2
ros2 run huro go2_publisher.py --use_spacemouse True

"""
import sys
import os
# Add parent directory to path to import controller and unitree_legged_const
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add grandparent directory to path to import unitree_sdk2_python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from controller.controller import ControllerMsg
import torch
import os
from utils import Mapper
from lidar_utils import process_height_map
from get_obs import get_obs, get_obs_lidar
import time
import sys
from unitree_sdk2_python.unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2_python.unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2_python.unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
from unitree_sdk2_python.unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2_python.unitree_sdk2py.idl.default import std_msgs_msg_dds__String_

from unitree_sdk2_python.unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_, LidarState_
from unitree_sdk2_python.unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2_python.unitree_sdk2py.utils.crc import CRC
from unitree_sdk2_python.unitree_sdk2py.utils.thread import RecurrentThread
import unitree_legged_const as go2
from unitree_sdk2_python.unitree_sdk2py.go2.robot_state.robot_state_client import RobotStateClient



"""
RUN:
python go2_controller.py
"""

class Go2PolicyController:
    """RL Policy controller for Unitree Go2 locomotion."""

    def __init__(
        self,
        newton = True,
        lidar = False
        ):
        """
        Initialize the policy controller.

        Args:
            policy_path: Path to the policy.pt file
            policy_freq: Policy inference frequency in Hz (default: 50Hz)
            control_freq: Motor command frequency in Hz (default: 500Hz, simulation dt=0.002)
            kp: Position gain/stiffness (default: 25.0) - MUST match training value!
            kd: Velocity gain/damping (default: 0.5)
            action_scale: Scale factor for policy actions (default: 0.25)
        """

        self.low_cmd = unitree_go_msg_dds__LowCmd_()  

        self.step_dt = 1 / 50  # policy freq = 50Hz
        self.run_policy = False # set to false to rely on joy buttons to lauch the policy
        self.lidar_obs = lidar

    
        # Emergency mode
        self.emergency_mode = False
        self.emergency_mode_start_time = None
        self.last_commanded_positions = None
        self.stand_down = False

        if newton:
            policy_name = "policy_newton.pt"
        elif lidar:
            policy_name = "policy_lidar3.pt"
        else:
            policy_name = "policy_asymmetric.pt"

        policy_path = os.path.join("resources", "models", policy_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[INFO] Loading policy from: {policy_name}")
        print(f"[INFO] Using device: {self.device}")

        if not os.path.exists(policy_path):
            raise FileNotFoundError(f"Policy file not found: {policy_path}")
        self.policy = torch.jit.load(policy_path, map_location=self.device)
        self.policy.eval()
        print("[INFO] Policy loaded successfully")

        # Initialize the mapper for the joints and the actions

        mapping_file = "newton_to_unitree.yaml" if newton else "isaaclab_to_unitree.yaml"
        mapping_path = os.path.join("resources", "mappings", mapping_file)


 
        self.default_pos_sdk = torch.tensor(
            [
                -0.1,
                0.8,
                -1.5,  # FR: hip, thigh, calf (actuators 0-2)
                0.1,
                0.8,
                -1.5,  # FL: hip, thigh, calf (actuators 3-5)
                -0.1,
                1.0,
                -1.5,  # RR: hip, thigh, calf (actuators 6-8)
                0.1,
                1.0,
                -1.5,  # RL: hip, thigh, calf (actuators 9-11)
            ]
        )
        self.stand_down_pos = torch.tensor(
            [-0.35, 1.36, -2.65, 0.35, 1.36, -2.65,
                             -0.5, 1.36, -2.65, 0.5, 1.36, -2.65]
        )
        self.stand_up_pos = self.default_pos_sdk
      
        self.mapper = Mapper(
            mapping_yaml_path=mapping_path, default_pos_sdk=self.default_pos_sdk
        )

        # Store latest action (for use between policy updates)
        self.current_action = torch.zeros(12)

        # Store latest messages
        self.latest_low_state = None
        self.controller_state = None

        self.kp = 60.0  # Position gain
        self.kd = 5.0  # Velocity gain
        self.kp_p = 25.0  # Position gain
        self.kd_p = 0.5 # Velocity gain
        self.action_scale = 0.25  # Scale policy output

        # Standing position (default joint positions but coud be different)
        
        self.time_to_stand = 5.0  # Time to reach the standing position
        self.curr_stand_down_time = 0.0

        # Statistics - initialize BEFORE callbacks
        self.tick_count = 0

        # thread handling
        self.lowCmdWriteThreadPtr = None
        self.height_map_dims = [1.0, 0.5] 
        self.max_height_map_dist = 1.0  # ±2m range
        self.height_map_res = 6.0  # cells per meter
        self.height_map = t = torch.zeros((2, 5, 3), dtype = torch.float32)
        self.min_x = 100.0
        self.max_x = 0.0
        self.min_z = 100.0
        self.max_z = 0.0
        self.max_x_z = 0.0
        self.min_x_z = 0.0
        # self.height_map[:, :, 0] = 1.0  # Initialize height with high value (1m obstacles)
        # height_map[:,:,0] = height (m), height_map[:,:,1] = age (frames since last update)

        # self.height_map[i, j, 0] is the height of the highest point located at:
        # - abscissa x in [i * res - max_dist, (i + 1) * res - max_dist] 
        # - ordinate y in [j * res - max_dist, (j + 1) * res - max_dist] 
        # self.height_map[i, j, 1] counts how many samples occured since the point has been sampled
              

        self.crc = CRC()

    def Init(self):
        self.InitLowCmd()


        # create publisher #
        self.low_cmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.low_cmd_pub.Init()

        # create lidar switch publisher
        self.lidar_switch_pub = ChannelPublisher("rt/utlidar/switch", String_)
        self.lidar_switch_pub.Init()

        self.joy_sub = ChannelSubscriber("controller_input", ControllerMsg)
        self.joy_sub.Init(self.joy_callback, 10)

        self.lidar_sub = ChannelSubscriber("rt/utlidar/cloud", PointCloud2_)
        self.lidar_sub.Init(self.lidar_callback, 10)

        # create subscriber 
        self.low_state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.low_state_sub.Init(self.low_state_callback, 10)
        self.InitLowCmd()

        
        robot_state = RobotStateClient()
        robot_state.Init()

        # Get list of running services
        code, services = robot_state.ServiceList()

        # Stop the sport mode service (default policy)
        code = robot_state.ServiceSwitch("sport_mode", False)
        while code != 0:
            print(f"Error switching service: {code}")
            code = robot_state.ServiceSwitch("sport_mode", False)
            time.sleep(1)
        if code == 0:
            print("Sport mode disabled - low-level control available")

        # Enable the lidar
        self.enable_lidar()

    def enable_lidar(self):
        """Enable the utlidar sensor."""
        lidar_msg = std_msgs_msg_dds__String_()
        lidar_msg.data = "ON"
        self.lidar_switch_pub.Write(lidar_msg)
        print("Lidar enabled")
        time.sleep(0.5)  # Give lidar time to start up


    def InitLowCmd(self):
        self.low_cmd.head[0]=0xFE
        self.low_cmd.head[1]=0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0
        for i in range(20):
            self.low_cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
            self.low_cmd.motor_cmd[i].q= go2.PosStopF
            self.low_cmd.motor_cmd[i].kp = 0
            self.low_cmd.motor_cmd[i].dq = go2.VelStopF
            self.low_cmd.motor_cmd[i].kd = 0
            self.low_cmd.motor_cmd[i].tau = 0

    def low_state_callback(self, msg: LowState_):
        """Log low state message."""
        self.latest_low_state = msg


    def joy_callback(self, msg: ControllerMsg):
        """Log joysticks state"""
        self.controller_state = msg

    def lidar_callback(self, msg: PointCloud2_):
        """Process lidar data into heightmap"""
        x_max, x_min, z_max, z_min, max_x_z, min_x_z = process_height_map(self.height_map, msg, self.height_map_dims, delete_count=10, min_x = self.min_x, max_x = self.max_x, min_z = self.min_z, max_z = self.max_z)
        if x_min<self.min_x:
            self.min_x = x_min
            self.min_x_z = min_x_z
        if x_max>self.max_x:
            self.max_x = x_max
            self.max_x_z = max_x_z
        if z_min<self.min_z:
            self.min_z = z_min
        if z_max>self.max_z:
            self.max_z = z_max

    def Start(self):
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval= self.step_dt, target=self.run, name="writebasiccmd"
        )
        self.lowCmdWriteThreadPtr.Start()

    def emergency_mode_control(self):
        """Smoothly reduce gains and torque to zero over release_duration."""
        if self.latest_low_state is None:
            return

        # Calculate progress (0 to 1)
        release_duration = 5.0
        elapsed = (
            time.perf_counter() - self.emergency_mode_start_time
        )
        r = min(elapsed / release_duration, 1.0)

        alpha = 1.0 - (1.0 - r) ** 10

        # Gradually reduce gains from current values to zero
        current_kp = self.kp * (1.0 - alpha)
        current_kd = self.kd * (1.0 - alpha)

        for i in range(12):
            q = self.latest_low_state.motor_state[i].q
            dq = self.latest_low_state.motor_state[i].dq

            # Compute diminishing torque
            tau = current_kp * (self.last_commanded_positions[i] - q) - current_kd * dq

            # self.low_cmd.motor_cmd[i].mode = 0x01
            self.low_cmd.motor_cmd[i].q = self.last_commanded_positions[i]
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kp = current_kp
            self.low_cmd.motor_cmd[i].kd = current_kd
            self.low_cmd.motor_cmd[i].tau = tau

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.low_cmd_pub.Write(self.low_cmd)

    def stand_control(self, dir = "up"):
        """PD control to standing position."""
        if self.latest_low_state is None:
            return
        # cmd.head[0] = 0xFE
        # cmd.head[1] = 0xEF
        # cmd.gpio = 0
        if dir == "up":
            ratio = min((self.curr_time - self.start_time) / self.time_to_stand, 1.0)
            self.last_commanded_positions = [(1.0 - ratio) * self.latest_low_state.motor_state[i].q + ratio * self.stand_up_pos[i] for i in range(12)]
        elif dir == "down":
            ratio = min((self.curr_time - self.curr_stand_down_time) / self.time_to_stand, 1.0)
            self.last_commanded_positions = [(1.0 - ratio) * self.latest_low_state.motor_state[i].q + ratio * self.stand_down_pos[i] for i in range(12)]
            # if ratio >= 1.0:
            #     self.stand_down = False
        for i in range(12):
            # self.low_cmd.motor_cmd[i].mode = 1
            self.low_cmd.motor_cmd[i].q = self.last_commanded_positions[i]
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].tau = 0.0
            self.low_cmd.motor_cmd[i].kp = self.kp
            self.low_cmd.motor_cmd[i].kd = self.kd

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.low_cmd_pub.Write(self.low_cmd)

    def send_motor_commands(self):
        """Send motor commands to the robot based on current action."""
        # Convert current action from policy order to SDK order
        actions_sdk_order = self.mapper.actions_policy_to_sdk(self.current_action)

        # Store last commanded positions for potential emergency mode
        self.last_commanded_positions = (
            self.mapper.default_pos_sdk
        ) + actions_sdk_order * self.action_scale

        # self.low_cmd.head[0] = 0xFE
        # self.low_cmd.head[1] = 0xEF
        # self.low_cmd.level_flag = 0xFF
        # self.low_cmd.gpio = 0
        # target_positions = self.mapper.default_pos_sdk + actions_sdk_order * self.action_scale
        # Set motor commands
        for i in range(12):
            self.low_cmd.motor_cmd[i].mode = 0x01  # PMSM mode
            self.low_cmd.motor_cmd[i].q = self.last_commanded_positions[i]
            self.low_cmd.motor_cmd[i].kp = self.kp_p
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kd = self.kd_p
            self.low_cmd.motor_cmd[i].tau = 0.0

        # Calculate CRC and publish
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.low_cmd_pub.Write(self.low_cmd)

    def run(self):
        """Main control loop running at control_freq Hz."""

        try:
            if self.latest_low_state is not None and self.controller_state is not None:
                if self.tick_count == 0:
                    self.start_time = time.perf_counter()
                self.process_control_step()
            else:
                print("Waiting for robot state...")
                self.start_time = time.perf_counter()

        except KeyboardInterrupt:
            print("\n\n" + "=" * 60)
            print("Shutting down...")
            print(f"Total inferences: {self.tick_count}")
            elapsed = time.perf_counter() - self.start_time
            print(f"Real time elapsed: {elapsed:.2f}s")
            print(
                f"Average policy frequency: {self.tick_count / elapsed:.1f}Hz (target: {1 / self.step_dt}Hz)"
            )
            print("=" * 60)

    def process_control_step(self):
        """Process one control step (called at control_freq Hz)."""
        self.tick_count += 1        
        self.curr_time = time.perf_counter()

        
        emergency_cond = (
            self.controller_state.RT
            or self.emergency_mode
        )
        policy_run_cond = (
            self.controller_state.LT
        )
        stand_down_cond = (self.controller_state.A)
        stand_up_cond = (self.controller_state.LB)

        if emergency_cond or self.emergency_mode:
            if not self.emergency_mode:
                self.emergency_mode_start_time = time.perf_counter()
            self.emergency_mode = True
            self.emergency_mode_control()

        elif policy_run_cond:
            self.run_policy = True

        elif (self.curr_time - self.start_time)<= self.time_to_stand or (self.run_policy == False and self.stand_down == False and stand_down_cond == False):
            self.stand_control("up")
        # Run policy
        elif (
            (self.curr_time - self.start_time
        )>= self.time_to_stand and stand_down_cond )or self.stand_down:
            if self.curr_stand_down_time == 0.0:
                self.curr_stand_down_time = time.perf_counter()
            self.run_policy = False
            self.stand_down = True
            self.stand_control("down")
        elif (
            self.curr_time - self.start_time
        )>= self.time_to_stand and self.run_policy:
            self.policy_control()

    def policy_control(self):
        torch.set_printoptions(precision=2, threshold=sys.maxsize, linewidth=200, edgeitems=100)
        print(self.height_map[0, :, :])
        
        if self.lidar_obs:
            obs = get_obs_lidar(
                self.latest_low_state,
                self.controller_state,
                self.height_map,
                prev_actions=self.current_action,
                mapper=self.mapper,
            )
        else:
            obs = get_obs(
                self.latest_low_state,
                self.controller_state,
                prev_actions=self.current_action,
                mapper=self.mapper,
            )

        with torch.no_grad():
            obs_tensor = torch.tensor(
                obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            actions_tensor = self.policy(obs_tensor)
        actions_policy_order = actions_tensor.squeeze(0)
        self.current_action = actions_policy_order.copy()
        self.send_motor_commands()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Go2 RL Policy Controller")
    parser.add_argument(
        "--newton", action="store_true", help="If the policy comes from newton, the mapping is not the same"
    )
    parser.add_argument(
        "--lidar", action="store_true", help="Whether to pass the lidar in the observation vector"
    )
    parser.add_argument(
        "--interface", type=str, default=None, help="Network interface for channel factory"
    )

    args = parser.parse_args()

    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")

    if args.interface:
        ChannelFactoryInitialize(0, args.interface)
    else:
        ChannelFactoryInitialize(0)
    custom = Go2PolicyController(newton = args.newton, lidar_obs=args.lidar)
    custom.Init()
    custom.Start()

    while True:        
        # if custom.percent_4 == 1.0: 
        #    time.sleep(1)
        #    print("Done!")
        #    sys.exit(-1)  
        # 
        # print("max_x sampled = ", custom.max_x)  
        # print("min_x sampled = ", custom.min_x)  
        # print("max_x_z sampled = ", custom.max_x_z)  
        # print("min_x_z sampled = ", custom.min_x_z)  
        # print("max_z sampled = ", custom.max_z)  
        # print("min_z sampled = ", custom.min_z) 
        time.sleep(1)


if __name__ == "__main__":
    main()
