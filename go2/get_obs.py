#!/usr/bin/env python3



import torch 
import yaml
import os
from utils import Mapper
from controller.controller import ControllerMsg
from unitree_sdk2_python.unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2_python.unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
import sys
from utils import quat_rotate_inverse


def get_obs(
    lowstate_msg: LowState_,
    controller_msg: ControllerMsg,
    prev_actions: torch.tensor,
    mapper: Mapper,
):
    """
    Extract observations from LowState message for RL policy.

    Args:
        msg: LowState message from robot
        obs_buffer: ObservationBuffer containing previous actions and commands

    Returns:
        obs: torch tensor of shape (49,) with observation vector

    Observation structure (49 dimensions):
    - obs[0:3]   : Base angular velocity (from IMU)
    - obs[3:6]   : Gravity direction (from IMU)
    - obs[6:9]  : Command velocity (x, y, yaw)
    - obs[9]    : Height command
    - obs[10:22] : Joint positions relative to default (12 joints)
    - obs[22:34] : Joint velocities (12 joints)
    - obs[34:46] : Previous actions (12 values)
    - obs[46:50] : Foot contacts
    """
    
    motor_states = lowstate_msg.motor_state[:12]
    # print(height_map[:,:,0])
    
    current_joint_pos_sdk = torch.tensor([motor_states[i].q for i in range(12)])
    current_joint_vel_sdk = torch.tensor([motor_states[i].dq for i in range(12)])

    current_joint_pos_policy = mapper.remap_joints_by_name(
        current_joint_pos_sdk,
        mapper.target_names,
        mapper.source_names,
        mapper.target_to_source,
    )
    current_joint_vel_policy = mapper.remap_joints_by_name(
        current_joint_vel_sdk,
        mapper.target_names,
        mapper.source_names,
        mapper.target_to_source,
    )
    default_pos_policy = mapper.default_pos_policy


    obs = torch.zeros(50)
    # Base linear velocity (obs[0:3])

    # Base angular velocity (gyroscope) (obs[0:3])
    obs[0:3] = torch.tensor(
        [
            lowstate_msg.imu_state.gyroscope[0],
            lowstate_msg.imu_state.gyroscope[1],
            lowstate_msg.imu_state.gyroscope[2],
        ]
    )
    # Computing projected gravity from IMU sensor
    quat = torch.tensor(
        [
            lowstate_msg.imu_state.quaternion[0],  # w
            lowstate_msg.imu_state.quaternion[1],  # x
            lowstate_msg.imu_state.quaternion[2],  # y
            lowstate_msg.imu_state.quaternion[3],  # z
        ]
    )

    gravity_world = torch.tensor([0.0, 0.0, -1.0])

    gravity_b = quat_rotate_inverse(quat, gravity_world)

    obs[3:6] = gravity_b

    
    obs[6:9] = torch.tensor([
        controller_msg.ly * 3.0 / 4.0,  # forward velocity
        controller_msg.lx * 3.0 / 4.0,  # lateral velocity (flip for correct direction)
        controller_msg.rx # yaw rate
    ])
    obs[9] = 0.3 + controller_msg.ry / 10

    # Fill joint positions (obs[13:25]) in policy order
    obs[10:22] = current_joint_pos_policy - default_pos_policy
    # Fill joint velocities (obs[25:37]) in policy order
    obs[22:34] = current_joint_vel_policy
    # Previous actions (obs[37:49]) - default to zero
    obs[34:46] = prev_actions
    # see the best threshold for real robot
    obs[46:50] = torch.tensor([
        float(lowstate_msg.foot_force[0]>20),
        float(lowstate_msg.foot_force[1]>20),
        float(lowstate_msg.foot_force[2]>20),
        float(lowstate_msg.foot_force[3]>20)
    ])

    return obs


def get_obs_lidar(
    lowstate_msg: LowState_,
    controller_msg: ControllerMsg,
    height_map: torch.tensor,
    prev_actions: torch.tensor,
    mapper: Mapper,
):
    """
    Extract observations from LowState message for RL policy.

    Args:
        msg: LowState message from robot
        obs_buffer: ObservationBuffer containing previous actions and commands

    Returns:
        obs: torch tensor of shape (49,) with observation vector

    Observation structure (49 dimensions):
    - obs[0:3]   : Base angular velocity (from IMU)
    - obs[3:6]   : Gravity direction (from IMU)
    - obs[6:9]  : Command velocity (x, y, yaw)
    - obs[9:21] : Joint positions relative to default (12 joints)
    - obs[21:33] : Joint velocities (12 joints)
    - obs[33:183] : Height map (150 values)
    - obs[183:195] : Previous actions (12 values)
    """
    
    motor_states = lowstate_msg.motor_state[:12]
    # print(height_map[:,:,0])
    height_map_copy = height_map[:,:,0].clone()
    
    current_joint_pos_sdk = torch.tensor([motor_states[i].q for i in range(12)])
    current_joint_vel_sdk = torch.tensor([motor_states[i].dq for i in range(12)])

    current_joint_pos_policy = mapper.remap_joints_by_name(
        current_joint_pos_sdk,
        mapper.target_names,
        mapper.source_names,
        mapper.target_to_source,
    )
    current_joint_vel_policy = mapper.remap_joints_by_name(
        current_joint_vel_sdk,
        mapper.target_names,
        mapper.source_names,
        mapper.target_to_source,
    )
    default_pos_policy = mapper.default_pos_policy

    obs = torch.zeros(195)

    # Base vel
    
    # Base angular velocity (gyroscope) (obs[0:3])
    obs[0:3] = torch.tensor([
        lowstate_msg.imu_state.gyroscope[0],
        lowstate_msg.imu_state.gyroscope[1],
        lowstate_msg.imu_state.gyroscope[2]
    ])
    
    # Computing projected gravity from IMU sensor
    quat = torch.tensor([
        lowstate_msg.imu_state.quaternion[0],  # w
        lowstate_msg.imu_state.quaternion[1],  # x
        lowstate_msg.imu_state.quaternion[2],  # y
        lowstate_msg.imu_state.quaternion[3]   # z
    ])
    
    gravity_world = torch.tensor([0.0, 0.0, -1.0])
    gravity_b = quat_rotate_inverse(quat, gravity_world)
    obs[3:6] = gravity_b
    
    # Command velocity (obs[6:9])
    obs[6:9] = torch.tensor([controller_msg.ly * 3.0 / 4.0, controller_msg.lx * 3.0 / 4.0, controller_msg.rx])
    
    
    # Fill joint positions (obs[10:22]) in policy order
    obs[9:21] = current_joint_pos_policy - default_pos_policy
    
    # Fill joint velocities (obs[22:34]) in policy order
    obs[21:33] = current_joint_vel_policy
    
    idx = 33+150
    height_map = height_map_copy - 0.28 
    obs[33:idx] = height_map

    obs[idx:idx + 12] = prev_actions

    return obs
