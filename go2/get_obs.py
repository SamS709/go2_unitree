#!/usr/bin/env python3



import numpy as np
import yaml
import os
from utils import Mapper
from controller.controller import ControllerMsg
from unitree_sdk2_python.unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2_python.unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
import sys
np.set_printoptions(precision=2, threshold=sys.maxsize, linewidth=np.inf, edgeitems=100, suppress=True)
from utils import quat_rotate_inverse


def get_obs_low_state(
    lowstate_msg: LowState_,
    controller_msg: ControllerMsg,
    height_map: np.array,
    height: float,
    prev_actions: np.array,
    mapper: Mapper,
    pass_lidar: bool
):
    """
    Extract observations from LowState message for RL policy.

    Args:
        msg: LowState message from robot
        obs_buffer: ObservationBuffer containing previous actions and commands

    Returns:
        obs: numpy array of shape (49,) with observation vector

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
    
    np.set_printoptions(precision=2, threshold=sys.maxsize, linewidth=np.inf, edgeitems=100, suppress=True)
    motor_states = lowstate_msg.motor_state[:12]
    print(height_map[:,:,0])
    height_map_copy = height_map[:,:,0].copy()
    height_map_copy[:,:] = np.array([[0.25 for j in range(height_map_copy.shape[0])] for i in range(height_map_copy.shape[0])])
    
    current_joint_pos_sdk = np.array([motor_states[i].q for i in range(12)])
    current_joint_vel_sdk = np.array([motor_states[i].dq for i in range(12)])

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

    # FILLING OBS VECTOR
    if lidar:
        obs = np.zeros(50 + height_map_copy.shape[0]**2)
        obs[50:] = height_map_copy.flatten()
    else:
        obs = np.zeros(50)
    # Base linear velocity (obs[0:3])

    # Base angular velocity (gyroscope) (obs[0:3])
    obs[0:3] = np.array(
        [
            lowstate_msg.imu_state.gyroscope[0],
            lowstate_msg.imu_state.gyroscope[1],
            lowstate_msg.imu_state.gyroscope[2],
        ]
    )
    # Computing projected gravity from IMU sensor
    quat = np.array(
        [
            lowstate_msg.imu_state.quaternion[0],  # w
            lowstate_msg.imu_state.quaternion[1],  # x
            lowstate_msg.imu_state.quaternion[2],  # y
            lowstate_msg.imu_state.quaternion[3],  # z
        ]
    )

    gravity_world = np.array([0.0, 0.0, -1.0])

    gravity_b = quat_rotate_inverse(quat, gravity_world)

    obs[3:6] = gravity_b

    
    obs[6:9] = [
        controller_msg.ly * 3.0 / 4.0,  # forward velocity
        controller_msg.lx * 3.0 / 4.0,  # lateral velocity (flip for correct direction)
        controller_msg.rx ,  # yaw rate
    ]
    obs[9] = 0.3 + controller_msg.ry / 10

    # Fill joint positions (obs[13:25]) in policy order
    obs[10:22] = current_joint_pos_policy - default_pos_policy
    # Fill joint velocities (obs[25:37]) in policy order
    obs[22:34] = current_joint_vel_policy
    # Previous actions (obs[37:49]) - default to zero
    obs[34:46] = prev_actions
    # see the best threshold for real robot
    obs[46:50] = [
        float(lowstate_msg.foot_force[0]>20),
        float(lowstate_msg.foot_force[1]>20),
        float(lowstate_msg.foot_force[2]>20),
        float(lowstate_msg.foot_force[3]>20)
    ]

    return obs
