import numpy as np
import torch
from articulate.math.angular import *
from articulate.math.angular import rotation_matrix_2_angle
from articulate.math.general import normalize_tensor
from Aplus.tools.data_visualize import data_dict_2_df

# 6个rotation的误差计算, 带std
def angle_diff(rot_1, rot_2, imu_num=6, print_result=False):
    rot_1 = rot_1.reshape(-1, imu_num, 3, 3)
    rot_2 = rot_2.reshape(-1, imu_num, 3, 3)
    rot_diff = rot_1.transpose(-2, -1).matmul(rot_2)
    rot_diff_ang = rotation_matrix_2_angle(rot_diff.reshape(-1, 3, 3)).reshape(-1, imu_num)
    rot_diff_ang = rot_diff_ang*180/np.pi
    if print_result:
        for i in range(imu_num):
            print(f'rot_{i+1}: {rot_diff_ang[:, i].mean()} ± {rot_diff_ang[:, i].std()}')
    return rot_diff_ang

def acc_diff(acc_1, acc_2, imu_num=6, print_result=False):
    diff = acc_2 - acc_1
    distance = torch.norm(diff.reshape(-1, imu_num, 3), p=2, dim=-1)
    if print_result:
        for i in range(imu_num):
            print(f'acc_{i+1}: {distance[:, i].mean()} ± {distance[:, i].std()}')
    return distance

def only_y_rot(rot):
    rot = rot.reshape(-1, 3, 3)
    y_mask = torch.tensor([[[1, 0, 1], [0, 1, 0], [1, 0, 1]]]).repeat(rot.shape[0], 1, 1)
    # x, z轴向世界坐标x-z平面投影, y轴向世界坐标y轴投影
    heading_ref = rot * y_mask
    heading_ref = normalize_tensor(x=heading_ref.reshape(-1, 3)).view_as(rot)
    return heading_ref

def to_ego_yaw(R, ego_idx=-1):
    # print(R.shape)
    ego_ori = R[:, ego_idx]
    ego_euler = rotation_matrix_to_euler_angle(r=ego_ori, seq='YZX')
    ego_euler[:, [1, 2]] *= 0
    yaw_euler = ego_euler
    yaw_rot = euler_angle_to_rotation_matrix(q=yaw_euler, seq='YZX').unsqueeze(1)
    R = yaw_rot.transpose(-2, -1).matmul(R)
    return R

def get_ego_yaw(R, ego_idx=-1):
    ego_ori = R[:, ego_idx]
    ego_euler = rotation_matrix_to_euler_angle(r=ego_ori, seq='YZX')
    ego_euler[:, [1, 2]] *= 0
    yaw_euler = ego_euler
    yaw_rot = euler_angle_to_rotation_matrix(q=yaw_euler, seq='YZX').unsqueeze(1)
    return yaw_rot




