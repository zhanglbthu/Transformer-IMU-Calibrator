import time

import numpy as np
import torch

import config
from articulate.math import r6d_to_rotation_matrix, rotation_matrix_to_r6d, normalize_tensor, axis_angle_to_rotation_matrix
from tqdm import tqdm

from articulate.math import rotation_matrix_to_euler_angle, euler_angle_to_rotation_matrix, quaternion_to_rotation_matrix

def ego_drift_regularization(rot, imu_num=config.imu_num, ego_yaw_idx=-1):
    rot = rot.reshape(imu_num, 3, 3)
    rot_ego = rot[ego_yaw_idx]

    rot_ego_euler = rotation_matrix_to_euler_angle(rot_ego, seq='YZX').squeeze(0)
    # heading_ref_euler[:, [1, 2]] *= 0
    rot_ego_euler[0] *= 0
    rot_ego = euler_angle_to_rotation_matrix(rot_ego_euler, seq='YZX')

    rot[ego_yaw_idx] = rot_ego
    return rot

@torch.no_grad()
def rotation_diversity(rot):
    """
    计算一段序列中rotation的丰富度
    :param rot: batch x seq_len x imu_num x 3 x 3
    :return:
    """
    n_batch, seq_len, imu_num = rot.shape[0], rot.shape[1], rot.shape[2]
    rot = rot.reshape(-1, 3, 3)
    euler_angle = rotation_matrix_to_euler_angle(rot).reshape(n_batch, seq_len, imu_num, 3) * 180 / np.pi
    # 离散化的角度
    dis_angle = torch.div(euler_angle, 15, rounding_mode='floor').long() + torch.LongTensor([12, 6, 12]).reshape(1,
                                                                                                                 1,
                                                                                                                 1,
                                                                                                                 3).to(
        euler_angle.device)
    # 离散空间索引
    dis_angle_idx = torch.clip(dis_angle[:, :, :, [0]], 0, 23) + torch.clip(dis_angle[:, :, :, [1]], 0, 11) * 24 + \
                    torch.clip(dis_angle[:, :, :, [2]], 0, 23) * 12 * 24

    angle_space = torch.zeros(n_batch, seq_len, imu_num, 24 * 12 * 24, dtype=torch.uint8).to(euler_angle.device)
    angle_space.scatter_add_(3, dis_angle_idx, torch.ones_like(angle_space, dtype=torch.uint8))
    angle_space_sum = angle_space.sum(dim=1)
    angle_space_mask = (angle_space_sum > 0).reshape(n_batch, imu_num, -1)
    diversity = angle_space_mask.sum(dim=-1)
    return diversity.cpu()

class TicOperator():
    def __init__(self, TIC_network, imu_num=6, ego_imu_idx=-1, data_frame_rate=60):
        self.buffer_size = 512
        self.TR_drift = torch.Tensor([10, 10, 10, 10, 10, 0]) * 1
        self.TR_offset = torch.Tensor([30, 50, 30, 30, 25, 15]) * 1
        self.data_frame_rate=data_frame_rate
        self.ego_idx = ego_imu_idx
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # R_{G'G} in paper. D: Drifted Global Coordinate System
        self.R_DG = torch.eye(3).reshape(-1, 3, 3).repeat(imu_num, 1, 1)
        # R_{BS} in paper.
        self.R_BS = torch.eye(3).reshape(-1, 3, 3).repeat(imu_num, 1, 1)

        self.GA = torch.FloatTensor([[0, -9.80848, 0]]).repeat(imu_num, 1).unsqueeze(-1)

        self.data_buffer = []
        self.imu_num = imu_num
        self.model = TIC_network

    def reset(self):
        self.R_DG = torch.eye(3).reshape(-1, 3, 3).repeat(self.imu_num, 1, 1)
        self.R_BS = torch.eye(3).reshape(-1, 3, 3).repeat(self.imu_num, 1, 1)
        self.GA = torch.FloatTensor([[0, -9.80848, 0]]).repeat(self.imu_num, 1).unsqueeze(-1)
        self.data_buffer = []

    @torch.no_grad()
    def calibrate_step(self, acc_cat_rot):
        acc, rot = acc_cat_rot[0:self.imu_num * 3].view(-1, self.imu_num, 3, 1), acc_cat_rot[self.imu_num * 3:].view(-1,
                                                                                                        self.imu_num, 3,
                                                                                                        3)
        rot = self.R_DG.transpose(-2, -1).matmul(rot).matmul(self.R_BS.transpose(-2, -1))
        acc = self.R_DG.transpose(-2, -1).matmul(acc - self.GA) + self.GA

        return torch.cat([acc.flatten(1), rot.flatten(1)], dim=-1)

    @torch.no_grad()
    def dynamic_calibration(self):
        if len(self.data_buffer) < 512:
            return
        frame_nums = 512

        # down sample
        acc_cat_oris = torch.stack(self.data_buffer[-frame_nums:]).reshape(frame_nums, -1)[::self.data_frame_rate//30]
        acc_cat_oris[:, :self.imu_num * 3] /= 30
        acc_cat_oris = acc_cat_oris.to(self.device)

        # regularization
        self.R_DG = r6d_to_rotation_matrix(rotation_matrix_to_r6d(self.R_DG))
        self.R_BS = r6d_to_rotation_matrix(rotation_matrix_to_r6d(self.R_BS))

        oris = acc_cat_oris[:, self.imu_num * 3:].reshape(1, -1, self.imu_num, 3, 3)
        # t1 = time.time()
        RD = rotation_diversity(oris).reshape(-1)
        # t2 = time.time()
        # print('RD:', t2-t1)

        acc_cat_oris = acc_cat_oris.reshape(1, -1, self.imu_num * 12)
        # print(self.R_DG)

        # t1 = time.time()
        delta_R_DG, delta_R_BS = self.model(acc_cat_oris)
        # t2 = time.time()
        # print('TIC Network:', t2 - t1)

        delta_R_DG = r6d_to_rotation_matrix(delta_R_DG.reshape(-1, 6).cpu())
        delta_R_BS = r6d_to_rotation_matrix(delta_R_BS.reshape(-1, 6).cpu())

        delta_R_DG = ego_drift_regularization(delta_R_DG)

        # RD Trigger
        trigger_drift = RD > self.TR_drift
        skip_mask_drift = ~trigger_drift

        trigger_offset = RD > self.TR_offset
        skip_mask_offset = ~trigger_offset

        skip_count_drift = torch.sum(skip_mask_drift).item()
        skip_count_offset = torch.sum(skip_mask_offset).item()

        if min(skip_count_drift, skip_count_offset) < self.imu_num:
            if skip_count_drift > 0:
                delta_R_DG[skip_mask_drift, :, :] = torch.eye(3).unsqueeze(0).repeat(skip_count_drift, 1, 1)
            if skip_count_offset > 0:
                delta_R_BS[skip_mask_offset, :, :] = torch.eye(3).unsqueeze(0).repeat(skip_count_offset, 1, 1)

            self.R_DG = self.R_DG.matmul(delta_R_DG)
            self.R_BS = delta_R_BS.matmul(self.R_BS)

            self.data_buffer = []


    def run(self, rot, acc, trigger_t=1):
        self.reset()
        trigger_gap = int(self.data_frame_rate*trigger_t)
        acc = acc.reshape(-1, self.imu_num*3)
        rot = rot.reshape(-1, self.imu_num*3*3)

        origin_acc_cat_rot = torch.cat([acc, rot], dim=-1)
        pred_drift = []
        pred_offset = []
        recali_data = []
        for i in tqdm(range(len(origin_acc_cat_rot))):
            recali_data.append(self.calibrate_step(origin_acc_cat_rot[i]))
            self.data_buffer.append(recali_data[-1].clone())
            pred_drift.append(self.R_DG.clone())
            pred_offset.append(self.R_BS.clone())
            if i % trigger_gap == 0:
                self.dynamic_calibration()
            if len(self.data_buffer) > self.buffer_size:
                self.data_buffer = self.data_buffer[-self.buffer_size:]

        recali_data = torch.cat(recali_data, dim=0)
        acc = recali_data[:, :3*self.imu_num].reshape(-1, self.imu_num, 3)
        rot = recali_data[:, 3 * self.imu_num:].reshape(-1, self.imu_num, 3, 3)

        return rot, acc, torch.stack(pred_drift, dim=0), torch.stack(pred_offset, dim=0)


class TicOperatorOverwrite():
    def __init__(self, TIC_network, imu_num=6, ego_imu_idx=-1, data_frame_rate=60):
        self.buffer_size = 512
        self.TR_drift = torch.Tensor([10, 10, 10, 10, 10, 0]) * 1
        self.TR_offset = torch.Tensor([30, 50, 30, 30, 25, 15]) * 1
        self.data_frame_rate=data_frame_rate
        self.ego_idx = ego_imu_idx
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # R_{G'G} in paper. D: Drifted Global Coordinate System
        self.R_DG = torch.eye(3).reshape(-1, 3, 3).repeat(imu_num, 1, 1)
        # R_{BS} in paper.
        self.R_BS = torch.eye(3).reshape(-1, 3, 3).repeat(imu_num, 1, 1)

        self.GA = torch.FloatTensor([[0, -9.80848, 0]]).repeat(imu_num, 1).unsqueeze(-1)

        self.data_buffer = []
        self.imu_num = imu_num
        self.model = TIC_network

    def reset(self):
        self.R_DG = torch.eye(3).reshape(-1, 3, 3).repeat(self.imu_num, 1, 1)
        self.R_BS = torch.eye(3).reshape(-1, 3, 3).repeat(self.imu_num, 1, 1)
        self.GA = torch.FloatTensor([[0, -9.80848, 0]]).repeat(6, 1).unsqueeze(-1)
        self.data_buffer = []

    @torch.no_grad()
    def calibrate_step(self, acc_cat_rot):
        acc, rot = acc_cat_rot[0:self.imu_num * 3].view(-1, self.imu_num, 3, 1), acc_cat_rot[self.imu_num * 3:].view(-1,
                                                                                                        self.imu_num, 3,
                                                                                                        3)
        rot = self.R_DG.transpose(-2, -1).matmul(rot).matmul(self.R_BS.transpose(-2, -1))
        acc = self.R_DG.transpose(-2, -1).matmul(acc - self.GA) + self.GA

        return torch.cat([acc.flatten(1), rot.flatten(1)], dim=-1)

    @torch.no_grad()
    def dynamic_calibration(self):
        if len(self.data_buffer) < 512:
            return
        frame_nums = 512

        # down sample
        acc_cat_oris = torch.stack(self.data_buffer[-frame_nums:]).reshape(frame_nums, -1)[::self.data_frame_rate//30]
        acc_cat_oris[:, :self.imu_num * 3] /= 30
        acc_cat_oris = acc_cat_oris.to(self.device)

        # regularization
        self.R_DG = r6d_to_rotation_matrix(rotation_matrix_to_r6d(self.R_DG))
        self.R_BS = r6d_to_rotation_matrix(rotation_matrix_to_r6d(self.R_BS))

        oris = acc_cat_oris[:, self.imu_num * 3:].reshape(1, -1, self.imu_num, 3, 3)

        RD = rotation_diversity(oris).reshape(-1)

        acc_cat_oris = acc_cat_oris.reshape(1, -1, self.imu_num * 12)
        # print(self.R_DG)

        delta_R_DG, delta_R_BS = self.model(acc_cat_oris)


        delta_R_DG = r6d_to_rotation_matrix(delta_R_DG.reshape(-1, 6).cpu())
        delta_R_BS = r6d_to_rotation_matrix(delta_R_BS.reshape(-1, 6).cpu())

        delta_R_DG = ego_drift_regularization(delta_R_DG)

        # RD Trigger
        trigger_drift = RD > self.TR_drift
        skip_mask_drift = ~trigger_drift

        trigger_offset = RD > self.TR_offset
        skip_mask_offset = ~trigger_offset

        skip_count_drift = torch.sum(skip_mask_drift).item()
        skip_count_offset = torch.sum(skip_mask_offset).item()

        if min(skip_count_drift, skip_count_offset) < self.imu_num:
            if skip_count_drift > 0:
                delta_R_DG[skip_mask_drift, :, :] = self.R_DG[skip_mask_drift, :, :]
            if skip_count_offset > 0:
                delta_R_BS[skip_mask_offset, :, :] = self.R_BS[skip_mask_offset, :, :]

            self.R_DG = delta_R_DG
            self.R_BS = delta_R_BS

            self.data_buffer = []


    def run(self, rot, acc, trigger_t=1):
        self.reset()
        trigger_gap = int(self.data_frame_rate*trigger_t)
        acc = acc.reshape(-1, self.imu_num*3)
        rot = rot.reshape(-1, self.imu_num*3*3)

        origin_acc_cat_rot = torch.cat([acc, rot], dim=-1)
        pred_drift = []
        pred_offset = []
        recali_data = []
        for i in tqdm(range(len(origin_acc_cat_rot))):
            recali_data.append(self.calibrate_step(origin_acc_cat_rot[i]))
            # self.data_buffer.append(recali_data[-1].clone())
            self.data_buffer.append(origin_acc_cat_rot[i].clone())
            pred_drift.append(self.R_DG.clone())
            pred_offset.append(self.R_BS.clone())
            if i % trigger_gap == 0:
                self.dynamic_calibration()
            if len(self.data_buffer) > self.buffer_size:
                self.data_buffer = self.data_buffer[-self.buffer_size:]

        recali_data = torch.cat(recali_data, dim=0)
        acc = recali_data[:, :3*self.imu_num].reshape(-1, self.imu_num, 3)
        rot = recali_data[:, 3 * self.imu_num:].reshape(-1, self.imu_num, 3, 3)

        return rot, acc, torch.stack(pred_drift, dim=0), torch.stack(pred_offset, dim=0)







