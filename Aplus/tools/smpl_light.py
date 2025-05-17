import time

import torch
import numpy as np
# 没smpl模型文件也能运行！用于纯姿态的IK&FK

# SMPL骨架
EDGES = {1:[[0, 1], [0, 2], [0, 3]],
         2:[[1, 4], [2, 5], [3, 6]],
         3:[[4, 7], [5, 8], [6, 9]],
         4:[[7, 10], [8, 11], [9, 12], [9, 13], [9, 14]],
         5:[[12, 15], [13, 16], [14, 17]],
         6:[[16, 18], [17, 19]],
         7:[[18, 20], [19, 21]],
         8:[[20, 22], [21, 23]]}

# local关节位置
JOINT_LOCAL = 0.001 * torch.FloatTensor([[   0.0000,    0.0000,    0.0000],
        [  58.5813,  -82.2800,  -17.6641],
        [ -60.3097,  -90.5133,  -13.5425],
        [   4.4394,  124.4036,  -38.3852],
        [  43.4514, -386.4695,    8.0370],
        [ -43.2566, -383.6879,   -4.8430],
        [   4.4884,  137.9564,   26.8203],
        [ -14.7903, -426.8745,  -37.4280],
        [  19.0555, -420.0456,  -34.5617],
        [  -2.2646,   56.0324,    2.8550],
        [  41.0544,  -60.2859,  122.0424],
        [ -34.8399,  -62.1055,  130.3233],
        [ -13.3902,  211.6355,  -33.4676],
        [  71.7025,  113.9997,  -18.8982],
        [ -82.9537,  112.4724,  -23.7074],
        [  10.1132,   88.9373,   50.4099],
        [ 122.9214,   45.2051,  -19.0460],
        [-113.2283,   46.8532,   -8.4721],
        [ 255.3319,  -15.6490,  -22.9465],
        [-260.1275,  -14.3692,  -31.2687],
        [ 265.7092,   12.6981,   -7.3747],
        [-269.1084,    6.7937,   -6.0268],
        [  86.6905,  -10.6360,  -15.5943],
        [ -88.7537,   -8.6516,  -10.1071]])

Z_90_p = torch.FloatTensor([[0, -1,  0],
                            [1,  0,  0],
                            [0,  0,  1]])

Z_90_n = Z_90_p.t()

class SMPLight:
    def __init__(self):
        # 父节点-子节点映射, 用于IK
        self.pc_mapping = []
        # 分层次的父节点-子节点映射, 用于FK
        self.layered_pc_mapping = {}
        for k, v in EDGES.items():
            self.pc_mapping += v
            v = torch.LongTensor(v)
            self.layered_pc_mapping.update({k:[np.array(v[:, 0]).tolist(), np.array(v[:, 1]).tolist()]})
        self.pc_mapping = torch.LongTensor(self.pc_mapping)
        self.pc_mapping = [np.array(self.pc_mapping[:, 0]).tolist(), np.array(self.pc_mapping[:, 1]).tolist()]
        self.joint_local = JOINT_LOCAL

    @torch.no_grad()
    def forward_kinematics(self, R, trans=None, calc_joint=False):
        if calc_joint:
            return self._forward_kinematics_with_joint(R, trans)

        for _, mapping in self.layered_pc_mapping.items():
            p_idx, c_idx = mapping[0], mapping[1]
            R[..., c_idx, :, :] = R[..., p_idx, :, :].matmul(R[..., c_idx, :, :])
        return R

    @torch.no_grad()
    def _forward_kinematics_with_joint(self, R, trans):
        # positions n x 24 x 3
        positions = torch.zeros_like(R[..., -1]) + self.joint_local.to(R.device)

        if trans is not None:
            positions[..., 0, :] += trans

        # n x 24 x 3 x 4
        Rk = torch.cat([R, positions.unsqueeze(-1)], dim=-1)
        padding = torch.zeros_like(Rk[..., [-1], :])
        padding[..., -1] += 1

        # n x 24 x 4 x 4
        Rk = torch.cat([Rk, padding], dim=-2)

        for _, mapping in self.layered_pc_mapping.items():
            p_idx, c_idx = mapping[0], mapping[1]
            Rk[..., c_idx, :, :] = Rk[..., p_idx, :, :].matmul(Rk[..., c_idx, :, :])

        # n x 24 x 3 x 4
        Rk = Rk[..., :-1, :]
        # n x 24 x 3 x 3
        R = Rk[..., :, :-1]
        # n x 24 x 3
        joint = Rk[..., :, -1]

        return R, joint

    @torch.no_grad()
    def inverse_kinematics(self, R):
        p_idx, c_idx = self.pc_mapping[0], self.pc_mapping[1]
        R[..., c_idx, :, :] = R[..., p_idx, :, :].transpose(-2, -1).matmul(R[..., c_idx, :, :])
        return R

class SMPLPose:
    body_model = SMPLight()
    t_pose = torch.eye(3).unsqueeze(0).repeat(24, 1, 1)
    n_pose = t_pose.clone()
    n_pose[17], n_pose[18] = Z_90_p, Z_90_n

    t_pose_ori, t_pose_joint = body_model.forward_kinematics(t_pose, calc_joint=True)
    n_pose_ori, n_pose_joint = body_model.forward_kinematics(n_pose, calc_joint=True)




import articulate as art
from config import paths

# 计算local joint

# body_model = art.ParametricModel(paths.smpl_file)
# p = torch.eye(3).unsqueeze(0).repeat(24, 1, 1).unsqueeze(0)
# body_shape = torch.zeros(10)
# init_trans = torch.zeros(3)
# # 输入24个关节旋转+体型参数+位移信息, 输出24个关节的旋转+蒙皮点加速度+运动速度
# grot, joint = body_model.forward_kinematics(p, body_shape, init_trans, calc_mesh=False)
# joint = joint.squeeze(0)
#
# sl = SMPLight()
# p_idx, c_idx = sl.pc_mapping[0], sl.pc_mapping[1]
#
# # print(joint)
# joint[c_idx] = joint[c_idx] - joint[p_idx]
# local_joint_position = joint
# #单位转为mm
# # print(local_joint_position*1000)

# # 测试
# sl = SMPLight()
# body_model = art.ParametricModel(paths.smpl_file)
# p = torch.eye(3).unsqueeze(0).repeat(24, 1, 1).unsqueeze(0).repeat(1000, 1, 1, 1)
# body_shape = torch.zeros(10)
# init_trans = torch.zeros(3).unsqueeze(0).repeat(1000,1)
# # 输入24个关节旋转+体型参数+位移信息, 输出24个关节的旋转+蒙皮点加速度+运动速度
# t1 = time.time()
# grot, joint = body_model.forward_kinematics(p, body_shape, init_trans, calc_mesh=False)
# t2 = time.time()
# print(joint, t2-t1)
#
# # p = p.cuda()
# # init_trans = init_trans.cuda()
#
# t1 = time.time()
# grot, joint = sl.forward_kinematics(R=p, trans=init_trans, calc_joint=True)
# t2 = time.time()
# print(joint, t2-t1)





