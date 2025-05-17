import torch
from articulate.math.angular import *
import numpy as np
from Aplus.data.process import add_gaussian_noise
import config

@torch.no_grad()
def imu_drift_offset_simulation(imu_rot, imu_acc, imu_num=6, ego_imu_id=-1, drift_range=60,
                 offset_range=45, random_global_yaw=True, global_yaw_only=False, acc_noise=0.025):
    """
    Simulates drift and offset effects on IMU (Inertial Measurement Unit) data.

    This function applies simulated drift and offset transformations to the rotation and acceleration data
    from multiple IMUs. It applies random drift and offset, and optionally applies random global yaw rotation.

    Parameters:
    ----------
    imu_rot : torch.Tensor
        A tensor of shape (batch_size, seq_len, imu_num, 3, 3) representing the well-calibrated IMU rotations
        (bone orientation measurements in SMPL frame) sequences in the batch.

    imu_acc : torch.Tensor
        A tensor of shape (batch_size, seq_len, imu_num, 3, 1) representing the acceleration data (in SMPL frame)
        for each IMU in the batch.

    imu_num : int, optional
        The number of IMUs in the simulation (default is 6).

    ego_imu_id : int, optional
        The ID of the ego-yaw IMU (default is -1).

    drift_range : float, optional
        The maximum range of random drift to apply to the IMU data, in degrees (default is 60 degrees).

    offset_range : float, optional
        The maximum range of random offset to apply to the IMU data, in degrees (default is 45 degrees).

    random_global_yaw : bool, optional
        If True, a random global yaw rotation will be applied to the IMU data (default is True).

    global_yaw_only : bool, optional
        If True, only global yaw will be applied without drift or offset (default is False).

    acc_noise : float, optional
        The standard deviation of the Gaussian noise to be added to the acceleration data (default is 0.025).

    Returns:
    -------
    imu_acc : torch.Tensor
        The modified acceleration data after applying drift, offset, and noise.

    imu_rot : torch.Tensor
        The modified rotation data after applying drift, offset, and any global yaw transformations.

    drift : torch.Tensor
        A tensor representing the simulated drift applied to the IMU data, shaped as (batch_size, imu_num, imu_num).

    offset : torch.Tensor
        A tensor representing the simulated offset applied to the IMU data, shaped as (batch_size, imu_num, imu_num).

    Notes:
    -----
    - The added drift matrices are in ego-yaw coordinate system.
    """

    batch_size = imu_rot.shape[0]
    seq_len = imu_rot.shape[1]
    drift_range = (drift_range / 180) * torch.pi
    non_yaw_drift_range = (20 / 180) * torch.pi
    offset_range = (offset_range / 180) * torch.pi
    GA = torch.FloatTensor([[0, -9.80665, 0]])

    # acc noise
    imu_acc = add_gaussian_noise(imu_acc, sigma=acc_noise)

    if global_yaw_only:
        # no drift
        drift = config.unit_r6d.reshape(1, 1, 6).repeat(batch_size, imu_num, 1).to(imu_rot.device)
        offset = drift.clone()
    else:
        drift = torch.zeros(batch_size, imu_num, 3).to(imu_rot.device)
        offset = torch.zeros(batch_size, imu_num, 3).to(imu_rot.device)
        
        # random drift
        drift[:, :, 0] = drift[:, :, 0].uniform_(-drift_range, drift_range)
        drift[:, :, [1, 2]] = drift[:, :, [1, 2]].uniform_(-non_yaw_drift_range, non_yaw_drift_range)
        drift[:, ego_imu_id, [0]] *= 0

        # random offset
        offset = offset.uniform_(-offset_range, offset_range)

        # random scaling
        if True:
            scale_mask = torch.zeros(batch_size, 1, 1).uniform_(0, 1).to(drift.device)
            drift *= scale_mask
            offset *= scale_mask
        
        drift = euler_angle_to_rotation_matrix(drift, seq='YZX').reshape(batch_size, imu_num, 3, 3)
        offset = euler_angle_to_rotation_matrix(offset).reshape(batch_size, imu_num, 3, 3)

    # random global yaw
    if random_global_yaw:
        global_yaw_rot = torch.zeros(batch_size, 1, 3).to(imu_rot.device)
        global_yaw_rot[:, :, 1] = global_yaw_rot[:, :, 1].uniform_(-np.pi, np.pi)
        global_yaw_rot = euler_angle_to_rotation_matrix(global_yaw_rot).reshape(batch_size, 1, 3, 3)

        global_yaw_rot = global_yaw_rot.unsqueeze(1).repeat(1, seq_len, imu_num, 1, 1)
        imu_rot = global_yaw_rot.matmul(imu_rot)

        if imu_acc is not None:
            imu_acc = global_yaw_rot.matmul(imu_acc)

    if global_yaw_only == False:
        # adding drift & offset
        drift = drift.unsqueeze(1).repeat(1, seq_len, 1, 1, 1)
        offset = offset.unsqueeze(1).repeat(1, seq_len, 1, 1, 1)
        imu_rot = drift.matmul(imu_rot).matmul(offset)
        # GA Leakage simulation
        if imu_acc is not None:
            GA = GA.to(imu_acc.device)
            GA = GA.reshape(1, 1, 1, 3, 1)
            GA = GA.repeat(imu_acc.shape[0], imu_acc.shape[1], imu_acc.shape[2], 1, 1)

            # [method 1 & 2 are equal]
            # method 1 (According to Hardware Level Acceleration formulation in the supp mat)
            imu_acc = drift.matmul(imu_acc) + (torch.eye(3, device=imu_acc.device) - drift).matmul(GA)

            # method 2
            # imu_acc -= GA
            # imu_acc = drift.matmul(imu_acc)
            # imu_acc += GA

    drift = rotation_matrix_to_r6d(drift[:, 0].reshape(-1, 3, 3)).reshape(batch_size, imu_num, 6)
    offset = rotation_matrix_to_r6d(offset[:, 0].reshape(-1, 3, 3)).reshape(batch_size, imu_num, 6)

    return imu_rot, imu_acc, drift, offset

