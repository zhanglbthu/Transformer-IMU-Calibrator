import articulate as art
import config
from my_model import *
from TicOperator import *
from evaluation_functions import *

import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = TIC(stack=3, n_input=6 * (3 + 3 * 3), n_output=6 * 6)
model.restore('./checkpoint/TIC_13.pth')
model = model.to(device).eval()

tag = 'TIC'
folders = ['s1', 's2', 's3', 's4', 's5']
# folders = ['s1']

# Inference
if True:
    print('=====Inference Start=====')
    for f in folders:
        data_root = os.path.join(config.paths.tic_dataset_dir, f)
        print(f'processing {f}')

        imu_acc = torch.load(os.path.join(data_root, 'acc.pt'))
        imu_rot = torch.load(os.path.join(data_root, 'rot.pt'))

        ts = TicOperator(TIC_network=model)
        # ts = TicOperatorOverwrite(TIC_network=model)
        rot, acc, pred_drift, pred_offset = ts.run(imu_rot, imu_acc, trigger_t=1)

        torch.save(acc, os.path.join(data_root, f'acc_fix_{tag}.pt'))
        torch.save(rot, os.path.join(data_root, f'rot_fix_{tag}.pt'))
        torch.save(pred_drift, os.path.join(data_root, f'pred_drift_{tag}.pt'))
        torch.save(pred_offset, os.path.join(data_root, f'pred_offset_{tag}.pt'))
    

# Evaluation
if True:
    print('=====Evaluation Start=====')
    result_drift_pred = []
    result_offset_pred = []
    result_drift_origin = []
    result_drift_err = []
    result_offset_origin = []
    result_offset_err = []

    result_static_ome = []
    result_dynamic_ome = []
    result_static_ame = []
    result_dynamic_ame = []


    for f in folders:
        print(f'processing {f}')
        data_root = f'/root/autodl-tmp/data/TIC_Dataset/{f}'

        drift_t = torch.load(os.path.join(data_root, 'drift.pt'))
        drift_p = torch.load(os.path.join(data_root, f'pred_drift_{tag}.pt'))

        offset_t = torch.load(os.path.join(data_root, 'offset.pt'))
        offset_p = torch.load(os.path.join(data_root, f'pred_offset_{tag}.pt'))

        # R_BS Err
        offset_err = angle_diff(offset_p, offset_t)
        result_offset_err.append(offset_err)

        # R_G'G Err
        drift_origin = drift_t.clone()
        drift_t = to_ego_yaw(drift_t)
        drift_err = angle_diff(drift_p, drift_t)
        result_drift_err.append(drift_err)

        # Get GT bone orientations from pose
        gt_pose = torch.load(os.path.join(data_root, 'pose.pt'))
        gt_pose = axis_angle_to_rotation_matrix(gt_pose.reshape(-1, 3)).reshape(-1, 24, 3, 3)
        body_model = art.ParametricModel('./SMPL_MALE.pkl')
        body_shape = torch.zeros(10)
        trans = torch.zeros(3)
        grot, joint = body_model.forward_kinematics(gt_pose, body_shape, trans, calc_mesh=False)
        gt_bone = grot[:, [18, 19, 4, 5, 15, 0]]
        imu_bone = torch.load(os.path.join(data_root, 'rot.pt'))
        tic_fix_bone = torch.load(os.path.join(data_root, f'rot_fix_{tag}.pt')).reshape(-1, 6, 3, 3)

        # Acc
        gt_acc = torch.load(os.path.join(data_root, 'acc_gt.pt')).reshape(-1, 6, 3, 1)
        imu_acc = torch.load(os.path.join(data_root, 'acc.pt')).reshape(-1, 6, 3, 1)
        tic_fix_acc = torch.load(os.path.join(data_root, f'acc_fix_{tag}.pt')).reshape(-1, 6, 3, 1)

        gt_yaw = get_ego_yaw(gt_bone)
        imu_yaw = get_ego_yaw(imu_bone)
        tic_fix_yaw = get_ego_yaw(tic_fix_bone)

        # transform rot to ego-yaw
        gt_bone = gt_yaw.transpose(-2, -1).matmul(gt_bone)
        imu_bone = imu_yaw.transpose(-2, -1).matmul(imu_bone)
        tic_fix_bone = tic_fix_yaw.transpose(-2, -1).matmul(tic_fix_bone)

        # transform acc to ego-yaw
        gt_acc = gt_yaw.transpose(-2, -1).matmul(gt_acc)
        imu_acc = imu_yaw.transpose(-2, -1).matmul(imu_acc)
        tic_fix_acc = tic_fix_yaw.transpose(-2, -1).matmul(tic_fix_acc)

        err_static = angle_diff(imu_bone, gt_bone)
        result_static_ome.append(err_static)

        acc_err_static = acc_diff(imu_acc, gt_acc)
        result_static_ame.append(acc_err_static)

        err_dynamic = angle_diff(tic_fix_bone, gt_bone)
        result_dynamic_ome.append(err_dynamic)

        acc_err_dynamic = acc_diff(tic_fix_acc, gt_acc)
        result_dynamic_ame.append(acc_err_dynamic)

    result_drift_err = torch.cat(result_drift_err, dim=0)
    result_offset_err = torch.cat(result_offset_err, dim=0)
    result_static_ome = torch.cat(result_static_ome, dim=0)
    result_dynamic_ome = torch.cat(result_dynamic_ome, dim=0)

    result_static_ame = torch.cat(result_static_ame, dim=0)
    result_dynamic_ame = torch.cat(result_dynamic_ame, dim=0)
    print('=====Results=====')
    print('R_DG (drift) Err')
    print(result_drift_err.mean(dim=0), '|', result_drift_err.mean())

    print('R_BS (offset) Err')
    print(result_offset_err.mean(dim=0), '|', result_offset_err.mean())

    print('OME-static')
    print(result_static_ome.mean(dim=0), '|', result_static_ome.mean())

    print('OME-dynamic')
    print(result_dynamic_ome.mean(dim=0), '|', result_dynamic_ome.mean())

    print('AME-static')
    print(result_static_ame.mean(dim=0), '|', result_static_ame.mean())

    print('AME-dynamic')
    print(result_dynamic_ame.mean(dim=0), '|', result_dynamic_ame.mean())

