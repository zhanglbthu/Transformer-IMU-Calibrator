import articulate as art
import torch
import os
import pickle
from config import paths, amass_data
import numpy as np
from tqdm import tqdm
import glob
from articulate.math.angular import RotationRepresentation
from articulate.evaluator import RotationErrorEvaluator, PerJointRotationErrorEvaluator
from Aplus.utils.data_meter import DataMeter

def process_amass(smooth_n=3):

    def _syn_acc(v, smooth_n=smooth_n):
        r"""
        Synthesize accelerations from vertex positions.
        """
        mid = smooth_n // 2
        acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * 3600 for i in range(0, v.shape[0] - 2)])
        acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
        if mid != 0:
            acc[smooth_n:-smooth_n] = torch.stack(
                [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * 3600 / smooth_n ** 2
                 for i in range(0, v.shape[0] - smooth_n * 2)])
        return acc

    # vi_mask = torch.tensor([1961, 5424, 1176, 4662, 411, 3021])
    vi_mask = torch.tensor([1961, 5424, 876, 4362, 411, 3021])
    # ji_mask = torch.tensor([18, 19, 4, 5, 15, 0])
    ji_mask = torch.tensor([18, 19, 1, 2, 15, 0])

    body_model = art.ParametricModel(paths.smpl_file)

    data_pose, data_trans, data_beta, length = [], [], [], []
    for ds_name in amass_data:
        print('\rReading', ds_name)
        for npz_fname in tqdm(glob.glob(os.path.join(paths.raw_amass_dir, ds_name, '*/*_poses.npz'))):
            try: cdata = np.load(npz_fname)
            except: continue

            framerate = int(cdata['mocap_framerate'])
            if framerate == 120: step = 2
            elif framerate == 60 or framerate == 59: step = 1
            else: continue

            data_pose.extend(cdata['poses'][::step].astype(np.float32))
            data_trans.extend(cdata['trans'][::step].astype(np.float32))
            data_beta.append(cdata['betas'][:10])
            # print(cdata['betas'][:10])
            length.append(cdata['poses'][::step].shape[0])

    assert len(data_pose) != 0, 'AMASS dataset not found. Check config.py or comment the function process_amass()'
    length = torch.tensor(length, dtype=torch.int)
    shape = torch.tensor(np.asarray(data_beta, np.float32))
    # print(shape)
    tran = torch.tensor(np.asarray(data_trans, np.float32))
    # print(tran)
    pose = torch.tensor(np.asarray(data_pose, np.float32)).view(-1, 52, 3)
    pose[:, 23] = pose[:, 37]     # right hand
    pose = pose[:, :24].clone()   # only use body

    # align AMASS global fame with DIP
    amass_rot = torch.tensor([[[1, 0, 0],
                               [0, 0, 1],
                               [0, -1, 0.]]])
    tran = amass_rot.matmul(tran.unsqueeze(-1)).view_as(tran)
    # print(tran)
    pose[:, 0] = art.math.rotation_matrix_to_axis_angle(
        amass_rot.matmul(art.math.axis_angle_to_rotation_matrix(pose[:, 0])))

    print('Synthesizing IMU accelerations and orientations')
    b = 0
    out_pose, out_shape, out_tran, out_joint, out_vrot, out_vacc = [], [], [], [], [], []

    for i, l in tqdm(list(enumerate(length))):
        if l <= 12: b += l; print('\tdiscard one sequence with length', l); continue
        p = art.math.axis_angle_to_rotation_matrix(pose[b:b + l]).view(-1, 24, 3, 3)
        # 输入24个关节旋转+体型参数+位移信息, 输出24个关节的旋转+蒙皮点加速度+运动速度
        # print(p.shape, shape[i].shape, tran[b:b + l].shape)
        grot, joint, vert = body_model.forward_kinematics(p, shape[i], tran[b:b + l], calc_mesh=True)

        out_pose.append(pose[b:b + l].clone())  # N, 24, 3
        out_tran.append(tran[b:b + l].clone())  # N, 3
        out_shape.append(shape[i].clone())  # 10
        out_joint.append(joint[:, :24].contiguous().clone())  # N, 24, 3
        out_vacc.append(_syn_acc(vert[:, vi_mask]))  # N, 6, 3
        out_vrot.append(grot[:, ji_mask])  # N, 6, 3, 3
        b += l

    print('Saving')
    os.makedirs(paths.amass_dir, exist_ok=True)
    torch.save(out_pose, os.path.join(paths.amass_dir, 'pose.pt'))
    torch.save(out_shape, os.path.join(paths.amass_dir, 'shape.pt'))
    torch.save(out_tran, os.path.join(paths.amass_dir, 'tran.pt'))
    torch.save(out_joint, os.path.join(paths.amass_dir, 'joint.pt'))
    torch.save(out_vrot, os.path.join(paths.amass_dir, 'vrot.pt'))
    torch.save(out_vacc, os.path.join(paths.amass_dir, f'vacc.pt'))
    print('Synthetic AMASS dataset is saved at', paths.amass_dir)

def amass_head_acc_syn(smooth_n=3):

    def _syn_acc(v, smooth_n=smooth_n):
        r"""
        Synthesize accelerations from vertex positions.
        """
        mid = smooth_n // 2
        acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * 3600 for i in range(0, v.shape[0] - 2)])
        acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
        if mid != 0:
            acc[smooth_n:-smooth_n] = torch.stack(
                [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * 3600 / smooth_n ** 2
                 for i in range(0, v.shape[0] - smooth_n * 2)])
        return acc

    # vi_mask = torch.tensor([1961, 5424, 1176, 4662, 411, 3021])
    vi_mask = torch.tensor([1961, 5424, 876, 4362, 411, 3021])
    head_vi_mask = torch.tensor([133,134,159,169,254,271,335,385,3645,3646,3673,3679,3764,3784])

    body_model = art.ParametricModel(paths.smpl_file)

    data_pose, data_trans, data_beta, length = [], [], [], []
    for ds_name in amass_data:
        print('\rReading', ds_name)
        for npz_fname in tqdm(glob.glob(os.path.join(paths.raw_amass_dir, ds_name, '*/*_poses.npz'))):
            try: cdata = np.load(npz_fname)
            except: continue

            framerate = int(cdata['mocap_framerate'])
            if framerate == 120: step = 2
            elif framerate == 60 or framerate == 59: step = 1
            else: continue

            data_pose.extend(cdata['poses'][::step].astype(np.float32))
            data_trans.extend(cdata['trans'][::step].astype(np.float32))
            data_beta.append(cdata['betas'][:10])
            length.append(cdata['poses'][::step].shape[0])

    assert len(data_pose) != 0, 'AMASS dataset not found. Check config.py or comment the function process_amass()'
    length = torch.tensor(length, dtype=torch.int)
    shape = torch.tensor(np.asarray(data_beta, np.float32))
    # print(shape)
    tran = torch.tensor(np.asarray(data_trans, np.float32))
    # print(tran)
    pose = torch.tensor(np.asarray(data_pose, np.float32)).view(-1, 52, 3)
    pose[:, 23] = pose[:, 37]     # right hand
    pose = pose[:, :24].clone()   # only use body

    # align AMASS global fame with DIP
    amass_rot = torch.tensor([[[1, 0, 0],
                               [0, 0, 1],
                               [0, -1, 0.]]])
    tran = amass_rot.matmul(tran.unsqueeze(-1)).view_as(tran)
    # print(tran)
    pose[:, 0] = art.math.rotation_matrix_to_axis_angle(
        amass_rot.matmul(art.math.axis_angle_to_rotation_matrix(pose[:, 0])))

    print('Synthesizing IMU accelerations and orientations')
    b = 0
    out_vacc = []
    out_head_vacc = []

    for i, l in tqdm(list(enumerate(length))):
        if l <= 12: b += l; print('\tdiscard one sequence with length', l); continue
        p = art.math.axis_angle_to_rotation_matrix(pose[b:b + l]).view(-1, 24, 3, 3)
        # 输入24个关节旋转+体型参数+位移信息, 输出24个关节的旋转+蒙皮点加速度+运动速度
        # print(p.shape, shape[i].shape, tran[b:b + l].shape)
        grot, joint, vert = body_model.forward_kinematics(p, shape[i], tran[b:b + l], calc_mesh=True)
        out_vacc.append(_syn_acc(vert[:, vi_mask]))  # N, 6, 3
        out_head_vacc.append(_syn_acc(vert[:, head_vi_mask]))  # N, 6, 3

        b += l

    print('Saving')
    os.makedirs(paths.amass_dir, exist_ok=True)
    torch.save(out_vacc, os.path.join(paths.amass_dir, f'vacc.pt'))
    torch.save(out_head_vacc, os.path.join(paths.amass_dir, f'vacc_head14.pt'))
    print('Synthetic AMASS dataset is saved at', paths.amass_dir)

def process_dipimu():
    # "head", "sternum", "pelvis", "lshoulder", "rshoulder", \
    #     "lupperarm", "rupperarm", "llowerarm", "rlowerarm", \
    #     "lupperleg", "rupperleg", "llowerleg", "rlowerleg", \
    #     "lhand", "rhand", "lfoot", "rfoot"
    # imu_mask = [7, 8, 3, 2, 11, 12, 2, 2]
    # imu_mask = [7, 8, 3, 2]
    imu_mask = [7, 8, 11, 12, 0, 2]
    test_split = ['s_09', 's_10']
    # test_split = [f's_0{i+1}' for i in range(8)]
    accs, oris, poses, trans = [], [], [], []

    for subject_name in test_split:
        for motion_name in os.listdir(os.path.join(paths.raw_dipimu_dir, subject_name)):
            path = os.path.join(paths.raw_dipimu_dir, subject_name, motion_name)
            data = pickle.load(open(path, 'rb'), encoding='latin1')
            acc = torch.from_numpy(data['imu_acc'][:, imu_mask]).float()
            ori = torch.from_numpy(data['imu_ori'][:, imu_mask]).float()
            pose = torch.from_numpy(data['gt']).float()

            # fill nan with nearest neighbors
            for _ in range(4):
                acc[1:].masked_scatter_(torch.isnan(acc[1:]), acc[:-1][torch.isnan(acc[1:])])
                ori[1:].masked_scatter_(torch.isnan(ori[1:]), ori[:-1][torch.isnan(ori[1:])])
                acc[:-1].masked_scatter_(torch.isnan(acc[:-1]), acc[1:][torch.isnan(acc[:-1])])
                ori[:-1].masked_scatter_(torch.isnan(ori[:-1]), ori[1:][torch.isnan(ori[:-1])])

            acc, ori, pose = acc[6:-6], ori[6:-6], pose[6:-6]
            if torch.isnan(acc).sum() == 0 and torch.isnan(ori).sum() == 0 and torch.isnan(pose).sum() == 0:
                accs.append(acc.clone())
                oris.append(ori.clone())
                poses.append(pose.clone())
                trans.append(torch.zeros(pose.shape[0], 3))  # dip-imu does not contain translations
            else:
                print('DIP-IMU: %s/%s has too much nan! Discard!' % (subject_name, motion_name))

    os.makedirs(paths.dipimu_dir, exist_ok=True)
    torch.save({'acc': accs, 'ori': oris, 'pose': poses, 'tran': trans}, os.path.join(paths.dipimu_dir, 'test.pt'))
    print('Preprocessed DIP-IMU dataset is saved at', paths.dipimu_dir)


if __name__ == '__main__':
    # process_amass(smooth_n=3)
    amass_head_acc_syn(smooth_n=3)
    # process_dipimu()
