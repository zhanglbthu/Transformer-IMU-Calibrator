import torch

imu_num = 6
unit_r6d = torch.FloatTensor([[1, 0, 0, 0, 1, 0]])

amass_data = ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh', 'Transitions_mocap', 'SSM_synced', 'CMU',
              'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'BMLmovi', 'EKUT', 'TCD_handMocap', 'ACCAD',
              'BioMotionLab_NTroje', 'BMLhandball', 'MPI_Limits', 'DFaust67']

class joint_set:
    joint_name_list = ["pelvis", "l_hip", "r_hip", "spine1", "l_knee", "r_knee", "spine2", "l_ankle", "r_ankle",
                       "spine3", "l_toe", "r_toe", "neck", "l_collar", "r_collar", "head", "l_shoulder", "r_shoulder",
                       "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_palm", "r_palm"]


class paths:
    imu4_dir = 'E:\DATA\processed_data\\'  # 采集的数据的路径

    raw_amass_dir = '/root/autodl-tmp/data/AMASS'      # raw AMASS dataset path (raw_amass_dir/ACCAD/ACCAD/s001/*.npz)
    amass_dir = '/root/autodl-tmp/data/AMASS_IMU6'  # output path for the synthetic AMASS dataset

    raw_dipimu_dir = 'E:\DATA\DIP-IMU\DIP_IMU'   # raw DIP-IMU dataset path (raw_dipimu_dir/s_01/*.pkl)
    dipimu_dir = 'E:\DATA\DIP_IMU6'      # output path for the preprocessed DIP-IMU dataset

    tic_dataset_dir = '/root/autodl-tmp/data/TIC_Dataset'

    smpl_file = './SMPL_MALE.pkl'  # official SMPL model path