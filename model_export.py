import torch

from my_model import *
import torch.nn.functional as F
from Aplus.tools.smpl_light import SMPLPose


if __name__ == '__main__':
    torch.no_grad()
    model_s1 = EasyLSTM(n_input=config.imu_num * 6 * 1 + config.imu_num * 3, n_hidden=256,
                        n_output=joint_set.endpoint_num * 3,
                        n_lstm_layer=2, bidirectional=False, output_type='seq', dropout=0, layer_norm=False,
                        act_func='leakyrelu')
    model_s2 = EasyLSTM(n_input=config.imu_num * 6 * 1 + config.imu_num * 3 + joint_set.endpoint_num * 3, n_hidden=256,
                        n_output=joint_set.endpoint_num * 3, n_lstm_layer=2, bidirectional=False, output_type='seq',
                        layer_norm=False, dropout=0, act_func='leakyrelu')
    model_s3 = EasyLSTM(n_input=config.imu_num * 6 * 2 + config.imu_num * 3 + joint_set.endpoint_num * 3, n_hidden=256,
                        n_output=joint_set.joint_num * 6, n_lstm_layer=2, bidirectional=False,
                        output_type='seq', dropout=0, layer_norm=False, act_func='leakyrelu')
    # --------合练--------
    # print(joint_set.joint_num * 6)
    model_name = 'LIPX'
    model = LIPX(net_s1=model_s1, net_s2=model_s2, net_s3=model_s3)
    #
    model.restore(checkpoint_path='./checkpoint/LIPX_10.pth')
    #
    poser = Poser(net=model, input_type='rotation_matrix', type='axis_angle')

    n_layer = 2

    poser.export_onnx(input_shapes={'imu_data': [-1, 12*config.imu_num], 'h_1': [n_layer, -1, 256], 'c_1': [n_layer, -1, 256],
                                    'h_2': [n_layer, -1, 256], 'c_2': [n_layer, -1, 256], 'h_3': [n_layer, -1, 256], 'c_3': [n_layer, -1, 256]},
                      output_shapes={'pose': [-1, 24, 3], 'joint': [-1, config.joint_set.endpoint_num, 3], 'vel': [-1, config.joint_set.endpoint_num, 3], 'h_1_n': [n_layer, -1, 256], 'c_1_n': [n_layer, -1, 256],
                                    'h_2_n': [n_layer, -1, 256], 'c_2_n': [n_layer, -1, 256], 'h_3_n': [n_layer, -1, 256], 'c_3_n': [n_layer, -1, 256]}, path='onnx_model/Poser_6imu.onnx')


    # 导出初始状态
    initializer_v = DualInitiallizer(n_input=joint_set.endpoint_num * 3, layer_num=2, hidden_size=256)
    initializer_p = DualInitiallizer(n_input=joint_set.endpoint_num * 3, layer_num=2, hidden_size=256)
    initializer_r = DualInitiallizer(n_input=joint_set.joint_num * 6, layer_num=2, hidden_size=256)
    model_name = 'LIPX'
    initializer_v.restore(checkpoint_path=f'./checkpoint/init_v_{model_name}_{10}.pth')
    initializer_p.restore(checkpoint_path=f'./checkpoint/init_p_{model_name}_{10}.pth')
    initializer_r.restore(checkpoint_path=f'./checkpoint/init_r_{model_name}_{10}.pth')

    input_v = torch.zeros(1, joint_set.endpoint_num * 3)
    tpose_joint = SMPLPose.t_pose_joint
    tpose_rot = SMPLPose.t_pose_ori

    input_p = tpose_joint[joint_set.index_joint].flatten(0).unsqueeze(0)
    input_r = tpose_rot[joint_set.index_pose]
    input_r = rotation_matrix_to_r6d(input_r).flatten(0).unsqueeze(0)

    h_v, c_v = initializer_v(input_v)
    h_p, c_p = initializer_p(input_p)
    h_r, c_r = initializer_r(input_r)

    torch.save([h_v, c_v, h_p, c_p, h_r, c_r], 'onnx_model/Poser_6imu_init_state.pt')



    # # 导出calibrator
    # model = TIC(stack=3, n_input=6 * (3 + 3 * 3), n_output=6 * 6).eval()
    # # model = TIC(stack=2, n_input=6 * (3 + 3 * 3), n_output=6 * 6).eval()
    # # model.restore('./checkpoint/TIC_acc_sota3.pth')
    # # model.restore('./checkpoint/TIC_no_acc_10.pth')
    # model.restore('./checkpoint/TIC_ft_t1_11.pth')
    # # model.export_onnx(input_shapes={'imu_rot': [-1, 128, 72]}, output_shapes={'global_shift':[1, 36], 'local_shift':[1, 36], 'confidence':[1, 6]}, path='TIC_6imu_no_prior.onnx')
    # model.export_onnx(input_shapes={'imu_rot': [1, -128, 72]}, output_shapes={'global_shift':[1, 36], 'local_shift':[1, 36]}, path='./onnx_model/TIC_6imu_acc.onnx')




