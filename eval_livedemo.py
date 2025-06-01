import articulate as art
import config
from my_model import *
from TicOperator import *
from evaluation_functions import *

import os

imu_num = config.imu_num
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = TIC(stack=3, n_input=imu_num * (3 + 3 * 3), n_output=imu_num * 6)
model.restore('./checkpoint/TIC_13.pth')
model = model.to(device).eval()

tag = 'TICnew_woRD'
folders = ['2imu_sit_20250526_135237']
combo = [0, 3]
# folders = ['s1']

# Inference
if True:
    print('=====Inference Start=====')
    for f in folders:
        data_root = os.path.join(config.paths.livedemo_dataset_dir, f)
        os.makedirs(data_root, exist_ok=True)
        print(f'processing {f}')

        data = torch.load(os.path.join(config.paths.livedemo_dataset_dir, f +'.pt'))
        
        imu_acc = data['acc'][:, combo]
        imu_rot = data['ori'][:, combo]

        ts = TicOperator(TIC_network=model, imu_num=imu_num, data_frame_rate=30)
        rot, acc, pred_drift, pred_offset = ts.run(imu_rot, imu_acc, trigger_t=1)

        torch.save(imu_acc, os.path.join(data_root, f'acc.pt'))
        torch.save(imu_rot, os.path.join(data_root, f'rot.pt'))
        torch.save(acc, os.path.join(data_root, f'acc_fix_{tag}.pt'))
        torch.save(rot, os.path.join(data_root, f'rot_fix_{tag}.pt'))
        torch.save(pred_drift, os.path.join(data_root, f'pred_drift_{tag}.pt'))
        torch.save(pred_offset, os.path.join(data_root, f'pred_offset_{tag}.pt'))