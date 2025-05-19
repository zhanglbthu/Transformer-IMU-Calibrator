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
folders = ['test20250518134839']
# folders = ['s1']

# Inference
if True:
    print('=====Inference Start=====')
    for f in folders:
        data_root = os.path.join(config.paths.livedemo_dataset_dir, f)
        os.makedirs(data_root, exist_ok=True)
        print(f'processing {f}')

        data = torch.load(os.path.join(config.paths.livedemo_dataset_dir, f +'.pt'))
        
        imu_acc = data['acc']
        imu_rot = data['ori']

        ts = TicOperator(TIC_network=model)
        # ts = TicOperatorOverwrite(TIC_network=model)
        rot, acc, pred_drift, pred_offset = ts.run(imu_rot, imu_acc, trigger_t=1)

        torch.save(imu_acc, os.path.join(data_root, f'acc.pt'))
        torch.save(imu_rot, os.path.join(data_root, f'rot.pt'))
        torch.save(acc, os.path.join(data_root, f'acc_fix_{tag}.pt'))
        torch.save(rot, os.path.join(data_root, f'rot_fix_{tag}.pt'))
        torch.save(pred_drift, os.path.join(data_root, f'pred_drift_{tag}.pt'))
        torch.save(pred_offset, os.path.join(data_root, f'pred_offset_{tag}.pt'))