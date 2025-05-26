from my_data import *
from my_trainner import *
from my_model import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = IMUData.load_data(folder_path=paths.amass_dir, step=2)

data_len = len(dataset['imu_rot'])

data_train = IMUData(rot=dataset['imu_rot'], acc=dataset['imu_acc'], seg_info=dataset['seg_info'],
                     head_acc=None, seq_len=256)

model = TIC(stack=3, n_input=config.imu_num*(3+3*3), n_output=config.imu_num*6).to(device)

optimizer = [torch.optim.Adam(model.parameters(), lr=1e-3)]
trainner = TicTrainner(model=model, data=data_train, optimizer=optimizer, batch_size=128)

# trainner.restore(checkpoint_path='checkpoint/TIC_13.pth', load_optimizer=True)
epoch=20
model_name = f'TIC'
print(model_name)
for i in range(epoch):
    trainner.run(epoch=1, data_shuffle=True)
    trainner.save(folder_path='./checkpoint', model_name=model_name)
    trainner.log_export(f'./log/{model_name}.xlsx')