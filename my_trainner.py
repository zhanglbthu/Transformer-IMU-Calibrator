import config
from Aplus.runner import *
from articulate.evaluator import RotationErrorEvaluator, PerJointRotationErrorEvaluator, PerJointAccErrorEvaluator
from articulate.math.angular import *
from tqdm import tqdm
from Aplus.data.process import *

class TicTrainner(BaseTrainer):
    def __init__(self, model: nn.Module, data, optimizer, batch_size):
        """
        Used for manage training process.
        Args:
            model: Your model.
            data: Dataset object. You can build dataset via Aplus.data.BaseDataset
            optimizer: Model's optimizer.
            batch_size: /
            loss_func: /
        """
        self.model = model
        self.optimizer = optimizer
        self.MSE = nn.MSELoss()

        self.data = data
        self.epoch = 0
        self.batch_size = batch_size
        self.log_manager = LogManager(items=['epoch', 'loss_train', 'err_drift', 'err_offset'])

        rep = RotationRepresentation.ROTATION_MATRIX
        self.rot_err_evaluator = RotationErrorEvaluator(rep=rep)
        self.per_joint_rot_err_evaluator = PerJointRotationErrorEvaluator(rep=rep)
        self.imu_num = config.imu_num
        self.checkpoint = None


    def run(self, epoch, data_shuffle=True, evaluator=None, noise_sigma=None):
        from simulations import imu_drift_offset_simulation, imu_offset_simulation

        # 获取当前模型所在device
        device = self.get_model_device()

        # AverageMeter用于计算整个epoch的loss
        avg_meter_loss = DataMeter()
        avg_meter_angle_drift = DataMeter()
        avg_meter_angle_offset = DataMeter()

        for e in range(epoch):
            data_loader = DataLoader(dataset=self.data, batch_size=self.batch_size, shuffle=data_shuffle,
                                     drop_last=False)
            # AverageMeter需要在每个epoch开始时置0
            avg_meter_loss.reset()
            avg_meter_angle_drift.reset()
            avg_meter_angle_offset.reset()
            self.model.train()

            for data in tqdm(data_loader):
                rot, acc = data
                rot = rot.to(device) # [128, 256, 2, 3, 3]
                acc = acc.to(device) # [128, 256, 2, 3, 1]


                # rot, acc, drift, offset = imu_drift_offset_simulation(imu_rot=rot, imu_acc=acc, imu_num=config.imu_num,
                #                                                       ego_imu_id=-1, drift_range=60,
                #                                                       offset_range=45, random_global_yaw=True)
                rot, acc, drift, offset = imu_offset_simulation(imu_rot=rot, imu_acc=acc, imu_num=config.imu_num)

                rot, acc, drift, offset = rot.flatten(2), acc.flatten(2), drift.flatten(1), offset.flatten(1)
                # rot: [128, 256, 18]; acc: [128, 256, 6]; drift: [128, 12]; offset: [128, 12]
                acc /= 30

                x = torch.cat([acc, rot], dim=-1)

                self.optimizer[0].zero_grad()

                drift_hat, offset_hat = self.model(x)
                loss = self.MSE(drift_hat, drift) + self.MSE(offset_hat, offset)

                loss.backward()

                self.optimizer[0].step()
                # ====================

                # 把估计的旋转转置后乘回去，计算与单位阵的角度误差，即校正后的残留误差
                drift = r6d_to_rotation_matrix(drift.reshape(-1, 6)).reshape(-1, self.imu_num, 3, 3)
                drift_hat = r6d_to_rotation_matrix(drift_hat.detach().reshape(-1, 6)).reshape(-1, self.imu_num, 3, 3)
                offset = r6d_to_rotation_matrix(offset.reshape(-1, 6)).reshape(-1, self.imu_num, 3, 3)
                offset_hat = r6d_to_rotation_matrix(offset_hat.detach().reshape(-1, 6)).reshape(-1, self.imu_num, 3, 3)

                # 每个batch记录一次

                avg_meter_loss.update(value=loss.item(), n_sample=len(x))

                ang_err_drift = self.per_joint_rot_err_evaluator(p=drift_hat, t=drift, joint_num=config.imu_num).cpu()
                avg_meter_angle_drift.update(value=ang_err_drift, n_sample=len(drift))

                ang_err_offset = self.per_joint_rot_err_evaluator(p=offset_hat, t=offset,
                                                                     joint_num=config.imu_num).cpu()
                avg_meter_angle_offset.update(value=ang_err_offset, n_sample=len(offset))


            # 获取整个epoch的loss
            loss_train = avg_meter_loss.get_avg()
            err_drift = avg_meter_angle_drift.get_avg()
            err_offset = avg_meter_angle_offset.get_avg()
            self.epoch += 1
            print('')

            self.log_manager.update(
                {'epoch': self.epoch, 'loss_train': loss_train, 'err_drift': err_drift.mean(), 'err_offset': err_offset.mean()})

            # 打印最新一个epoch的训练记录
            self.log_manager.print_latest()


