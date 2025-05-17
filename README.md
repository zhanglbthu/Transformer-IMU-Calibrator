<div align="center">

<h1>Transformer IMU Calibrator: Dynamic On-body IMU Calibration for Inertial Motion Capture</h1>

<div>
    <a target='_blank'>Chengxu Zuo<sup>1</sup></a>&emsp;
    <a target='_blank'>Jiawei Huang<sup>1</sup></a>&emsp;
    <a target='_blank'>Xiao Jiang<sup>1</sup></a>&emsp;
    <a target='_blank'>Yuan Yao<sup>1</sup></a>&emsp;
    <a target='_blank'>Xiangren Shi<sup>4</sup></a>&emsp;
    <a target='_blank'>Rui Cao<sup>1</sup></a>&emsp;
    <a target='_blank'>Yinyu Yi<sup>2</sup></a>&emsp;
    <a target='_blank'>Feng Xu<sup>2</sup></a>&emsp;
    <a target='_blank'>Shihui Guo<sup>1*</sup></a>&emsp;
    <a target='_blank'>Yipeng Qin<sup>3</sup></a>&emsp;
</div>
<div>
    <sup>1</sup>Xiamen University &nbsp; <sup>2</sup>Tsinghua University &nbsp; <sup>3</sup>Cardiff University &nbsp; <sup>4</sup>Bournemouth University
</div>

<div>
    <sup>*</sup>Corresponding Author
</div>

<div>
    <strong>Accepted to SIGGRAPH 2025</strong>
</div>

<h4 align="center">

[//]: # (<a href="https://arxiv.org/abs/2312.02196" target='_blank'>[Paper]</a> •)

[//]: # (<a href="https://www.youtube.com/watch?v=88_CyBNtEe8&t=168s" target='_blank'>[Demo]</a>)
  
</h4>

</div>

![](figs/teaser.jpg)
Implementation of our SIGGRAPH 2025 paper "Transformer IMU Calibrator: Dynamic On-body IMU Calibration for Inertial Motion Capture". Including network weights, training and evaluation scripts.

[train.py](./train.py): TIC Network training.

[eval.py](./eval.py): Run our dynamic calibration on dataset and calculate OME, AME and R_G'G/R_BS Error.

## Synthesized Dataset for training

Coming Soon.

[//]: # (1. Download required training data at xxxx.)

[//]: # (2. Copy all data in folder: [root/data_train])

[//]: # ()
[//]: # (*Note: We expand synthesized head IMU acc data with 14 different vertices on head mesh &#40;head_acc.pt&#41;, thus covering acc variances on different IMU location when rotating head.)


## TIC Dataset
The TIC dataset is available at https://www.dropbox.com/scl/fo/ggrvm8x2xjhu1m0pjomc9/ADClW3gbt4swggoulhndBKA?rlkey=bagguhrnze7fdvgr2toggce0v&st=p3fj8g1e&dl=0.

The data was collected from 5 subjects (s1~s5).
For each subject, the dataset provides:
1. **acc.pt**----Acceleration of 6 on-body IMU, calibrated by static calibration at begin.
2. **rot.pt**----Orientation of 6 on-body IMU, calibrated by static calibration at begin.
3. **pose.pt**----SMPL pose captured by NOKOV System (use optical tracker).
4. **trans.pt**----Global body translation (location).
5. **drift.pt**----Absolute coordinate drift of 6 on-body IMU.
6. **offset.pt**----Measurement offset of 6 on-body IMU.
7. **acc_gt.pt**----GT IMU acceleration captured by NOKOV System (use optical tracker).

*Note 1: IMU order: left forearm, right forearm, left lower leg, right lower leg, head, hip

*Note 2: All data are in SMPL frame. 

## Acknowledgement

Some of our codes are adapted from [PIP](https://github.com/Xinyu-Yi/PIP).
The SMPL_MALE model is download from https://smpl.is.tue.mpg.de/.

[//]: # (## Citation)

[//]: # ()
[//]: # (If you find this project helpful, please consider citing us:)
