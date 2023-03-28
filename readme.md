# Pedestrian-Retrograde-Detection-in-Video-Surveillance

视频监控中的逆行检测
===========================

###########环境依赖
Python 3.7

Opencv3.4.3

Tensorflow2.0

###########目录结构描述
├── Readme.md                   // help
├── src                     // 源代码
│   ├── Optical Flow Method       //传统方法（光流法）
│   ├── YOLO Method                // Yolo方法（精检测方案二）
│   ├── Deep Learning Method         // 深度学习自编码器方法
├── dataset                 //数据集和采集原视频

│   ├── SCUT_HEAD_Part_A       //YOLO方法用数据集
│   ├── mydataset                // 深度学习方法用数据集

├── model.hdf5                 //深度学习方法神经网络模型

├── lib                         // 转换格式与视频测试用工具

├── results                         // 存放结果

│   ├── Optical Flow Method       //传统方法（光流法）结果
│   ├── YOLO Method                // Yolo方法（精检测方案二）结果
│   ├── Deep Learning Method         // 深度学习自编码器方法结果

├── image                       // 存放原理图





@Copyright wky# Pedestrian-Retrograde-Detection-in-Video-Surveillance---Research-and-Engineering-Practice

2022-SEU-Automation

# 
