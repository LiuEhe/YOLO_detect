# yolov8目标检测APP


## 项目描述

本项目是一个基于Tkinter和OpenCV的目标检测应用程序，实现了摄像头和视频文件的实时目标检测。通过YOLOv8模型进行目标检测，支持定位、分割和姿势三种模型类型，以及不同模型大小。

## 项目运行效果截图

<img src="https://github.com/LiuEhe/YOLO_detect/blob/main/result/1.jpg" width="400" height="220.5"><img src="https://github.com/LiuEhe/YOLO_detect/blob/main/result/2.jpg" width="400" height="220.5"><img src="https://github.com/LiuEhe/YOLO_detect/blob/main/result/3.jpg" width="400" height="220.5">

## 功能
- 支持摄像头和视频文件的实时目标检测
- 支持定位、分割和姿势三种模型类型
- 支持不同模型大小
- 支持在视频上显示边界框和遮罩
- 支持暂停、播放和重新播放视频文件
- 支持目标检测的开始/停止

## 依赖

- Python 3
- OpenCV
- Tkinter
- ultralytics YOLO

## 使用

1. 克隆项目到本地
2. 创建conda虚拟环境
3. 安装依赖
4. 运行项目 `python obj_tkinterapp.py`


## 注意
- 要解压`video_imgs.zip`
- 要自行下载`yolov8`的预训练权重文件
- 确保摄像头可用且没有被其他应用程序占用
- 确保视频文件格式正确且路径合法
- 请自行创建weights文件夹
- 在选择模型类型和大小时，确保模型文件存在于项目目录的“weights”文件夹下

