# 引入YOLO模型用于图像目标检测
from ultralytics import YOLO
# matplotlib.pyplot 用于显示和渲染图像
import matplotlib.pyplot as plt
# cv2 用于处理和转化图像格式
import cv2
# ipywidgets 提供了交互性组件，如滑动条和复选框等
from ipywidgets import interact, FloatSlider, Checkbox

# 加载YOLO模型，进行目标检测
model = YOLO('./weights/yolov8s.pt')

# 定义一个函数，接受阈值参数和标签选项，进行目标检测并绘图显示结果
def predict_and_plot(conf, iou, boxes, masks):
    # 根据输入参数，对指定图片进行目标检测，返回检测结果
    results = model.predict(source="./assets/people_walking.jpg", conf=conf, iou=iou, verbose=False)
    # 根据检测结果和输入参数，生成标注后的图片
    res_plotted = results[0].plot(boxes=boxes, masks=masks)
    # 将BGR格式的图像转为RGB格式，以便matplotlib正确显示
    res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
    plt.imshow(res_plotted_rgb) # 使用matplotlib显示图片
    plt.show()

# 创建滑动条和复选框，用于动态调整参数和选项，并显示目标检测结果
interact(predict_and_plot, 
         conf=FloatSlider(min=0, max=1, step=0.01, value=0.25),  # 置信度阈值滑动条
         iou=FloatSlider(min=0, max=1, step=0.01, value=0.7),  # 交并比阈值滑动条
         boxes=Checkbox(value=True, description='Boxes'),  # 是否显示目标框复选框
         masks=Checkbox(value=False, description='Masks'));  # 是否显示目标掩码复选框
