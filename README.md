# -pytorch+YOLOv5s+deepsort进行行人跟踪-
# -Pedestrian tracking with pytorch+YOLOv5s+deepsort-

#配置环境
#configure the environment

在cmd/终端里切到文件下载到的目录，输入

pip install -r requirements.txt

# Counting Line + YOLO Standard + Deep Sort Standard
python3 count.py --deepsort 1
# Counting Line + YOLO Custom + Deep Sort Custom
python3 count.py --deepsort 2 

or

# same for track without counting - use 1 or 2 as for count
python3 track.py --deepsort 2


这里是已经放好了模型，在yolo那个文件夹。
原项目地址：https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch
