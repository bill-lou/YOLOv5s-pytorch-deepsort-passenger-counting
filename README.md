# -pytorch+YOLOv5s+deepsort进行行人跟踪-
# -Pedestrian tracking with pytorch+YOLOv5s+deepsort-

#配置环境
#configure the environment

在cmd/终端里切到文件下载到的目录，输入

pip install -U -r requirements.txt

python3 track.py --source 视频文件的名字（比如1.mp4）

视频应该提前放到track.py同一路径下面

这里是已经放好了模型，在yolo那个文件夹。
原项目地址：https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch
