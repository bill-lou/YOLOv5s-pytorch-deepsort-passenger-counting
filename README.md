# YOLOv5 + Deep Sort Passanger Counter by Heads 
A computer vision and deep learning project for counting passengers using head recognition.

This project was developed by both Swiss and Chinese students as part of a joint venture between University of Applied Science Northwestern Switzerland FHNW and Shenzhen Technology University SZTU.


| Standard      | Custom        | 
| ------------- |:-------------:| 
| <img src="https://github.com/bill-lou/YOLOv5s-pytorch-deepsort-passenger-counting/blob/main/sample/standard.png" width="400" height="273">     | <img src="https://github.com/bill-lou/YOLOv5s-pytorch-deepsort-passenger-counting/blob/main/sample/custom.png" width="400" height="273"> | 

The main task includes to count passenger flows in a metro station. Therefore YOLOv5 and Deep Sort as well as a counting mechanism were merged (from already existing Repos). In addition, new weights of both YOLO and the feature extractor of Deep Sort were trained using custom datasets.

The following repos were adapted and reused for this purpose:

[YOLOv5 + Deep Sort Baseline Implementation](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)

[Setup + Training of a Siamese Network for Custom Head Tracking](https://github.com/abhyantrika/nanonets_object_tracking)

[YOLOv5](https://github.com/ultralytics/yolov5)

[Deep Sort](https://github.com/nwojke/deep_sort)

[Deep Sort Custom Head Dataset](https://www.di.ens.fr/willow/research/headdetection/)

[Dataset Prep](https://github.com/giovannicimolin/PascalVOC-to-Images/blob/master/main.py)

# Installation

First, clone the repository: 
```
git clone git@github.com:bill-lou/YOLOv5s-pytorch-deepsort-passenger-counting.git
```

Use this command to install all the necessary packages. Note that we are using `python3`
```
pip install -r requirements.txt
```

# Configuration
There are two wheights available for YOLOv5, make sure to adjust the parser argument in `count.py` befor running the code 

**You can Download the Checkpoints from:**

[Standard YOLO](https://github.com/ultralytics/yolov5/releases)

[Custom Head](https://drive.google.com/drive/folders/1NY_uZuHOqogKOk49UVEyyXlODtRBGy4z?usp=sharing)

**yolov5x.pt** for standard YOLO Detection
```python
parser.add_argument('--weights',
                    type=str,
                    default='yolov5/weights/yolov5x.pt',
                    help='model.pt path')
```

**custom_train_best.pt** for Custom Head Detection
```python
parser.add_argument('--weights',
                    type=str,
                    default='yolov5/weights/custom_train_best.pt',
                    help='model.pt path')
```

Download as well the Deep Sort Checkpoints  and add it to the folder `deep_sort_pytorch\deep_sort\deep\checkpoint`

**You can Download the Checkpoints from:**

[Standard Deep Sort](https://drive.google.com/drive/folders/18fKzfqnqhqW3s9zwsCbnVJ5XF2JFeqMp)

[Custom Head Deep Sort](https://drive.google.com/drive/folders/1hshpI1gaMw2Uoz9hnO10_j3rqcmClnV_?usp=sharing)

*IN and Out Folder*

**Input Video Stream**
Add your input Stream (Video File) into the **in** folder

**Output Video Stream**
Once the code is successfully completed - you should find a video of your input + stream + the augmentend information of tracker, detector and counting in the **out** folder

# Run the Code with a Counter
** Counting Line + YOLO + Deep Sort Standard**
```
python3 count.py --deepsort 1
```
**Counting Line + YOLO + Deep Sort Custom**
```
python3 count.py --deepsort 2
``` 
# Run the Code without a Counter
**YOLO + Deep Sort Standard**
```
python3 track.py --deepsort 1
```
**YOLO + Deep Sort Standard**
```
python3 track.py --deepsort 2
```
