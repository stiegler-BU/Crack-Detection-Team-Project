# Crack-Detection-Team-Project







**Translated from original source README by Hua Tong:
**
A mask detection model was trained using YOLOV5

> explanation video from Bilibili：(https://www.bilibili.com/video/BV1YL4y1J7xz)
>
> CSDN blogs：(https://blog.csdn.net/ECHOSON/article/details/121939535)
>
> Code：(https://gitee.com/song-laogou/yolov5-mask-42)
>
> Processed data sets and trained models：(https://download.csdn.net/download/ECHOSON/63290559)
>
> related datasets：(https://blog.csdn.net/ECHOSON/article/details/121892887)

These are the effects we want to achieve:

![image-20211212181048969](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20211212181048969.png)

![image-20211212194124635](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20211212194124635.png)

## download code

The download address of the code is：(https://github.com/ultralytics/yolov5)

![image-20211214191424378](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20211214191424378.png)

## configuration environment

If you are not familiar with pycharm's anaconda, please read this csdn blog first to understand the basic operation of pycharm and anaconda:

(https://blog.csdn.net/ECHOSON/article/details/117220445)

After anaconda is installed, please switch to the Chinese source to improve the download speed:

```bash
conda config --remove-key channels
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.bfsu.edu.cn/anaconda/cloud/pytorch/
conda config --set show_channel_urls yes
pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple
```



To create a python3.8 virtual environment, do the following on the command line:

```bash
conda create -n yolo5 python==3.8.5
conda activate yolo5
```

### pytorch installation (gpu version and cpu version installation)

The actual test situation is that YOLOv5 can be used under both CPU and GPU conditions, but the training speed will be very slow under CPU conditions, so the conditional partner must install the GPU version of Pytorch, and the small partner without conditions is best to rent a server to use.

The detailed steps for installing the GPU version can be found in this article：(https://blog.csdn.net/ECHOSON/article/details/118420968)

The following points should be noted:

* Be sure to update your graphics driver before installation, go to the official website to download the corresponding driver installation
* 30 Series graphics cards can only use the cuda11 version
* Be sure to create a virtual environment so that there is no conflict between deep learning frameworks

Here I am creating a python3.8 environment and installing Pytorch version 1.8.0 with the following command:

```cmd
conda install pytorch==1.8.0 torchvision torchaudio cudatoolkit=10.2 # Note that this command specifies both the Pytorch version and the cuda version
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cpuonly # Friends who use the CPU can directly execute this command
```

### pycocotools installation

<font color='red'>Later I found a simpler installation method under windows, you can use the following command to install directly, do not need to download and then install </font>
```
pip install pycocotools-windows
```

### Installation of other packages

In addition, you also need to install other packages required by the program, including opencv, matplotlib, these packages, but the installation of these packages is relatively simple, directly through the pip command can be executed, we cd to the yolov5 code directory, directly execute the following instructions to complete the package installation.

```bash
pip install -r requirements.txt
pip install pyqt5
pip install labelme
```

### test

Execute the following code in the yolov5 directory

```bash
python detect.py --source data/images/bus.jpg --weights pretrained/yolov5s.pt
```

The following information will be output after execution

![image-20210610111308496](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20210610111308496.png)

The results of the tests can be found in the runs directory

![image-20210610111426144](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20210610111426144.png)

According to the official instructions, the detection code here is very powerful, and supports the detection of a variety of images and video streams. The specific use method is as follows:

```bash
 python detect.py --source 0  # webcam
                            file.jpg  # image 
                            file.mp4  # video
                            path/  # directory
                            path/*.jpg  # glob
                            'https://youtu.be/NUsoVlDFqZg'  # YouTube video
                            'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```



## data processing

I'm going to change it to yolo notation.

The recommended software here is labelimg, which can be installed with the pip command

Execute in your virtual environment`pip install labelimg  -i https://mirror.baidu.com/pypi/simple`command to install, and then directly run the labelimg software in the command line to start the data annotation software.

![image-20210609172156067](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20210609172156067.png)

The following interface is displayed after the software is started:

![image-20210609172557286](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20210609172557286.png)

### Data annotation

Although it is yolo model training, we still choose to label voc format here, one is to facilitate the use of data sets in other code, and the other is to provide data format conversion

**The process of labeling is：**

**1.Open picture directory**

![image-20210610004158135](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20210610004158135.png)

**2.Set the directory for saving annotation files and set automatic saving**

![image-20210610004215206](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20210610004215206.png)

**3.Start labeling, frame, label the target, 'crtl+s' save, and then d switch to the next sheet to continue labeling, repeat and repeat**

![image-20211212201302682](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20211212201302682.png)

The shortcut keys of labelimg are as follows. Learning the shortcut keys can help you improve the efficiency of data annotation.

![image-20210609171855504](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20210609171855504.png)

After the annotation is completed, you will get a series of txt files. The txt file here is the annotation file of target detection. The name of the txt file and the image file correspond one by one, as shown in the following figure:

![image-20211212170509714](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20211212170509714.png)

Open the specific annotation file, you will see the following content, each line in the txt file represents a target, distinguished by Spaces, respectively, indicating the category id of the target, the center point x coordinates, y coordinates, and the w and h of the target box after normalization.

![image-20211212170853677](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20211212170853677.png)

**4.Modify the data set configuration file**

The marked data should be placed in the following format for easy indexing by the program.

```bash
YOLO_Mask
└─ score
       ├─ images
       │    ├─ test # Put the test set picture
       │    ├─ train # Show the training set picture
       │    └─ val # Put a validation set image
       └─ labels
              ├─ test # Put the test set label
              ├─ train # Put the training set label
              ├─ val # Put the validation set label
```

The configuration file here is for the convenience of our later training, we need to create a 'mask_data.yaml' file in the data directory, as shown in the following figure:
![image-20211212174510070](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20211212174510070.png)

At this point, the data set processing part is basically completed, and the following content will be the model training!

## model training

### Basic training of the model

Create a 'mask_yolov5s.yaml' model configuration file under models as follows:

![image-20211212174749558](C:\Users\chenmingsong\AppData\Roaming\Typora\typora-user-images\image-20211212174749558.png)

Before you train your model, make sure you have the following files in your code directory

![image-20211212174920551](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20211212174920551.png)

Execute the following code to run the program:

```
python train.py --data mask_data.yaml --cfg mask_yolov5s.yaml --weights pretrained/yolov5s.pt --epoch 100 --batch-size 4 --device cpu
```

![image-20210610113348751](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20210610113348751.png)

The training code will output the following information on the command line after successful execution, and then wait for the model to finish training.

![image-20210610112655726](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20210610112655726.png)

Depending on the size of the data set and the performance of the device, the model is trained after a long wait, and the output is as follows:

![image-20210610134412258](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20210610134412258.png)

The trained model and log files can be found in the 'train/runs/exp3' directory

![image-20210610145140340](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20210610145140340.png)


## model evaluation

The most common evaluation indicator for object detection is mAP, which is a number between 0 and 1. The closer the number is to 1, the better the performance of your model.

Generally, we will come into contact with two indicators, recall and precision. The two indicators p and r are both values between 0 and 1 to judge the quality of the model simply from one perspective, where a value close to 1 indicates the better performance of the model, and a value close to 0 indicates the worse performance of the model. In order to comprehensively evaluate the performance of target detection, the average density map is generally used to further evaluate the model. By setting different confidence thresholds, we can get the P-value and R-value calculated by the model under different thresholds. In general, P-value and R-value are negatively correlated, and the curve shown in the figure below can be drawn. The area of the curve is called AP, and one AP value can be calculated for each target in the target detection model. The mAP value of the model can be obtained by averaging all the AP values. Taking this paper as an example, we can calculate the AP values of the two targets wearing helmets and those without helmets. By averaging the AP values of the two groups, we can obtain the mAP value of the entire model. The closer the value is to 1, the better the performance of the model.

For a more academic definition you can look it up yourself. Taking the model we trained this time as an example, at the end of the model you will find three images representing the recall rate, accuracy rate and mean mean density of our model on the verification set.

![image-20211212175851524](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20211212175851524.png)

Using the PR-curve as an example, you can see that our model has a mean density of 0.832 on the verification set.

![PR_curve](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/PR_curve.png)

If you don't have such curves in your catalog, probably because your model stopped halfway through training and didn't perform the validation process, you can generate these images by using the following command.

```bash
python val.py --data data/mask_data.yaml --weights runs/train/exp_yolov5s/weights/best.pt --img 640
```


## Model use

The use of the model is all integrated in the 'detect.py' directory, you can follow the instructions below to indicate what you want to detect

```bash
 # Detect camera
 python detect.py  --weights runs/train/exp_yolov5s/weights/best.pt --source 0  # webcam
 # Detect picture file
  python detect.py  --weights runs/train/exp_yolov5s/weights/best.pt --source file.jpg  # image 
 # Detect video file
   python detect.py --weights runs/train/exp_yolov5s/weights/best.pt --source file.mp4  # video
 # Detect files in a directory
  python detect.py --weights runs/train/exp_yolov5s/weights/best.pt path/  # directory
 # Detect network video
  python detect.py --weights runs/train/exp_yolov5s/weights/best.pt 'https://youtu.be/NUsoVlDFqZg'  # YouTube video
 # Detect streaming media
  python detect.py --weights runs/train/exp_yolov5s/weights/best.pt 'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream                            
```

Take our model of a mask, for example, If we execute 'python detect.py --weights runs/train/exp_yolov5s/weights/best.pt --source The data/images/fishman.jpg 'command can get such a test result.

![fishman](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/fishman.jpg)

## Building a visual interface

The visual interface part is in the 'window.py' file, is completed by pyqt5 interface design, in front of the startup, you need to replace the model with your trained model, the replacement position is in 'window.py' line 60, modify the address of your model can be, if you have a GPU, device can be set to 0 to indicate the use of line 0 GPU, which can speed up the recognition of the model.

![image-20211212194547804](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20211212194547804.png)

After the replacement, you can directly right-click run to start the graphical interface, go and test it for yourself to see the effect

![image-20211212194914890](https://vehicle4cm.oss-cn-beijing.aliyuncs.com/typoraimgs/image-20211212194914890.png)

