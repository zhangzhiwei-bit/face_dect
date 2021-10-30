# ----------------------------------------------------#
#   对视频中的predict.py进行了修改，
#   将单张图片预测、摄像头检测和FPS测试功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
# ----------------------------------------------------#
import time

import cv2
import numpy as np

from retinaface import Retinaface

import torch
from torchvision import transforms
import torch.nn.functional as F
from torch import nn
from torchvision.models.mobilenet import mobilenet_v2
import pre_picture
import numpy as np
import cv2
from PIL import Image
import tkinter as tk



def conv_bn(inp, oup, stride=1, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )


def conv_dw(inp, oup, stride=1, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )


class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            # 640,640,3 -> 320,320,8
            conv_bn(3, 8, 2, leaky=0.1),  # 3
            # 320,320,8 -> 320,320,16
            conv_dw(8, 16, 1),  # 7

            # 320,320,16 -> 160,160,32
            conv_dw(16, 32, 2),  # 11
            conv_dw(32, 32, 1),  # 19

            # 160,160,32 -> 80,80,64
            conv_dw(32, 64, 2),  # 27
            conv_dw(64, 64, 1),  # 43
        )
        # 80,80,64 -> 40,40,128
        self.stage2 = nn.Sequential(
            conv_dw(64, 128, 2),  # 43 + 16 = 59
            conv_dw(128, 128, 1),  # 59 + 32 = 91
            conv_dw(128, 128, 1),  # 91 + 32 = 123
            conv_dw(128, 128, 1),  # 123 + 32 = 155
            conv_dw(128, 128, 1),  # 155 + 32 = 187
            conv_dw(128, 128, 1),  # 187 + 32 = 219
        )
        # 40,40,128 -> 20,20,256
        self.stage3 = nn.Sequential(
            conv_dw(128, 256, 2),  # 219 +3 2 = 241
            conv_dw(256, 256, 1),  # 241 + 64 = 301
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x


data_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class Net1(nn.Module):
    def __init__(self, model):
        super(Net1, self).__init__()
        self.mobilenet_layer = model
        self.Linear_layer1 = nn.Linear(1000, 512)
        self.Linear_layer2= nn.Linear(512,2)
    def forward(self, x):
        x = self.mobilenet_layer(x)
       # x = x.view(x.size(0), -1)
        x = F.relu(x)
        x = self.Linear_layer1(x)
        x = F.relu(x)
        x = self.Linear_layer2(x)
        return x


#戴口罩是0,不戴是1
mobilenet = mobilenet_v2(pretrained=True)
net = Net1(mobilenet)
net.load_state_dict(torch.load('model_data/test_mask3.pth',map_location=torch.device('cpu')))
net.eval()

net1=MobileNetV1()
net1.load_state_dict(torch.load('model_data/test_mobile1.pth',map_location=torch.device('cpu')))
net1.eval()
def dect_mask(net,frame):
    img = Image.fromarray(frame)
    return int(net(data_transform(img).unsqueeze(0)).argmax(1))


if __name__ == "__main__":
    retinaface = Retinaface()
    # -------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'表示单张图片预测
    #   'video'表示视频检测
    #   'fps'表示测试fps
    # -------------------------------------------------------------------------#
    mode = "video"
    # -------------------------------------------------------------------------#
    #   video_path用于指定视频的路径，当video_path=0时表示检测摄像头
    #   video_save_path表示视频保存的路径，当video_save_path=""时表示不保存
    #   video_fps用于保存的视频的fps
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出才会完成完整的保存步骤，不可直接结束程序。
    # -------------------------------------------------------------------------#
    video_path = 0
    video_save_path = ""
    video_fps = 25.0

    net.eval()


    def camera():
        capture = cv2.VideoCapture(0)
        fps = 0.0
        while (True):
            t1 = time.time()
            tem = 0
            dect = 'mask'
            # 读取某一帧
            ref, frame = capture.read()

            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 进行检测

            tem = dect_mask(net1, frame)
            if tem:
                dect = 'unmasked'
            else:
                dect = 'masked'

            # frame = cv2.putText(frame, dect, (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            frame = np.array(retinaface.detect_image(frame, dect))

            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            fps = (fps + (1. / (time.time() - t1))) / 2
            #print("fps= %.2f" % (fps))
            frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if c == 27:
                capture.release()
                break
        capture.release()
        cv2.destroyAllWindows()


    def imp():
        print("ertt")

    def rel():
        print("ertt")

   

    if mode == "predict":
        while True:
            img = input('Input image filename:')
            image = cv2.imread(img)
            if image is None:
                print('Open Error! Try again!')
                continue
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                r_image = retinaface.detect_image(image)
                r_image = cv2.cvtColor(r_image, cv2.COLOR_RGB2BGR)
                cv2.imshow("after", r_image)
                cv2.waitKey(0)
    
    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)
    
        fps = 0.0
        while (True):
            t1 = time.time()
            tem=0
            dect='mask'
            # 读取某一帧
            ref, frame = capture.read()
    
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 进行检测
    
            tem = dect_mask(net1, frame)
            if tem:
                dect = 'unmasked'
            else:
                dect = 'masked'
    
            # frame = cv2.putText(frame, dect, (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            frame = np.array(retinaface.detect_image(frame,dect))
    
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % (fps))
            frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    
            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_path != "":
                out.write(frame)
    
            if c == 27:
                capture.release()
                break
        capture.release()
        out.release()
        cv2.destroyAllWindows()
    
    elif mode == "fps":
        test_interval = 100
        img = cv2.imread('img/obama.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tact_time = retinaface.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1 / tact_time) + 'FPS, @batch_size 1')
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video' or 'fps'.")


