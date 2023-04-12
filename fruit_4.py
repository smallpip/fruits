#测试
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Linear
from torch.nn.modules.flatten import Flatten
from torch.utils.data import DataLoader

img_path="image/apple_1.png"
img_path1="image/香蕉_1.png"
# img_path2="image/橘子3.png"
image=Image.open(img_path)
image1=Image.open(img_path1)
# image2=Image.open(img_path2)


transform=torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)),
                                          torchvision.transforms.ToTensor()])
image=transform(image)
image1=transform(image1)
# image2=transform(image2)
# print(image.shape)


model=torch.load("fruit_5.pth")
image=torch.reshape(image,(1,3,224,224))
image=image.cuda()
with torch.no_grad():#节约内存，提升性能
    output=model(image)
    output=output.cuda()
i=output.argmax(1).item()
print(output.argmax(1))
list=['苹果','香蕉','葡萄','橘子','梨']
print("识别apple_1为{}".format(list[i]))

image1=torch.reshape(image1,(1,3,224,224))
image1=image1.cuda()
with torch.no_grad():#节约内存，提升性能
    output=model(image1)
    output=output.cuda()
i=output.argmax(1).item()
list=['苹果','香蕉','葡萄','橘子','梨']
print("识别香蕉_1为{}".format(list[i]))
#
# image2=torch.reshape(image2,(1,3,32,32))
# image2=image2.cuda()
# with torch.no_grad():#节约内存，提升性能
#     output=model(image2)
#     output=output.cuda()
# i=output.argmax(1).item()
# list=['小飞机','汽车','鸟','猫','鹿','狗','蛙','马','船','卡车']
# print("识别陈风顺为{}".format(list[i]))