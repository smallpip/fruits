#2.数据预处理
import torch
from PIL import Image

import torchvision.transforms as transforms
from PIL import ImageFile
from tensorboardX import SummaryWriter

ImageFile.LOAD_TRUNCATED_IMAGES = True

from torch.utils.data import Dataset


# 数据归一化与标准化
class LoadData(Dataset):
    def __init__(self, txt_path, train_flag=True):
        self.imgs_info = self.get_images(txt_path)
        self.train_flag = train_flag

        self.train_tf = transforms.Compose([
                transforms.Resize(224,224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#归一化，归一化就是要把图片3个通道中的数据整理到[-1, 1]区间。
            #transforms.Normalize（mean,std）
            #mean：各通道的均值
            #std：各通道的标准差
            ])

        self.val_tf = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def get_images(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            imgs_info = f.readlines()#读取每一行信息
            imgs_info = list(map(lambda x:x.strip().split('\t'), imgs_info))#\t是表示tab键隔开，分别读取
        return imgs_info

    def padding_black(self, img):#填充图片变成224
        w, h  = img.size
        scale = 224. / max(w, h)
        img_fg = img.resize([int(x) for x in [w * scale, h * scale]])
        size_fg = img_fg.size
        size_bg = 224
        img_bg = Image.new("RGB", (size_bg, size_bg))
        img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
                              (size_bg - size_fg[1]) // 2))
        img = img_bg
        return img

    def __getitem__(self, index):
        img_path, label = self.imgs_info[index]
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = self.padding_black(img)
        if self.train_flag:
            img = self.train_tf(img)
        else:
            img = self.val_tf(img)
        # img = torch.reshape(img, (1, 3, 224, 224))
        label = int(label)

        return img, label

    def __len__(self):
        return len(self.imgs_info)


if __name__ == "__main__":
    train_dataset = LoadData("train.txt", True)
    print("数据个数：", len(train_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=10,
                                               shuffle=True)

    writer=SummaryWriter("logs_train")
    step=0
    for image, label in train_loader:
        writer.add_images("two",image,step)
        step+=1
        print(image.shape)
        print(image)
        # img = transform_BZ(image)
        # print(img)
        print(label)

    writer.close()