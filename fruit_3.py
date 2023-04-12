#3.训练
import time

import torch
from tensorboardX import SummaryWriter
from torch import nn
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import fruit_2

# 网络搭建、模型训练
train_dataset = fruit_2.LoadData("train.txt", True)
test_dataset = fruit_2.LoadData("test.txt", True)
print("数据个数：", len(train_dataset))
print("测试数据个数：", len(test_dataset))
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=10,
                                               shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=10,
                                              shuffle=True)

train_data_size = len(train_dataset)  # 数据集长度
test_data_size = len(train_dataset)  # 数据集长度

#引入算法模型
tudui = models.AlexNet()
tudui = tudui.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()  # 参数默认。交叉熵作为损失函数的收敛速度比均方误差快，且较为容易找到函数最优解
loss_fn = loss_fn.to(device)

# 优化器（BGD,SGD,MBGD）梯度下降算法、随机梯度下降算法、小批量梯度下降算法
learning_rate = 0.01  # 学习速率
# learning_rate=1e-2,   1e-2=1x(10)^(-2)=1/100=0.01
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)  # 随机梯度下降

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的次数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("logs_train")
start_time = time.time()
for i in range(epoch):
    print("------第{}轮开始的------".format(i + 1))

    # 训练步骤开始
    tudui.train()
    for data in train_dataloader:
        imgs, targets = data
        # print("1{}".format(targets))
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = tudui(imgs)  # 对输入进行处理
        loss = loss_fn(outputs, targets)  # 损失函数
        # 优化器优化模型
        optimizer.zero_grad()  # 梯度置0
        loss.backward()  # 后向传播
        optimizer.step()  # 权重偏执进行梯度下降，进行优化

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print("训练次数:{},loss:{}".format(total_train_step, loss.item()))  # item去掉输出时的类型
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():  # with里面梯度就没有了
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = tudui(imgs)
            print(outputs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss
            print("targets为_{}".format(targets))
            print("outputs.argmax(1)为_{}".format(outputs.argmax(1)))
            accuracy = (outputs.argmax(1) == targets).sum()  # 计算样本正确率
            print(outputs.argmax(1))
            total_accuracy = int(accuracy + total_accuracy)  # 计算正确率之和

    print("整体测试集上的loss:{}".format(total_test_loss))
    print("整体测试集上的正确率:{}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step += 1

    if i == 5:
        torch.save(tudui, "fruit_{}.pth".format(i))
    if i == 10:
        torch.save(tudui, "fruit_{}.pth".format(i))
    print("模型已保存")

writer.close()
