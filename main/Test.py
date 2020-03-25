from torch.utils.data.dataloader import DataLoader

import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from torch.autograd import Variable

from main.ImageDataset import ImageDataSet
from main.ImageUtils import draw_rect
from main.Models import ClassifyNet

imageDataSet = ImageDataSet(file_name="./dataset/test.txt")
dataLoader = DataLoader(imageDataSet, batch_size=20, shuffle=True)

net = torch.load("net_1.0_.pt")
net.eval()


def show(index):
    image, label = imageDataSet[index]
    draw_rect(image.numpy())

    if torch.cuda.is_available():
        image = image.view(1, image.shape[0], image.shape[1], image.shape[2]).cuda().float()
    y = net(image)
    y = torch.argmax(y)
    print("预测：{}".format(y))
    print("标签：{}".format(label))


def predict():
    times = 0
    scores = 0
    for index, data in enumerate(dataLoader):
        times += 1
        image, label = data
        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()
        # image = image.view(image.shape[0], 3, image.shape[1], image.shape[2])
        image = Variable(image).float()
        label = Variable(label).long()
        y = net(image)

        score = torch.argmax(y, 1)
        score = score == label
        score = score.sum()
        score = score.float() / y.shape[0]
        score = score.cpu().numpy()
        scores += score
    scores = scores / times
    print("测试集上平均尊准确率：{}%".format(scores * 100))


# predict()

# i = int(input("输入index："))
predict()
