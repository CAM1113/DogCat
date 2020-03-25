from torch.utils.data.dataloader import DataLoader
import time
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from torch.autograd import Variable

from main.ImageDataset import ImageDataSet
from main.Models import ClassifyNet

imageDataSet = ImageDataSet()
dataLoader = DataLoader(imageDataSet, batch_size=30, shuffle=True)

total_epoch = 20
lr = 0.01
momentum = 0.5

net = ClassifyNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

index = 1
num_x = []
num_y = []
num_predict = []
low_predict = 0.9

for epoch in range(total_epoch):
    now = time.time()
    for _, data in enumerate(dataLoader):
        image, label = data
        if torch.cuda.is_available():
            image = image.cuda()
            label = label.cuda()
            net = net.cuda()

        image = Variable(image).float()
        label = Variable(label).long()
        y = net(image)
        optimizer.zero_grad()
        loss = criterion(y, label)
        loss.backward()
        optimizer.step()

        # 计算准确率
        predict = torch.argmax(y, 1)
        predict = predict == label
        predict = predict.sum()
        predict = predict.float() / y.shape[0]
        predict = predict.cpu().numpy()
        if predict >= low_predict & epoch > total_epoch / 2:
            low_predict = predict
            torch.save(net, f='./net_{}_.pt'.format(predict))
        num_x.append(index)
        index += 1
        num_y.append(loss.item())
        num_predict.append(predict)

        if _ == 0:
            # 计算准确率
            print("epoch:{}/{},loss = {},acc = {}".format(total_epoch, epoch + 1, loss.item(), predict))
    print(time.time() - now)
plt.plot(num_x, num_y)
plt.plot(num_x, num_predict)
plt.show()
