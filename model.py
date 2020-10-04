#! /usr/bin/env python
import os, glob, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils as utils
import torchvision as tv
import torchvision.transforms as transforms
import torch.optim as optim

SEED = 0
CATEGORY_THRESH = 1000
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.manual_seed(SEED)
print(DEVICE)

class Image:
    SIZE = 48

    transform = transforms.Compose([
        transforms.Resize((SIZE, SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    def load(path):
        return Image.transform(PIL.Image.open(path).convert('RGB'))


class Dataset(utils.data.Dataset):
    def __init__(self, data, *, field, class_list):
        super(Dataset).__init__()
        self.data = data
        self.field = field
        self.class_list = class_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        img = Image.load(row['path'])
        label = self.class_list[row[self.field]]
        return img, label, row['idx']


class Data:
    def __init__(self, *, field, imdir):
        self.field = field
        self.classes = os.listdir(imdir)
        self.class_list = {
            v: i for i, v in enumerate(self.classes)
        }

        data = [
            (d, os.listdir(os.path.join(imdir, d)))
            for d in self.classes
        ]

        data = [
            {field: d, 'path': os.path.join(imdir, d, f)}
            for d, fs in filter(lambda x: len(x[1]) >= CATEGORY_THRESH, data)
            for f in fs
        ]

        self.data = [{**x, 'idx': i} for i, x in enumerate(data)]

        self.train, self.test = train_test_split(
            self.data, train_size=0.8, random_state=SEED, shuffle=True)

        self.train_set = Dataset(self.train,
                                 field=field, class_list=self.class_list)
        self.test_set = Dataset(self.test,
                                field=field, class_list=self.class_list)


class Net(nn.Module):
    def __init__(self, dout):
        super(Net, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 9 ** 2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, dout)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 9 ** 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def model_train(data):
    loader = utils.data.DataLoader(
        data.train_set, batch_size=16, shuffle=True, num_workers=0)

    net = Net(len(data.classes))
    net.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(4):

        running_loss = 0.0
        for i, data in enumerate(loader, 1):
            inputs, labels, idx = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 50 == 0:
                print('[%d, %4d, %3.0f%%] loss: %.3f' %
                  (epoch + 1, i, 100. * i / len(loader), running_loss / 50))
                running_loss = 0.0

    print('Done')
    torch.save(net.state_dict(), './model.pth')


def model_test(data):
    net = Net(len(data.classes))
    net.to(DEVICE)
    net.load_state_dict(torch.load('./model.pth'))

    testloader = utils.data.DataLoader(
        data.test_set, batch_size=4, shuffle=False, num_workers=0)

    correct = 0
    total = 0
    class_correct = [0] * len(data.classes)
    class_total = [0] * len(data.classes)
    with torch.no_grad():
        for images, labels, idx in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            c = (predicted == labels).squeeze()
            for i, l in enumerate(labels.tolist()):
                class_correct[l] += c[i].item()
                class_total[l] += 1

    print('Accuracy: %3.1f%% on %d' % (100. * correct / total, total))

    for i, c in enumerate(data.classes):
        print('%20s: %3.1f%% on %4d' %
          (c, 100. * class_correct[i] / class_total[i], class_total[i]))


if __name__ == '__main__':
    data = Data(field='breed', imdir='./raw-img')
    model_train(data)
    model_test(data)
