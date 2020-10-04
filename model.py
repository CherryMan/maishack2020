#! /usr/bin/env python
import os, glob
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

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

class Image:
    SIZE = 40

    transform = transforms.Compose([
        transforms.Resize((SIZE, SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    def load(fname):
        return Image.transform(PIL.Image.open(fname))


class Dataset(utils.data.Dataset):
    def __init__(self, data: pd.DataFrame, *, class_type, class_list):
        super(Dataset).__init__()
        self.data = data
        self.class_type = class_type
        self.class_list = class_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx]
        img = Image.load(Dataset._fname(row))
        label = self.class_list[row[self.class_type]]
        return img, label

    def _fname(row: pd.Series):
        return glob.glob(f"./images/*/{row['id']}_*.jpg")[0]


class Data:
    def __init__(self, *, class_type):
        self.data = pd.read_csv(
            './data/cats.csv',
            usecols=['id', 'type', 'breed', 'age', 'gender', 'size', 'coat'],
        )

        self.classes = list(pd.unique(self.data[class_type]))
        self.class_type = class_type
        self.class_list = {
            v: i for i, v in enumerate(self.classes)
        }

        train, test = train_test_split(self.data, train_size=0.8)

        self.train = train.copy().reset_index()
        self.test = test.copy().reset_index()

        self.train_set = Dataset(self.train,
            class_type=class_type, class_list=self.class_list)

        self.test_set = Dataset(self.test,
            class_type=class_type, class_list=self.class_list)


class Net(nn.Module):
    def __init__(self, din, dout):
        super(Net, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, din, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, dout)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def model_train(data):
    loader = utils.data.DataLoader(
        data.train_set, batch_size=16, shuffle=False, num_workers=0)

    net = Net(6, len(data.classes))
    net.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):

        running_loss = 0.0
        for i, data in enumerate(loader, 1):
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 50 == 0:
                print('[%d, %5d, %3.0f%%] loss: %.3f' %
                  (epoch + 1, i, 100. * i / len(loader), running_loss / 50))
                running_loss = 0.0

    print('Done')
    torch.save(net.state_dict(), './model.pth')

def model_test(data):
    net = Net(6, len(data.classes))
    net.to(DEVICE)
    net.load_state_dict(torch.load('./model.pth'))

    testloader = utils.data.DataLoader(
        data.test_set, batch_size=4, shuffle=False, num_workers=0)

    def imshow(img):
        img = img / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    it = iter(testloader)
    images, labels = it.next()
    imshow(tv.utils.make_grid(images))
    print(' '.join('%5s' % data.classes[labels[i]] for i in range(4)))


if __name__ == '__main__':
    data = Data(class_type='breed')
    model_train(data)
    model_test(data)
    import time
    for train_batch, labels_batch in loader:
        print(train_batch)
        print(labels_batch)
        time.sleep(1)

print('Finished Training')
