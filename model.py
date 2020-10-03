#! /usr/bin/env python
import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import PIL
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils as utils
import torchvision.transforms as transforms

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
        return glob.glob(f"./images/{row['breed']}/{row['id']}_*.jpg")[0]


class Data:
    def __init__(self):
        self.data = pd.read_csv(
            './data/cats.csv',
            usecols=['id', 'type', 'breed', 'age', 'gender', 'size', 'coat'],
        )

        self.class_breed = {
            v: i for i, v in enumerate(pd.unique(self.data['breed']))
        }

        train, test = train_test_split(self.data, train_size=0.8)

        self.train = train.copy().reset_index()
        self.test = test.copy().reset_index()

        self.train_set = Dataset(self.train,
            class_type='breed', class_list=self.class_breed)

        self.test_set = Dataset(self.test,
            class_type='breed', class_list=self.class_breed)


class Net(nn.Module):
    def __init__(self, *, din, dout):
        super(Net, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(din, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, dout)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    data = Data()
    loader = utils.data.DataLoader(
        data.train_set, batch_size=16, shuffle=False, num_workers=0)

    import time
    for train_batch, labels_batch in loader:
        print(train_batch)
        print(labels_batch)
        time.sleep(1)
