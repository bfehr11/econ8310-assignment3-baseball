# For reading data
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# For visualizing
import plotly.express as px

# For model building
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2

def load_labels_from_xml(xml_path, num_frames):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    labels = np.zeros(num_frames, dtype=np.int64)

    for track in root.findall("track"):
        for box in track.findall("box"):
            frame_idx = int(box.get("frame"))
            outside = int(box.get("outside"))

            if outside == 0:
                labels[frame_idx] = 1

    return labels

class CustomBaseballLoader(Dataset):
    def __init__(self, folder_path):
        self.samples = []

        video_files = [f for f in os.listdir(folder_path) if f.endswith(".mov")]
        
        for video_name in video_files:
            video_path = os.path.join(folder_path, video_name)

            cap = cv2.VideoCapture(video_path)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            labels_path = folder_path + "/" + video_name[:-3] + "xml"
            labels = load_labels_from_xml(labels_path, num_frames=num_frames)

            for frame_idx in range(num_frames):
                self.samples.append((video_path, frame_idx, labels[frame_idx]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, frame_idx, label = self.samples[idx]
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = min(frame_idx, total_frames - 1)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret or frame is None:
            # skip this frame
            return self.__getitem__((idx + 1) % len(self))
        cap.release()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frame = np.transpose(frame, (2, 0, 1))
        frame = torch.tensor(frame, dtype=torch.float32) / 255.0
        label = torch.tensor(label, dtype=torch.long)

        return frame, label

class Residual(nn.Module):
    """The Residual block of ResNet models."""
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super(Residual, self).__init__()
        self.conv1 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1,
                                   stride=strides)
        self.conv2 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1,
                                       stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

class ResNet(nn.Module):
    def __init__(self, arch, lr=0.1, num_classes=2):
        super(ResNet, self).__init__()
        self.net = nn.Sequential(self.b1())
        for i, b in enumerate(arch):
            self.net.add_module(f'b{i+2}', 
                self.block(*b, first_block=(i==0)))
        self.net.add_module('last', nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.LazyLinear(num_classes)))

    def b1(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def block(self, num_residuals, num_channels, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(num_channels,
                 use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(num_channels))
        return nn.Sequential(*blk)

    def forward(self, x):
        x = self.net(x)
        return x

class ResNet18(ResNet):
    def __init__(self, lr=0.1, num_classes=2):
        super(ResNet18, self).__init__(((2, 64), (2, 128),
         (2, 256), (2, 512)),
                       lr, num_classes)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X = X.to('cuda')
        y = y.to('cuda')

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to('cuda')
            y = y.to('cuda')

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {
        (100*correct):>0.1f}%, Avg loss: {
            test_loss:>8f} \n")
    
if __name__ == '__main__':
    train_data = CustomBaseballLoader("train_vids")
    test_data = CustomBaseballLoader("test_vids")

    train_dataloader = DataLoader(train_data, batch_size=8)
    test_dataloader = DataLoader(test_data, batch_size=8)

    model = ResNet18().to('cuda')

    learning_rate = 1e-2
    batch_size = 64
    epochs = 20 

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(),
        lr=learning_rate)
    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")

    EPOCH = epochs
    PATH = "model.pt"

    torch.save({'epoch': EPOCH, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),}, PATH)