import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as init
import numpy as np
import random

class DigitClassificationNN(nn.Module):
    def __init__(self, num_classes=10, **kwargs):
        super(DigitClassificationNN, self).__init__()
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        np.random.seed(42)
        random.seed(42)
        torch.backends.cudnn.deterministic=True
        
        self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
        init.xavier_normal_(self.conv1.weight.data)
        init.zeros_(self.conv1.bias.data)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 5, 1, 2)
        init.xavier_normal_(self.conv2.weight.data)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
        init.xavier_normal_(self.conv3.weight.data)
        self.bn3 = nn.BatchNorm2d(128)
    
        self.fc1 = nn.Linear(6272, 2048)
        self.bn4 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, num_classes)


    def forward(self, x):
        x = func.relu(self.bn1(self.conv1(x)))
        x = func.max_pool2d(x, 2)

        x = func.relu(self.bn2(self.conv2(x)))
        x = func.max_pool2d(x, 2)

        x = func.relu(self.bn3(self.conv3(x)))

        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = self.bn4(x)
        x = func.relu(x)

        x = self.fc2(x)
        x = self.bn5(x)
        x = func.relu(x)

        x = self.fc3(x)
        return x
