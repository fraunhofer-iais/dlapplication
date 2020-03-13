import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import numpy as np
import random

class DeepDrivingNet(nn.Module):
    def __init__(self):
        super(DeepDrivingNet, self).__init__()
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        np.random.seed(42)
        random.seed(42)
        torch.backends.cudnn.deterministic=True

        self.conv1 = torch.nn.Conv2d(1, 24, kernel_size=5, stride=2, padding=0)
        init.xavier_normal_(self.conv1.weight.data)
        init.zeros_(self.conv1.bias.data)
        self.conv2 = torch.nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=0)
        init.xavier_normal_(self.conv2.weight.data)
        init.zeros_(self.conv2.bias.data)
        self.conv3 = torch.nn.Conv2d(36, 48, kernel_size=5, stride=2, padding=0)
        init.xavier_normal_(self.conv3.weight.data)
        init.zeros_(self.conv3.bias.data)
        self.conv4 = torch.nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=0)
        init.xavier_normal_(self.conv4.weight.data)
        init.zeros_(self.conv4.bias.data)
        self.conv5 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        init.xavier_normal_(self.conv5.weight.data)
        init.zeros_(self.conv5.bias.data)
        
        self.fc1 = torch.nn.Linear(64 * 2790, 100)
        init.xavier_normal_(self.fc1.weight.data)
        init.zeros_(self.fc1.bias.data)
        self.fc2 = torch.nn.Linear(100, 50)
        init.xavier_normal_(self.fc2.weight.data)
        init.zeros_(self.fc2.bias.data)
        self.fc3 = torch.nn.Linear(50, 10)
        init.xavier_normal_(self.fc3.weight.data)
        init.zeros_(self.fc3.bias.data)
        self.fc4 = torch.nn.Linear(10, 1)
        init.xavier_normal_(self.fc4.weight.data)
        init.zeros_(self.fc4.bias.data)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        x = x.view(-1, 64 * 2790)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return(x)

    def __str__(self):
        return "Deep Driving CNN"

class DrivingCNNBatchNorm(torch.nn.Module):
    def __init__(self):
        super(DrivingCNNBatchNorm, self).__init__()
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        np.random.seed(42)
        random.seed(42)
        torch.backends.cudnn.deterministic=True
        
        self.conv1 = torch.nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=0)
        init.xavier_normal_(self.conv1.weight.data)
        init.zeros_(self.conv1.bias.data)
        
        self.conv1_bn = torch.nn.BatchNorm2d(24)
        
        self.conv2 = torch.nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=0)
        init.xavier_normal_(self.conv2.weight.data)
        init.zeros_(self.conv2.bias.data)
        
        self.conv2_bn = torch.nn.BatchNorm2d(36)
        
        self.conv3 = torch.nn.Conv2d(36, 48, kernel_size=5, stride=2, padding=0)
        init.xavier_normal_(self.conv3.weight.data)
        init.zeros_(self.conv3.bias.data)
        
        self.conv3_bn = torch.nn.BatchNorm2d(48)
        
        self.conv4 = torch.nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=0)
        #self.conv4 = torch.nn.Conv2d(48, 64, kernel_size=5, stride=2, padding=0)
        init.xavier_normal_(self.conv4.weight.data)
        init.zeros_(self.conv4.bias.data)
        
        self.conv4_bn = torch.nn.BatchNorm2d(64)
        
        self.conv5 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        #self.conv5 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0)
        init.xavier_normal_(self.conv5.weight.data)
        init.zeros_(self.conv5.bias.data)
        
        self.conv5_bn = torch.nn.BatchNorm2d(64)
        
        
        self.fc1 = torch.nn.Linear(2*32*1302, 100)#83328      ##self.fc1 = torch.nn.Linear(64 * 2790, 100)
        init.xavier_normal_(self.fc1.weight.data)
        init.zeros_(self.fc1.bias.data)
        
        self.fc1_bn = torch.nn.BatchNorm1d(100)
        
        self.fc2 = torch.nn.Linear(100, 50)
        init.xavier_normal_(self.fc2.weight.data)
        init.zeros_(self.fc2.bias.data)
        
        self.fc2_bn = torch.nn.BatchNorm1d(50)
        
        self.fc3 = torch.nn.Linear(50, 10)
        init.xavier_normal_(self.fc3.weight.data)
        init.zeros_(self.fc3.bias.data)
        
        self.fc3_bn = torch.nn.BatchNorm1d(10)
        
        
        self.fc4 = torch.nn.Linear(10, 1)
        init.xavier_normal_(self.fc4.weight.data)
        init.zeros_(self.fc4.bias.data)
        
    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        #print(x.shape)
        x = F.relu(self.conv2_bn(self.conv2(x)))
        #print(x.shape)
        x = F.relu(self.conv3_bn(self.conv3(x)))
        #print(x.shape)
        x = F.relu(self.conv4_bn(self.conv4(x)))
        #print(x.shape)
        x = F.relu(self.conv5_bn(self.conv5(x)))
        #print(x.shape)
        x = x.view(-1, 2*32*1302)#x = x.view(-1, 83328)     #######x = x.view(-1, 64 * 2790)
        #print("After viewing")
        #print(x.shape)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        #print(x.shape)
        x = F.relu(self.fc2_bn(self.fc2(x)))
        #print(x.shape)
        x = F.relu(self.fc3_bn(self.fc3(x)))
        #print(x.shape)
        x = self.fc4(x)
        #print(x.shape)
        return(x)

    def __str__(self):
        return "Deep Driving CNN with batch normalization"
