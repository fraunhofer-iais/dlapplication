import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import numpy as np
import random

class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()

        torch.manual_seed(7)
        torch.cuda.manual_seed_all(7)
        np.random.seed(7)
        random.seed(7)
        torch.backends.cudnn.deterministic=True

        self.fc1 = nn.Linear(150, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
    
    def __str__(self):
        return "MLP 2 layered"

class DeepMLPNet(nn.Module):
    def __init__(self):
        super(DeepMLPNet, self).__init__()

        torch.manual_seed(7)
        torch.cuda.manual_seed_all(7)
        np.random.seed(7)
        random.seed(7)
        torch.backends.cudnn.deterministic=True

        self.fc1 = nn.Linear(150, 150)
        init.xavier_normal_(self.fc1.weight.data)
        init.zeros_(self.fc1.bias.data)
        self.fc2 = nn.Linear(150, 100)
        init.xavier_normal_(self.fc2.weight.data)
        init.zeros_(self.fc2.bias.data)
        self.fc3 = nn.Linear(100, 100)
        init.xavier_normal_(self.fc3.weight.data)
        init.zeros_(self.fc3.bias.data)
        self.fc4 = nn.Linear(100, 50)
        init.xavier_normal_(self.fc4.weight.data)
        init.zeros_(self.fc4.bias.data)
        self.fc5 = nn.Linear(50, 50)
        init.xavier_normal_(self.fc5.weight.data)
        init.zeros_(self.fc5.bias.data)
        self.fc6 = nn.Linear(50, 25)
        init.xavier_normal_(self.fc6.weight.data)
        init.zeros_(self.fc6.bias.data)
        self.fc7 = nn.Linear(25, 25)
        init.xavier_normal_(self.fc7.weight.data)
        init.zeros_(self.fc7.bias.data)
        self.fc8 = nn.Linear(25, 25)
        init.xavier_normal_(self.fc8.weight.data)
        init.zeros_(self.fc8.bias.data)
        self.fc9 = nn.Linear(25, 10)
        init.xavier_normal_(self.fc9.weight.data)
        init.zeros_(self.fc9.bias.data)
        self.fc10 = nn.Linear(10, 10)
        init.xavier_normal_(self.fc10.weight.data)
        init.zeros_(self.fc10.bias.data)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        x = F.relu(self.fc10(x))
        return x

    def __str__(self):
        return "MLP 10 layered"

class OneOutputMLPNet(nn.Module):
    def __init__(self):
        super(OneOutputMLPNet, self).__init__()

        torch.manual_seed(7)
        torch.cuda.manual_seed_all(7)
        np.random.seed(7)
        random.seed(7)
        torch.backends.cudnn.deterministic=True

        self.fc1 = nn.Linear(150, 150)
        init.xavier_normal_(self.fc1.weight.data)
        init.zeros_(self.fc1.bias.data)
        self.fc2 = nn.Linear(150, 150)
        init.xavier_normal_(self.fc2.weight.data)
        init.zeros_(self.fc2.bias.data)
        self.fc3 = nn.Linear(150, 100)
        init.xavier_normal_(self.fc3.weight.data)
        init.zeros_(self.fc3.bias.data)
        self.fc4 = nn.Linear(100, 100)
        init.xavier_normal_(self.fc4.weight.data)
        init.zeros_(self.fc4.bias.data)
        self.fc5 = nn.Linear(100, 50)
        init.xavier_normal_(self.fc5.weight.data)
        init.zeros_(self.fc5.bias.data)
        self.fc6 = nn.Linear(50, 25)
        init.xavier_normal_(self.fc6.weight.data)
        init.zeros_(self.fc6.bias.data)
        self.fc7 = nn.Linear(25, 10)
        init.xavier_normal_(self.fc7.weight.data)
        init.zeros_(self.fc7.bias.data)
        self.fc8 = nn.Linear(10, 1)
        init.xavier_normal_(self.fc8.weight.data)
        init.zeros_(self.fc8.bias.data)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        return x

    def __str__(self):
        return "MLP with 1 output"

class WideOneOutputMLPNet(nn.Module):
    def __init__(self):
        super(WideOneOutputMLPNet, self).__init__()

        torch.manual_seed(7)
        torch.cuda.manual_seed_all(7)
        np.random.seed(7)
        random.seed(7)
        torch.backends.cudnn.deterministic=True

        self.fc1 = nn.Linear(150, 200)
        init.xavier_normal_(self.fc1.weight.data)
        init.zeros_(self.fc1.bias.data)
        self.fc2 = nn.Linear(200, 250)
        init.xavier_normal_(self.fc2.weight.data)
        init.zeros_(self.fc2.bias.data)
        self.fc3 = nn.Linear(250, 100)
        init.xavier_normal_(self.fc3.weight.data)
        init.zeros_(self.fc3.bias.data)
        self.fc4 = nn.Linear(100, 50)
        init.xavier_normal_(self.fc4.weight.data)
        init.zeros_(self.fc4.bias.data)
        self.fc5 = nn.Linear(50, 1)
        init.xavier_normal_(self.fc5.weight.data)
        init.zeros_(self.fc5.bias.data)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return x

    def __str__(self):
        return "MLP with wide layers"
