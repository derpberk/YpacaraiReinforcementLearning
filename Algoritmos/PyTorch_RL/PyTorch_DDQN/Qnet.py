# -*- coding: utf-8 -*-

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DeepQNetwork(nn.Module):
    def __init__(self, lr):
        super(DeepQNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(3,8,5)
        self.fc1 = nn.Linear(2520, 1024)
        self.fc2 = nn.Linear(1024,1024)
        self.fc3 = nn.Linear(1024,8)
        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()

        
    def forward(self,x):
        
        x = F.relu(self.conv1(x))
        x = T.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        return x
    
    
    
        