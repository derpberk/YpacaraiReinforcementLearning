# -*- coding: utf-8 -*-

import torch as T
import torch.nn as nn
import torch.optim as optim

class DeepQNetwork(nn.Module):
    def __init__(self, lr):
        super(DeepQNetwork, self).__init__()
        
        # La primera capa es una capa convolucional #
        self.conv1 = nn.Conv2d(3,8,5) # 3 canales de entrada, 8 filtros de 5x5.

    
        # La siguiente capa sera una capa LSTM#
        # Tiene 256 estados internos ocultos y 2 capas #
        self.lstm = nn.LSTM(input_size = self.calc_conv_output_dim(), \
                            hidden_size = 256, \
                            num_layers = 2)
        
        # Capa de salida #
        self.fc = nn.Linear(256, 8) 
        

        
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.loss = nn.SmoothL1Loss()
        
        
    def forward(self,x):
        
        batch_size, timesteps, C, H, W = x.size()
        
        c_in = x.view(batch_size * timesteps, C, H, W)
        
        c_out = T.relu(self.conv1(c_in))
        
        r_in = c_out.view(batch_size, timesteps, -1)
        
        r_out, (h_n, h_c) = self.lstm(r_in)
        
        r_out = T.relu(r_out)
        
        r_out2 = T.sigmoid(self.fc(r_out[:, -1, :]))
           
        return r_out2
    
    
    def calc_conv_output_dim(self):
        
         x = T.ones(1,10,3,25,19)
         batch_size, timesteps, C, H, W = x.size()
         c_in = x.view(batch_size * timesteps, C, H, W)
         c_out = T.relu(self.conv1(c_in))
         r_in = c_out.view(batch_size, timesteps, -1)
         
         dim = r_in.shape[2]
         
         return dim
       
       
        
    
    
        