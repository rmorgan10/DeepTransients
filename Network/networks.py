"""
Neural network for DeepTransient Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN, self).__init__()
        
        #Network Components
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=48, 
                               kernel_size=15, 
                               stride=3,
                               padding=2)
        
        self.conv2 = nn.Conv2d(in_channels=48, 
                               out_channels=96,
                               kernel_size=5, 
                               stride=1,
                               padding=2)
        
        self.dropout1 = nn.Dropout2d(0.25)
        
        self.dropout2 = nn.Dropout2d(0.5)
        
        self.fc1 = nn.Linear(in_features=3456, 
                             out_features=408)
        
        self.fc2 = nn.Linear(in_features=408, 
                             out_features=25)
        
        
    def forward(self, x):
        #Network Flow
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
class RNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=input_size,
            hidden_size=128,         # rnn hidden unit
            num_layers=2,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(128, 25)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out
    
class ZipperNN(nn.Module):
    def __init__(self, in_channels, input_size, num_classes):
        super(ZipperNN, self).__init__()
        
        #Network Components
        self.cnn = CNN(in_channels, num_classes)
        
        self.rnn = RNN(input_size, num_classes)
        
        self.connection = nn.Linear(50, num_classes*2)
        self.connection2 = nn.Linear(num_classes*2, num_classes)
        
    def forward(self, x, y):
        rnn_output = self.rnn(x)
        cnn_output = self.cnn(y)
        
        full_output = self.connection(torch.cat((rnn_output, cnn_output), dim=1))
        full_output = F.relu(full_output)
        full_output = self.connection2(full_output)
        full_output = F.log_softmax(full_output, dim=1)
        
        
        return full_output
