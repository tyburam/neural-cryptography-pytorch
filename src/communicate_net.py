import torch.nn as nn
import torch.nn.functional as F

from src.config import *


class CommunicateNet(nn.Module):
    def __init__(self):
        super(CommunicateNet, self).__init__()
        self.input = nn.Linear(MSG_LEN + KEY_LEN, MSG_LEN + KEY_LEN)
        self.hidden = nn.Sigmoid()
        self.conv0 = nn.Conv1d(1, 2, 1, stride=1)
        self.conv1 = nn.Conv1d(2, 4, 1, stride=2)
        self.conv2 = nn.Conv1d(4, 4, 1, stride=1)
        self.conv3 = nn.Conv1d(4, 1, 1, stride=1)

    def forward(self, x):
        x = self.input(x)
        x = self.hidden(x).unsqueeze(1)
        x = F.relu(self.conv0(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x).tanh()
        x = x.squeeze()
        return x
