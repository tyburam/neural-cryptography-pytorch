import numpy as np
import torch.nn as nn

from src.config import *


# Function to generate n random messages and keys
def gen_data(n=BATCH_SIZE, msg_len=MSG_LEN, key_len=KEY_LEN):
    return (np.random.randint(0, 2, size=(n, msg_len)) * 2 - 1), \
           (np.random.randint(0, 2, size=(n, key_len)) * 2 - 1)


def init_xavier(m):
    if type(m) in [nn.Linear, nn.Conv1d]:
        nn.init.xavier_normal_(m.weight.data)
