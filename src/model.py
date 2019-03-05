import itertools
import torch

from src.config import *
from src.communicate_net import CommunicateNet
from src.eve_net import EveNet
from src.utils import gen_data


class CryptoNet(object):
    def __init__(self, msg_len=MSG_LEN, batch_size=BATCH_SIZE,
                 epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE):
        """
        Args:
            msg_len: The length of the input message to encrypt.
            batch_size: Minibatch size for each adversarial training
            epochs: Number of epochs in the adversarial training
            learning_rate: Learning Rate for Adam Optimizer
        """

        self.msg_len = msg_len
        self.key_len = self.msg_len
        self.N = self.msg_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.alice = None
        self.bob = None
        self.eve = None

        self.build_model()

    def build_model(self):
        self.alice = CommunicateNet()
        self.bob = CommunicateNet()
        self.eve = EveNet()

    def train(self):
        mae = torch.nn.L1Loss()
        comm_optimizer = torch.optim.Adam(itertools.chain(self.alice.parameters(), self.bob.parameters()),
                                          lr=self.learning_rate)
        eve_optimizer = torch.optim.Adam(self.eve.parameters(), lr=self.learning_rate)

        for i in range(self.epochs):
            data = gen_data(self.batch_size, self.msg_len, self.key_len)
            msg = torch.tensor(data[0], dtype=torch.float)
            key = torch.tensor(data[1], dtype=torch.float)
            alice_input = torch.cat((msg, key), 1)

            comm_optimizer.zero_grad()

            alice_output = self.alice(alice_input)
            bob_input = torch.cat((alice_output, key), 1)
            bob_output = self.bob(bob_input)
            eve_output = self.eve(alice_output)

            err_eve = mae(msg, eve_output)
            err_bob = mae(msg, bob_output)
            bob_loss = err_bob + (1. - err_eve) ** 2
            if i % 10 == 0:
                print('epoch: {} bob error: {} eve error: {}'.format(i, err_bob.data, err_eve.data))

            bob_loss.backward(retain_graph=True)
            comm_optimizer.step()

            eve_optimizer.zero_grad()
            err_eve.backward()
            eve_optimizer.step()
