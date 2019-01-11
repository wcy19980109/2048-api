import numpy as np
import torch

from Nathan_Wang.Network import NaiveNetwork
from Nathan_Wang.generator import Generator
from game2048.agents import Agent

onehot = Generator.onehot

USE_GPU = True
DEVICE_ID = 1
torch.cuda.set_device(DEVICE_ID)


class NaiveAgent(Agent):
    def __init__(self, game, display=None, model_path=None):
        super(NaiveAgent, self).__init__(game, display)

        self.model = NaiveNetwork(128)
        if USE_GPU:
            self.model.cuda()
        if model_path is not None:
            self.model.load(model_path, USE_GPU)

    def step(self):
        onehot_board = onehot(self.game.board)
        ds = np.zeros(shape=(4,))

        for i in range(4):
            state = np.rot90(onehot_board, k=i, axes=(1, 2)).copy()
            d = self.model.predict(state,USE_GPU)
            ds[(d - i) % 4] += 1

        return int(np.argmax(ds))
