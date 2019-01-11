import time
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim

from Nathan_Wang import Dataset, NaiveNetwork, NaiveAgent, Generator
from game2048.game import Game

GEN_DATA = False
GOAL_SCORES = [512, 1024, 2048]
TRAIN_DATA_DIRS = [F"./data/{s}" for s in GOAL_SCORES]

LOAD_MODEL = True
MODEL_PATH = "./model/model.pth"
OUT_CHANNELS = 128
ONEHOT_SIZE = Generator.ONEHOT_SIZE
BATCH_SIZE = 5120 * 4
EPOCHS = 5
LEARN_RATE = 3e-4
WEIGHT_DECAY = 0.0005
N_EVAL = 50
HIGH_SCORE = 256

USE_GPU = torch.cuda.is_available()

print(F"use gpu = {USE_GPU}")
print(F"LOAD_MODEL = {LOAD_MODEL}")


class Train:
    def __init__(self):
        self.high_score = HIGH_SCORE

        # data
        self.dataset = Dataset()
        for d in TRAIN_DATA_DIRS:
            self.dataset.load(d)
            print(len(self.dataset))

        # model
        self.model = NaiveNetwork(OUT_CHANNELS)
        if USE_GPU:
            self.model.cuda()
        if LOAD_MODEL:
            self.model.load(MODEL_PATH, use_gpu=USE_GPU)

        # train components
        self.criterion = nn.CrossEntropyLoss()
        if USE_GPU:
            self.criterion.cuda()

        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=LEARN_RATE,
                                    weight_decay=WEIGHT_DECAY)

    def train(self):
        self.model.train()
        for epoch in range(EPOCHS):
            since = time.time()
            tot_loss, tot_cnt = 0, 0
            for s, a in self.dataset.get_loader(BATCH_SIZE, USE_GPU):
                s = s.float().view(-1, ONEHOT_SIZE, 4, 4)
                a = torch.LongTensor(a).view(-1)
                if USE_GPU:
                    s = s.cuda()
                    a = a.cuda()

                o = self.model(s)

                loss = self.criterion(o, a)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                tot_loss += loss.item()
                tot_cnt += 1

            print(F"Epoch: {epoch}/{EPOCHS}, "
                  F"time(s) = {time.time() - since:.1f}, "
                  F"loss = {tot_loss / tot_cnt:.4f}, ")

        self.model.save(MODEL_PATH)

    def eval(self):
        self.model.eval()
        scores = []
        n_iter = 0
        for i in range(N_EVAL):
            game = Game(4, 2048)
            path = MODEL_PATH if LOAD_MODEL else None
            n_iter += NaiveAgent(game, model_path=path).play()
            scores.append(game.score)

        average_iter = n_iter / N_EVAL
        average_score = sum(scores) / len(scores)
        print(F"Average_iter={average_iter},"
              F"average_score={average_score}",
              Counter(scores))

        if average_score > self.high_score:
            self.high_score = average_score
            self.model.save(F"./model/model_{average_score}.pth")


if __name__ == '__main__':
    if GEN_DATA:
        for score, data_dir in zip(GOAL_SCORES, TRAIN_DATA_DIRS):
            Generator.generate(
                score_to_win=score,
                count=1000000,
                data_dir=data_dir,
            )

    t = Train()
    from torchsummary import summary

    summary(t.model, input_size=(12, 4, 4))
    # while True:
    #     t.train()
    #     t.eval()
