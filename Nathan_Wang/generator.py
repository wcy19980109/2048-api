import numpy as np

from Nathan_Wang import Dataset
from game2048.expectimax import board_to_move
from game2048.game import Game


class Generator:
    ONEHOT_SIZE = 12
    map_table = {2 ** i: i for i in range(1, ONEHOT_SIZE)}
    map_table[0] = 0

    @staticmethod
    def onehot(state):  # (4,4)-->(ONEHOT_SIZE,4,4)
        ret = np.zeros(shape=(Generator.ONEHOT_SIZE, 4, 4))
        for r in range(4):
            for c in range(4):
                ret[Generator.map_table[state[r, c]], r, c] = 1
        return ret

    @staticmethod
    def generate(score_to_win, count, data_dir, file_name=None):
        dataset = Dataset()
        while len(dataset) < count:
            game = Game(4, score_to_win)
            states = []
            actions = []
            while not game.end:
                s, a = Generator.onehot(game.board), board_to_move(game.board)
                states.append(s)
                actions.append(a)
                game.move(a)
            dataset.push(states, actions)
            print(F"dataset.count = {len(dataset)}/{count}={len(dataset) / count * 100:.2f}%", end='\r')
        dataset.save(data_dir, file_name)
