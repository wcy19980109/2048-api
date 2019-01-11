from game2048.game import Game
from game2048.displays import Display
from collections import Counter
import time


def single_run(size, score_to_win, AgentClass, **kwargs):
    game = Game(size, score_to_win)
    agent = AgentClass(game, display=None, **kwargs)
    n_iter = agent.play(verbose=False)
    return game.score, n_iter


if __name__ == '__main__':
    GAME_SIZE = 4
    SCORE_TO_WIN = 2048
    N_TESTS = 50
    '''====================
    Use your own agent here.'''
    # from game2048.agents import ExpectiMaxAgent as TestAgent
    from Nathan_Wang.naive_agent import NaiveAgent as TestAgent

    '''===================='''

    since = time.time()
    scores = []
    tot_iter = 0
    for _ in range(N_TESTS):
        score, n_iter = single_run(
            GAME_SIZE,
            SCORE_TO_WIN,
            AgentClass=TestAgent,
            model_path='./model/model_622.08.pth')
        scores.append(score)
        tot_iter += n_iter
        # print(F"{_}/{N_TESTS}: {score}")

    print("Distribution: ", Counter(scores))
    print(F"Total time: {time.time() - since}, total step: {tot_iter}")
    print(F"Average time per step: {(time.time() - since) / tot_iter}s")
    print("Average scores: @%s times" % N_TESTS, sum(scores) / len(scores))
