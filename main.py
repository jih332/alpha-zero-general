from Coach import Coach
from gomoku.GomokuGame import GomokuGame as Game
from gomoku.tensorflow.GomokuNNet import NNetWrapper as nn
from utils import *

args = dotdict({
    'numIters': 1000,
    'numEps': 60,
    'tempThreshold': 20,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 200,
    'arenaCompare': 30,
    'cpuct': 1,

    'checkpoint': './checkpoints/',
    'load_model': True,
    'load_examples': True,
    'load_folder_file': ('ckpt_', 55),
    'load_best_model': True,
    'load_latest_examples': True,
    'numItersForTrainExamplesHistory': 5,

})

if __name__=="__main__":
    g = Game(9)
    nnet = nn(g)

    if args.load_model:
        if args.load_best_model:
            nnet.load_checkpoint(args.checkpoint, 'best')
        else:
            nnet.load_checkpoint(args.checkpoint, args.load_folder_file)

    c = Coach(g, nnet, args)
    if args.load_examples:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
