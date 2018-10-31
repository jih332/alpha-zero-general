from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .GomokuLogic import Board
import numpy as np


class GomokuGame(Game):
    def __init__(self, n):
        self.n = n
        self.last_move = (n, n)


    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.n)
        return np.array(b.pieces)


    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.n)


    def getActionSize(self):
        # return number of actions
        return self.n * self.n


    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        # action = self.n*column+row
        b = Board(self.n)
        b.pieces = np.copy(board)
        move = (int(action / self.n), action % self.n)
        
        b.execute_move(move, player)
        self.last_move = move
        return (b.pieces, -player)


    def getValidMoves(self, board, player): #?
        # return a fixed size binary vector
        valids = [0]*self.getActionSize()
        b = Board(self.n)
        b.pieces = np.copy(board)
        legalMoves = b.get_legal_moves()
        for x, y in legalMoves:
            valids[self.n * x + y] = 1
        return np.array(valids)


    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost, 0.1 if a draw
        # player = 1
        b = Board(self.n)
        b.pieces = np.copy(board)
        self.winner = b.check_win(self.last_move)
        if self.winner == 1:
            return 1
        elif self.winner == -1:
            return -1
        if b.has_legal_moves():
            return 0
        else:
            return 0.001


    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        return player * board


    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert(len(pi) == self.n**2)  # 1 for pass
        pi_board = np.reshape(pi, (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()))]
        return l

    def stringRepresentation(self, board):
        # 8x8 numpy array (canonical board)
        return board.tostring()

    def getScore(self, board, player):
        # getScore after a game end?
        score = 2 * (self.winner == player) - 1
        # shall we add any parameters to encourage fast winning? 
        # i.e., reward fast winning with high score?
        return score

def display(board):
    n = board.shape[0]

    print("   ", end = "")
    for y in range(n):
        print (str(y) + "|",end="")
    print("")
    print(" -----------------------")
    for y in range(n):
        print(y, "|",end="")    # print the row #
        for x in range(n):
            piece = board[y][x]    # get the piece to print
            if piece == -1: print("w ",end="")
            elif piece == 1: print("b ",end="")
            else:
                if x==n:
                    print("-",end="")
                else:
                    print("- ",end="")
        print("|")

    print(" -----------------------")
