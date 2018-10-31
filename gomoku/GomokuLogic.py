'''
Board class for Gomoku.
Board data:
	Pieces:	white = 1, black = -1, empty = 0
	Board:	1st index represents column, 2nd index represents row
Illustrate the basic move logic for Gomoku.
'''

class Board():

	# 4 directions
	__directions = [(0, 1), (1, 1), (1, 0), (1, -1)]

	def __init__(self, n):
		# size of board
		self.n = n
		# create an empty board
		self.pieces = [None] * self.n
		for i in range(self.n):
			self.pieces[i] = [0]*self.n


	def __getitem__(self, index):
		#if index >= self.n:
		#	print(index)
		return self.pieces[index]


	def get_legal_moves(self):
		# legal moves for each color are the empty position on board
		moves = list()

		for y in range(self.n):
			for x in range(self.n):
				if self[x][y] == 0:
					moves.append((x, y))
		return moves


	def has_legal_moves(self):
		for y in range(self.n):
			for x in range(self.n):
				if self[x][y] == 0:
					return True
		return False


	def execute_move(self, move, color):
		# perform the given move on the board
		(x, y) = move

		self[x][y] = color


	def check_win(self, last_move):
		# check whether the last_move result in an instantaneous victory
		origin = last_move
		(x, y) = origin
		if x == self.n or y == self.n:
			return 0
		color = self[x][y]
		for direction in self.__directions:
			count = 1
			for move in Board._increment_move(origin, direction, self.n):
				(x, y) = move
				if self[x][y] == color:
					count += 1
					if count == 5:
						return color
				else:
					break	
			ops_direction = [-x for x in direction]
			for move in Board._increment_move(origin, ops_direction, self.n):
				(x, y) = move
				if self[x][y] == color:
					count += 1
					if count == 5:
						return color
				else:
					break
		return 0


	@staticmethod
	def _increment_move(move, direction, n):
		# generator for moves along a given direction
		move = list(map(sum, zip(move, direction)))
		while all(map(lambda x: 0 <= x < n, move)):
			yield move
			move = list(map(sum, zip(move, direction)))