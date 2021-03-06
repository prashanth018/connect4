from agent import DQNAgent

import numpy as np
import random
import pygame
import sys
import math
import time
import tensorflow as tf
import matplotlib.pyplot as plt
tf.enable_eager_execution()


BLUE = (0,0,255)
BLACK = (0,0,0)
RED = (255,0,0)
YELLOW = (255,255,0)

ROW_COUNT = 6
COLUMN_COUNT = 7

PLAYER1 = 0
PLAYER2 = 1

EMPTY = 0
PLAYER1_PIECE = 1
PLAYER2_PIECE = -1

WINDOW_LENGTH = 4

def create_board():
	board = np.zeros((ROW_COUNT,COLUMN_COUNT))
	return board

def drop_piece(board, row, col, piece):
	board[row][col] = piece

def is_valid_location(board, col):
	return board[ROW_COUNT-1][col] == 0

def get_next_open_row(board, col):
	for r in range(ROW_COUNT):
		if board[r][col] == 0:
			return r

def print_board(board):
	print(np.flip(board, 0))

def winning_move(board, piece):
	# Check horizontal locations for win
	for c in range(COLUMN_COUNT-3):
		for r in range(ROW_COUNT):
			if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
				return True

	# Check vertical locations for win
	for c in range(COLUMN_COUNT):
		for r in range(ROW_COUNT-3):
			if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
				return True

	# Check positively sloped diaganols
	for c in range(COLUMN_COUNT-3):
		for r in range(ROW_COUNT-3):
			if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
				return True

	# Check negatively sloped diaganols
	for c in range(COLUMN_COUNT-3):
		for r in range(3, ROW_COUNT):
			if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
				return True

def evaluate_window(window, piece):
	score = 0
	opp_piece = PLAYER1_PIECE
	if piece == PLAYER1_PIECE:
		opp_piece = PLAYER2_PIECE

	if window.count(piece) == 4:
		score += 100
	elif window.count(piece) == 3 and window.count(EMPTY) == 1:
		score += 5
	elif window.count(piece) == 2 and window.count(EMPTY) == 2:
		score += 2

	if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
		score -= 4

	return score

def score_position(board, piece):
	score = 0

	## Score center column
	center_array = [int(i) for i in list(board[:, COLUMN_COUNT//2])]
	center_count = center_array.count(piece)
	score += center_count * 3

	## Score Horizontal
	for r in range(ROW_COUNT):
		row_array = [int(i) for i in list(board[r,:])]
		for c in range(COLUMN_COUNT-3):
			window = row_array[c:c+WINDOW_LENGTH]
			score += evaluate_window(window, piece)

	## Score Vertical
	for c in range(COLUMN_COUNT):
		col_array = [int(i) for i in list(board[:,c])]
		for r in range(ROW_COUNT-3):
			window = col_array[r:r+WINDOW_LENGTH]
			score += evaluate_window(window, piece)

	## Score posiive sloped diagonal
	for r in range(ROW_COUNT-3):
		for c in range(COLUMN_COUNT-3):
			window = [board[r+i][c+i] for i in range(WINDOW_LENGTH)]
			score += evaluate_window(window, piece)

	for r in range(ROW_COUNT-3):
		for c in range(COLUMN_COUNT-3):
			window = [board[r+3-i][c+i] for i in range(WINDOW_LENGTH)]
			score += evaluate_window(window, piece)

	return score

def is_terminal_node(board):
	return winning_move(board, PLAYER1_PIECE) or winning_move(board, PLAYER2_PIECE) or len(get_valid_locations(board)) == 0

def minimax(board, depth, alpha, beta, maximizingPlayer):
	valid_locations = get_valid_locations(board)
	is_terminal = is_terminal_node(board)
	if depth == 0 or is_terminal:
		if is_terminal:
			if winning_move(board, PLAYER2_PIECE):
				return (None, 100000000000000)
			elif winning_move(board, PLAYER1_PIECE):
				return (None, -10000000000000)
			else: # Game is over, no more valid moves
				return (None, 0)
		else: # Depth is zero
			return (None, score_position(board, PLAYER2_PIECE))
	if maximizingPlayer:
		value = -math.inf
		column = random.choice(valid_locations)
		for col in valid_locations:
			row = get_next_open_row(board, col)
			b_copy = board.copy()
			drop_piece(b_copy, row, col, PLAYER2_PIECE)
			new_score = minimax(b_copy, depth-1, alpha, beta, False)[1]
			if new_score > value:
				value = new_score
				column = col
			alpha = max(alpha, value)
			if alpha >= beta:
				break
		return column, value

	else: # Minimizing player
		value = math.inf
		column = random.choice(valid_locations)
		for col in valid_locations:
			row = get_next_open_row(board, col)
			b_copy = board.copy()
			drop_piece(b_copy, row, col, PLAYER1_PIECE)
			new_score = minimax(b_copy, depth-1, alpha, beta, True)[1]
			if new_score < value:
				value = new_score
				column = col
			beta = min(beta, value)
			if alpha >= beta:
				break
		return column, value

def get_valid_locations(board):
	valid_locations = []
	for col in range(COLUMN_COUNT):
		if is_valid_location(board, col):
			valid_locations.append(col)
	return valid_locations

def pick_best_move(board, piece):

	valid_locations = get_valid_locations(board)
	best_score = -10000
	best_col = random.choice(valid_locations)
	for col in valid_locations:
		row = get_next_open_row(board, col)
		temp_board = board.copy()
		drop_piece(temp_board, row, col, piece)
		score = score_position(temp_board, piece)
		if score > best_score:
			best_score = score
			best_col = col

	return best_col

def draw_board(board):
	for c in range(COLUMN_COUNT):
		for r in range(ROW_COUNT):
			pygame.draw.rect(screen, BLUE, (c*SQUARESIZE, r*SQUARESIZE+SQUARESIZE, SQUARESIZE, SQUARESIZE))
			pygame.draw.circle(screen, BLACK, (int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)), RADIUS)
	
	for c in range(COLUMN_COUNT):
		for r in range(ROW_COUNT):		
			if board[r][c] == PLAYER1_PIECE:
				pygame.draw.circle(screen, RED, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
			elif board[r][c] == PLAYER2_PIECE: 
				pygame.draw.circle(screen, YELLOW, (int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)), RADIUS)
	pygame.display.update()


board = create_board()

player1 = DQNAgent(np.asarray(board))
player2 = DQNAgent(np.asarray(board))

#player1.load_weights('player1.h5')
#player2.load_weights('player2.h5')

player1_wins = []
player2_wins = []
player1_disqualifications = []
player2_disqualifications = []
draws = []

NUM_GAMES = 100 * 10000
TRAIN_AFTER_GAMES = 100

num_wins1 = 0
num_wins2 = 0
num_disqualifications1 = 0
num_disqualifications2 = 0

eps = 1.0
eps_decay = 0.9995

for episode in range(NUM_GAMES):


	board = create_board()
	#print_board(board)
	game_over = False

	#pygame.init()

	SQUARESIZE = 100

	width = COLUMN_COUNT * SQUARESIZE
	height = (ROW_COUNT+1) * SQUARESIZE

	size = (width, height)

	RADIUS = int(SQUARESIZE/2 - 5)

	#screen = pygame.display.set_mode(size)
	#draw_board(board)
	#pygame.display.update()

	#myfont = pygame.font.SysFont("monospace", 75)

	turn = random.randint(PLAYER1, PLAYER2)

	if (episode+1) % TRAIN_AFTER_GAMES == 0:
		
		loss1 = player1.train()
		loss2 = player2.train()

		player1_wins.append(num_wins1)
		player2_wins.append(num_wins2)
		player1_disqualifications.append(num_disqualifications1)
		player2_disqualifications.append(num_disqualifications2)
		draws.append(TRAIN_AFTER_GAMES - num_wins1 - num_wins2 - num_disqualifications1 - num_disqualifications2)

		print()
		print('*' * 100)
		print()
		print('After', episode, 'games,')
		print('loss for player1 =', loss1)
		print('loss fontor player2 =', loss2)

		print('Player 1 won:', num_wins1, 'games')
		print('Player 2 won:', num_wins2, 'games')
		print('Player 1 disqualified for', num_disqualifications1, 'games')
		print('Player 2 disqualified for', num_disqualifications2, 'games')

		num_wins1 = 0
		num_wins2 = 0
		num_disqualifications1 = 0
		num_disqualifications2 = 0

		player1.save_weights('player1.h5')
		player2.save_weights('player2.h5')

		player1.adjust_target_net()
		player2.adjust_target_net()

		eps *= eps_decay

	game_over = False


	while not game_over:

		# Ask for Player 2 Input
		if turn == PLAYER2 and not game_over:				

			#col = random.randint(0, COLUMN_COUNT-1)
			#col = pick_best_move(board, PLAYER2_PIECE)
			col = player2.get_action(board, eps)
			#print('col from DQN:', col)

			if not is_valid_location(board, col):
				game_over = True
				reward = -1
				player2.receive_next_obs_rew_done(board, reward, game_over)
				num_disqualifications2 += 1

			if is_valid_location(board, col):
				#pygame.time.wait(500)
				row = get_next_open_row(board, col)
				drop_piece(board, row, col, PLAYER2_PIECE)

				reward = 0

				if winning_move(board, PLAYER2_PIECE):
					game_over = True
					reward = 1	
					player2.receive_next_obs_rew_done(board, reward, game_over)
					num_wins2 += 1

				#print_board(board)
				#draw_board(board)

				turn += 1
				turn = turn % 2

				player1.receive_next_obs_rew_done(board, -1 * reward, game_over)

		else:

			#col, minimax_score = minimax(board, 5, -math.inf, math.inf, True)
			col = player1.get_action(board, eps)
			#print('col from minimax:', col)

			if not is_valid_location(board, col):
				game_over = True
				reward = -1
				player1.receive_next_obs_rew_done(board, reward, game_over)
				num_disqualifications1 += 1

			if is_valid_location(board, col):
				#pygame.time.wait(500)
				row = get_next_open_row(board, col)
				drop_piece(board, row, col, PLAYER1_PIECE)

				reward = 0

				if winning_move(board, PLAYER1_PIECE):
					game_over = True
					reward = 1
					player1.receive_next_obs_rew_done(board, reward, game_over)
					num_wins1 += 1
				#print_board(board)
				#draw_board(board)

				turn += 1
				turn = turn % 2

				player2.receive_next_obs_rew_done(board, -1*reward, game_over)
		

plt.figure()	
plt.plot(player1_wins, label='player 1 wins')
plt.plot(player2_wins, label='player 2 wins')
plt.plot(player1_disqualifications, label='player 1 disqualifications')
plt.plot(player2_disqualifications, label='player 2 disqualifications')
plt.plot(draws, label='draws')
plt.legend()
plt.show()