import random
import chess
import numpy as np
import pandas as pd
import random
import time
import threading
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras import losses

from utils import display_board, outcome_to_numeric, NOTEBOOK

# Piece to index map:
# (this is according to Piece.piece_type - 1 on black lowercase)
# P p 6 0
# N n 7 1
# B b 8 2
# R r 9 3
# Q q 10 4
# K k 11 5
PIECE_TO_IDX = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
                'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}

IDX_TO_PIECE = {value: key for key, value in PIECE_TO_IDX.items()}

WINNER_TO_INT = {None: 0, True: 1, False: 0}

WHITES_TURN = 'w'
BLACKS_TURN = 'b'

STOP = 'STOP'

STOCKFISH_FILE_PATH = r"C:\Users\Nimi\Documents\Stockfish\stockfish_14.1_win_x64_avx2\stockfish_14.1_win_x64_avx2"

HUMAN = "Human Agent"
RANDOM = "Random Agent"
STOCKFISH = "STOCKFISH Agent"
NN = "NN Agent"


class ChessAgent:
    """
    Abstract class for a chess agent
    """
    def get_move(self, board):
        raise NotImplementedError()


class RandomChessAgent(ChessAgent):
    agent_type = RANDOM

    def get_move(self, board):
        return random.choice(list(board.legal_moves))


class HumanChessAgent(ChessAgent):
    agent_type = HUMAN

    def get_move(self, board):

        def get_move_from_user():
            move_stack = board.move_stack
            legal_moves = board.legal_moves
            if move_stack:
                last_move = board.pop()
                print("Opponent played {}".format(board.san(last_move)))
                board.push(last_move)
            while True:
                san_move = input("Please enter your move in (SAN): ")
                if san_move == "":
                    print("Null move is not accepted")
                    continue
                try:
                    move = board.parse_san(san_move)
                    return move
                except ValueError:
                    print("Invalid move. Legal moves: {}".format(legal_moves))

        display_board(board)
        move = get_move_from_user()
        return move


class StockfishChessAgent(ChessAgent):

    agent_type = STOCKFISH
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_FILE_PATH)

    def evaluate_board(board):
        raise NotImplementedError()  # TODO

    def get_move(self, board):
        result = StockfishChessAgent.engine.play(board, chess.engine.Limit(time=0.1))
        return result.move


class NNChessAgent(ChessAgent):
    """
    A chess agent with an evaluation NN and minimax approach
    """

    # Number of squares X number of pieces X number of colors
    # Who's turn it is right now, and castling options
    INPUT_DIM = 64 * 6 * 2 + 2 + 4

    agent_type = NN

    def __init__(self, depth=1, check_depth=3):
        # Create a nn:
        model = Sequential()
        model.add(layers.Dense(2 * NNChessAgent.INPUT_DIM, input_shape=(NNChessAgent.INPUT_DIM,), activation='relu'))
        model.add(layers.Dense(int(1.5 * NNChessAgent.INPUT_DIM), activation='relu'))
        model.add(layers.Dense(int(0.75 * NNChessAgent.INPUT_DIM), activation='relu'))
        model.add(layers.Dense(int(0.1125 * NNChessAgent.INPUT_DIM), activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        self.model = model
        self.depth = depth
        self.check_depth = check_depth

    @staticmethod
    def board_to_vector(board):
        # Not efficient! There's a HEAP of optimizations to make here.
        # Currently there is no en passant and 50-move draw stuff.
        # We use FEN format: https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation

        board_fen = board.fen()
        board_position, turn, castling, _, _, _ = board_fen.split()
        ranks = board_position.split("/")

        # It hurts, but let us parse:
        vec = np.zeros(NNChessAgent.INPUT_DIM)
        for rank_index, rank in enumerate(ranks):
            col_counter = 0  # columns counter
            while col_counter < 8:
                c = rank[0]
                if c.isnumeric():
                    # This means empty squares
                    col_counter += int(c)
                else:
                    piece_index = PIECE_TO_IDX[c]
                    vec[(rank_index * 8 * 12) + (col_counter * 12) + piece_index] = 1
                    col_counter += 1
                rank = rank[1:]
                continue
            assert col_counter == 8

        if turn == WHITES_TURN:
            vec[-6] = 1
        elif turn == BLACKS_TURN:
            vec[-5] = 1
        else:
            assert 'turn is bad!', turn

        if 'K' in castling:
            vec[-4] = 1
        if 'Q' in castling:
            vec[-3] = 1
        if 'k' in castling:
            vec[-2] = 1
        if 'q' in castling:
            vec[-1] = 1

        return tf.convert_to_tensor(vec.reshape(1, NNChessAgent.INPUT_DIM))

    @staticmethod
    def vector_to_board(vec):
        vec = vec.numpy().flatten()
        fen = ''
        # This hurts even more, but optimizations will come
        # Let us at least use the fact that we can read batches of 12s:
        for rank_index in range(8):
            calculated_rank_index = rank_index * 8 * 12
            for col_index in range(8):
                calculated_col_index = col_index * 12
                for piece_index in range(12):
                    if vec[calculated_rank_index + calculated_col_index + piece_index]:
                        c = IDX_TO_PIECE[piece_index]
                        break
                else:
                    c = ' '
                fen += c
            if rank_index != 7:
                fen += '/'

        # This is terrible:
        for i in range(8, 0, -1):
            fen = fen.replace(' ' * i, str(i))

        fen += ' '

        if vec[-6] == 1:
            fen += WHITES_TURN
        else:
            assert vec[-5]
            fen += BLACKS_TURN

        fen += ' '

        empty = True
        if vec[-4] == 1:
            fen += 'K'
            empty = False
        if vec[-3] == 1:
            fen += 'Q'
            empty = False
        if vec[-2] == 1:
            fen += 'k'
            empty = False
        if vec[-1] == 1:
            fen += 'q'
            empty = False
        if empty:
            fen += '-'

        fen += ' - 0 0'

        return chess.Board(fen)

    def evaluate_board_by_model(self, board):
        """
        return an evaluation (float) in range [-1, 1] by model (even if game is over)
        """
        vec = NNChessAgent.board_to_vector(board)
        return self.model(vec)

    def evaluate_board(self, board):
        """
        return an evaluation (float) in range [-1, 1]
        """
        outcome = board.outcome()
        if outcome:
            return outcome_to_numeric(outcome)
        return self.evaluate_board_by_model(board)

    def get_move(self, board):

        def greater_than(this, than):
            return this > than

        def less_than(this, than):
            return this < than

        def minimax(rem_depth):
            outcome = board.outcome()
            if outcome:
                return None, outcome_to_numeric(outcome)
            if not rem_depth:
                return None, self.evaluate_board_by_model(board)

            turn = board.turn
            if turn == chess.WHITE:
                better = greater_than
                best_value = -1
            else:
                better = less_than
                best_value = 1

            legal_moves = list(board.legal_moves)  # Optimization - does it have to be a list?
            best_move = legal_moves[0]

            for legal_move in legal_moves:
                board.push(legal_move)
                _, new_value = minimax(rem_depth - 1)
                board.pop()
                if better(this=new_value, than=best_value):
                    best_value = new_value
                    best_move = legal_move

                # If found mate - no need to further explore:
                if (turn == chess.WHITE and best_value == 1) or (turn == chess.BLACK and best_value == -1):
                    break

            return best_move, best_value

        num_legal_moves = board.legal_moves.count()
        if num_legal_moves > 15:
            depth = 2
        else:
            depth = self.depth * 2
        minimax_best_move, _ = minimax(depth)
        logging.info("Move chosen - {}".format(board.san(minimax_best_move)))
        return minimax_best_move

