import chess
import chess.engine
import chess.svg
import cProfile
import pstats

from agents import *
from utils import *


def play_chess_game(agent_white, agent_black, logs=True):
    if logs:
        print(f"Starting a chess game, White is {agent_white.agent_type} and black is {agent_black.agent_type}")

    is_any_human = agent_white.agent_type == HUMAN or agent_black.agent_type == HUMAN
    board = chess.Board()

    if is_any_human:
        if not NOTEBOOK:
            display_t = threading.Thread(target=open_tk_visualizer_in_background, kwargs={'initial_board': board})
            display_t.daemon = True
            display_t.start()

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            move = agent_white.get_move(board)
        else:
            move = agent_black.get_move(board)
        board.push(move)

        # Test board to vector:
        # vec = NNChessAgent.board_to_vector(board)
        # new_board = NNChessAgent.vector_to_board(vec)
        # old_fen = board.fen()
        # new_fen = new_board.fen()
        #
        # board_position, turn, castling, _, _, _ = old_fen.split()
        # new_board_position, new_turn, new_castling, _, _, _ = new_fen.split()
        # assert board_position == new_board_position
        # assert turn == new_turn
        # assert castling == new_castling

    outcome = board.outcome()

    if is_any_human:
        display_board(board)
        input("Game is over, outcome is {}, press any key to continue".format(outcome))
        Visualizer.visualizer.quit()

    return outcome


def play_multiple_chess_games(agent_white, agent_black, num_games=300):
    white_wins = 0
    black_wins = 0
    for _ in range(num_games):
        outcome = play_chess_game(agent_white, agent_black)
        if outcome.winner:
            white_wins += 1
        elif outcome.winner is False:
            black_wins += 1
    print(
        f"{num_games} games have been played\n{agent_white.agent_type} as White won {white_wins} games\n{agent_black.agent_type} as Black won {black_wins} games\nand {num_games - white_wins - black_wins} games where drawen")


def main():
    stockfish_agent = StockfishChessAgent()
    random_agent = RandomChessAgent()
    human_agent = HumanChessAgent()
    nn_agent = NNChessAgent()
    play_multiple_chess_games(nn_agent, random_agent, num_games=3)


if __name__ == '__main__':
    # profiler = cProfile.Profile()
    # profiler.enable()
    main()
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumtime')
    # stats.print_stats()
