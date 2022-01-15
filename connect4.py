import pygame
import math
import random
import sys
from neural_net import *
import os

os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (620, 40)

board_width = 7
board_height = 6
main_board = numpy.zeros((board_height, board_width))

pygame.init()
blue_rgb = (8, 103, 136)
beige_rgb = (255, 241, 208)
red_rgb = (221, 28, 26)
yellow_rgb = (240, 200, 8)

square_size = 90
window_width = board_width * square_size
window_height = (board_height + 1) * square_size
window_size = (window_width, window_height)
main_window = pygame.display.set_mode(window_size, 0, 800)
pygame.display.set_caption("Connect-4")

pygame.font.init()
font = pygame.font.SysFont("Comic Sans MS", 50)


def draw_screen(window, board):
    for r in range(board_height):
        for c in range(board_width):
            square_x = c * square_size
            square_y = (r + 1) * square_size
            pygame.draw.rect(window, blue_rgb, (square_x, square_y, square_size, square_size))
            if board[r][c] == -1:
                circle_colour = red_rgb
            elif board[r][c] == 1:
                circle_colour = yellow_rgb
            else:
                circle_colour = beige_rgb
            circle_coordinates = ((c + 0.5) * square_size, (r + 1.5) * square_size)
            pygame.draw.circle(window, circle_colour, circle_coordinates, square_size / 2.5)


def is_win(board, piece):
    # horizontal check (left to right)
    for r in range(board_height):
        for c in range(board_width - 3):
            if board[r][c] == board[r][c + 1] == board[r][c + 2] == board[r][c + 3] == piece:
                return True

    # vertical check (top to bottom)
    for r in range(board_height - 3):
        for c in range(board_width):
            if board[r][c] == board[r + 1][c] == board[r + 2][c] == board[r + 3][c] == piece:
                return True

    # diagonal \ (top to bottom, left to right)
    for r in range(board_height - 3):
        for c in range(board_width - 3):
            if board[r][c] == board[r + 1][c + 1] == board[r + 2][c + 2] == board[r + 3][c + 3] == piece:
                return True

    # diagonal / (bottom to top, left to right)
    for r in range(3, board_height):
        for c in range(board_width - 3):
            if board[r][c] == board[r - 1][c + 1] == board[r - 2][c + 2] == board[r - 3][c + 3] == piece:
                return True


def is_draw(board):
    for c in board[0]:
        if c == 0:
            return False
    else:
        return True


def play_piece(board, c, piece):
    for r in reversed(board):
        if r[c] == 0:
            r[c] = piece
            break


def find_possible_moves(board):
    possible_moves = []
    for number, c in enumerate(board[0]):
        if c == 0:
            possible_moves.append(number)
    return possible_moves


def human_input(window, board, piece):
    thinking = True
    if piece == 1:
        piece_colour = yellow_rgb
    else:
        piece_colour = red_rgb
    while thinking:
        x_mouse = pygame.mouse.get_pos()[0]
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # check if player quits the game
                pygame.display.quit()
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                selected_column = int(x_mouse / square_size)
                if board[0][selected_column] == 0:  # check if move is legal
                    return selected_column
        window.fill(beige_rgb)
        pygame.draw.circle(window, piece_colour, (x_mouse, square_size / 2), square_size / 2.5)
        draw_screen(window, board)
        pygame.display.update()


def random_input(board):
    moves = find_possible_moves(board)
    selected_column = random.choice(moves)
    return selected_column


def get_game_input(window, board, agent, piece, heuristic_function, nn, depth):
    if agent == "human":
        return human_input(window, board, piece)
    elif agent == "random":
        return random_input(board)
    elif agent == "minmax":
        move = minmax(board, depth, -math.inf, math.inf, piece, heuristic_function, nn)[1]
        return move


def heuristic_analyse_threats(board):
    for r in range(board_height):
        for c in range(board_width):
            if board[board_height - r - 2][c] == 0 and board[board_height - r - 1][c] == 0:
                new_board = board.copy()
                if r % 2 == 1:  # odd row
                    new_board[board_height - r - 1][r] = 1
                    if is_win(new_board, 1):
                        return 1000 * r
                    new_board[board_height - r - 1][r] = -1
                    if is_win(new_board, -1):
                        return -10 * r
                else:  # even row
                    new_board[board_height - r - 1][r] = -1
                    if is_win(new_board, -1):
                        return -1000 * r
                    new_board[board_height - r - 1][r] = 1
                    if is_win(new_board, 1):
                        return 10 * r
    else:
        return 0


def heuristic_neural_network(board, neural_network):
    if isinstance(neural_network[0], list):
        length = len(neural_network)
        neural_network = neural_network[random.randrange(0, length - 1)]
    output = numpy.resize(board, (board_width * board_height, 1))
    for layer in neural_network:
        output = layer.forward(output)
    score = output[0] - output[1]
    return score


def get_heuristic_function(board, heuristic_function, nn):
    score = 0
    if heuristic_function == "threats":
        score += heuristic_analyse_threats(board)
    elif heuristic_function == "random":
        score += random.random()
    elif nn:
        score = heuristic_neural_network(board, nn)
    return score


def is_game_over(board):
    if is_win(board, 1):
        return 1, True
    elif is_win(board, -1):
        return -1, True
    elif is_draw(board):
        return 0, True
    else:
        return None, False


def minmax(board, depth, alpha, beta, current_player, heuristic_function, nn):
    result, game_over = is_game_over(board)
    if game_over:
        if game_over:  # a player has won or it's a draw
            return result * (1000000 + depth), None
    if depth == 0:  # final node
        return get_heuristic_function(board, heuristic_function, nn), None
    possible_moves = find_possible_moves(board)
    random.shuffle(possible_moves)
    # sort moves from centre column to edges
    possible_moves.sort(key=lambda x: abs(x - board_width // 2))
    best_move = possible_moves[0]
    # set value to wost case scenario
    value = -math.inf * current_player
    if current_player == 1:  # maximising player
        for move in possible_moves:
            new_board = board.copy()
            play_piece(new_board, move, current_player)
            new_value = minmax(new_board, depth - 1, alpha, beta, -1, heuristic_function, nn)[0]
            if new_value > value:
                value = new_value
                best_move = move
                alpha = max(value, alpha)
                if value >= beta:
                    break
    else:  # minimising player
        for move in possible_moves:
            new_board = board.copy()
            play_piece(new_board, move, current_player)
            new_value = minmax(new_board, depth - 1, alpha, beta, 1, heuristic_function, nn)[0]
            if new_value < value:
                value = new_value
                best_move = move
                beta = min(value, beta)
                if value <= alpha:
                    break
    return value, best_move


def play_game(window, board,
              agent_1, agent_2,
              heuristic_1=None, heuristic_2=None,
              nn1=None, nn2=None,
              depth_1=1, depth_2=1,
              wait=False):
    running = True
    turn = 0
    # create list to track all board states
    input_data = numpy.resize(board, (1, board_width * board_height, 1))
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # check if player quits the game
                pygame.display.quit()
                pygame.quit()
                sys.exit()
        window.fill(beige_rgb)
        draw_screen(window, board)
        pygame.display.update()
        if turn % 2 == 0:  # it's player 1's turn
            current_player = 1
            column = get_game_input(window, board, agent_1, current_player, heuristic_1, nn1, depth_1)
        else:  # it's player 2's turn
            current_player = -1
            column = get_game_input(window, board, agent_2, current_player, heuristic_2, nn2, depth_2)
        play_piece(board, column, current_player)
        result, game_over = is_game_over(board)
        # add current board state to list
        input_data = numpy.vstack([input_data, numpy.resize(board, (1, board_width * board_height, 1))])
        if game_over:
            if result == 1:  # player 1 wins
                text = "Player 1 (yellow) has won!"
                result_data = numpy.resize([1, 0, 0], (1, 3, 1))
            elif result == -1:  # player 2 wins
                text = "Player 2 (red) has won!"
                result_data = numpy.resize([0, 1, 0], (1, 3, 1))
            else:  # result == 0, it's a draw
                text = "It's a draw"
                result_data = numpy.resize([0, 0, 1], (1, 3, 1))
            window.fill(beige_rgb)
            draw_screen(window, board)
            window.blit(font.render(text, False, (0, 0, 0)), (0, 300))
            pygame.display.update()
            # delete board from turn 0, which is empty
            input_data = numpy.delete(input_data, 0, 0)
            # create a list for results, to match the list of board states
            full_result_data = result_data
            for data in range(len(input_data) - 1):
                full_result_data = numpy.vstack([full_result_data, result_data])
            if wait:
                pygame.time.wait(3000)
            return input_data, full_result_data
        turn += 1


main_window.fill(beige_rgb)
draw_screen(main_window, main_board)
pygame.display.update()