from connect4 import *
from neural_net import *
import openpyxl

wb = openpyxl.load_workbook("testrun.xlsx")
network_data = wb.active
network_data.append(["", "", ""])


network_amount = 10
all_networks = []
# create a generation of networks
for j in range(network_amount):
    new_network = [
        Dense(board_width * board_height, 10),
        ReLUActivation(),
        Dense(10, 10),
        ReLUActivation(),
        Dense(10, 10),
        ReLUActivation(),
        Dense(10, 10),
        ReLUActivation(),
        Dense(10, 10),
        ReLUActivation(),
        Dense(10, 3),
        SigmoidActivation()
    ]
    all_networks.append(new_network)
network_scores = numpy.zeros(network_amount)


def train_connect_4(window, board, games_played, epochs, batch_size, learning_rate, networks,
                    full_inputs=numpy.zeros((1, board_width * board_height, 1)),
                    full_outputs=numpy.zeros((1, 3, 1))
                    ):
    global best_results
    # each network in the generation plays a game against each other network
    for i in range(games_played):
        for score1, nn1 in enumerate(all_networks):
            for score2, nn2 in enumerate(all_networks):
                x, y = play_game(window, board.copy(),
                                 "minmax", "minmax",
                                 depth_1=1, depth_2=1,
                                 nn1=nn1, nn2=nn2)
                if y[-1][0] == 1:
                    network_scores[score1] += 1
                elif y[-1][1] == 1:
                    network_scores[score2] += 1
                # prioritise data later in the game
                indices = numpy.arange(-len(y) + 1, 1)
                indices = numpy.exp(indices) * 0.25 + 0.75
                y = (y.T * indices).T
                full_inputs = numpy.vstack([full_inputs, x])
                full_outputs = numpy.vstack([full_outputs, y])
    # each network trains using gradient descent
    for nn in networks:
        train(nn, epochs, batch_size, learning_rate, full_inputs, full_outputs)
    return full_inputs, full_outputs


best_results = [[0], [0], [0]]
best_total_results = [[0], [0], [0]]
bestNN = all_networks[0]
best_generation = all_networks
inputs = numpy.zeros((1, board_width * board_height, 1))
outputs = numpy.zeros((1, 3, 1))
generation = 0
while True:
    print(f"generation: {generation}, "
          f"best results: {best_results}")
    network_scores = numpy.zeros(network_amount)
    # play and train each network
    inputs, outputs = train_connect_4(main_window, main_board, 1, 500, 100, 1, all_networks,
                                      full_inputs=inputs, full_outputs=outputs)
    # delete half of all data
    reduced_length = int(0.5 * len(inputs))
    inputs = numpy.delete(inputs, slice(reduced_length), 0)
    outputs = numpy.delete(outputs, slice(reduced_length), 0)
    # test each network by playing 100 games
    total_results = [[0], [0], [0]]
    for network in all_networks:
        results = [[0], [0], [0]]
        for j in range(50):
            X, Y = play_game(main_window, main_board.copy(),
                             "minmax", "minmax",
                             depth_1=1, depth_2=1,
                             nn1=network, heuristic_2="random")
            results += Y[0]
        for j in range(50):
            X, Y = play_game(main_window, main_board.copy(),
                             "minmax", "minmax",
                             depth_1=1, depth_2=1,
                             heuristic_1="random", nn2=network)
            Y_temporary = Y
            for column in Y_temporary:
                column[[0, 1]] = column[[1, 0]]
            Y = Y_temporary
            results += Y[0]
        print(results)
        total_results += results
        # save best network
        if results[0] > best_results[0]:
            best_results = results
            numpy.save("bestNN", network)
            bestNN = network
    # save best generation of networks
    if total_results[0] > best_total_results[0]:
        best_total_results = total_results
        numpy.save("best_generation", all_networks)
        best_generation = all_networks
    print("total results:", numpy.reshape(total_results, 3).tolist())
    # add data to excel sheet
    network_data.append(numpy.reshape(total_results, 3).tolist())
    wb.save("testrun.xlsx")
    generation += 1