import pickle

from tic_tac_toe import TicTacToeState


def save_table(table):
    with open('learned.pickle', 'wb') as handle:
        pickle.dump(table, handle)


def load_table():
    with open('learned.pickle', 'rb') as handle:
        return pickle.load(handle)


def describe_table(table):
    num_with_value = {}
    for (key, value) in table.items():
        existing_value = num_with_value.get(value, 0.0)
        num_with_value[value] = int(existing_value + 1)
    return num_with_value


def query(table):
    representation = [[None] * 3] * 3
    while True:
        for y in range(0, 3):
            for x in range(0, 3):
                in_value = input(str(x) + ", " + str(y))
                if in_value == " ":
                    in_value = None
                representation[y][x] = in_value

        state = TicTacToeState(representation)
        print(table.get(state))
