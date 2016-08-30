import pickle

import matplotlib.pyplot as pp

from tic_tac_toe import RandomAgent, WinTicTacToeTask, TicTacToeDomain, InteractiveAgent, TicTacToeState
from tic_tac_toe.learning_agent import LearningAgent


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


def plot(winner):
    to_plot = {}
    for i in range(0, len(winner)):
        item = winner[i]
        list = to_plot.get(item, [])
        list.append(i)
        to_plot[item] = list

    markers = {}
    markers["X"] = "x"
    markers["O"] = "o"
    markers[None] = "D"

    position = {}
    position["X"] = 0
    position["O"] = 0.5
    position[None] = -0.5

    for (key, value) in to_plot.items():
        size = [20 * 4 ** n for n in range(len(value))]
        pp.scatter(value, len(value) * [position[key]], marker=markers[key], s=80)
    pp.show()


evaluate()


def train_agent(episodes):
    task = WinTicTacToeTask()
    domain = TicTacToeDomain(3)
    # task = WinLadder()
    # domain = LadderDomain(5)

    random_agent = RandomAgent("X", domain, task)
    learning_agent = LearningAgent("O", domain, task)
    agents = [random_agent, learning_agent]

    winner = []

    for i in range(0, episodes):
        if i % 100 is 0:
            print(i)
        match_ended = False
        while not match_ended:
            unseen_agents = set(agents.copy())
            for agent in agents:
                unseen_agents.remove(agent)
                action = agent.choose_action(domain.current_state())
                domain.apply_action(action)
                # print(domain.current_state())
                if task.stateisfinal(domain.current_state()):
                    match_ended = True
                    final_state = domain.current_state()
                    domain.reset()
                    task_winner = task.winner(final_state)
                    winner.append(task_winner)
                    for other_agent in unseen_agents:
                        other_agent.see_result(final_state)
                        other_agent.prepare_for_new_episode(domain.current_state())

                    break

    print("Learning agent: " + str(winner.count(learning_agent.symbol)))
    print("Random agent: " + str(winner.count(random_agent.symbol)))
    print("Draw: " + str(winner.count(None)))

    plot(winner)

    return learning_agent.table


def interactive(table):
    task = WinTicTacToeTask()
    domain = TicTacToeDomain(3)

    interactive_agent = InteractiveAgent("X", domain, task)
    learning_agent = LearningAgent("O", domain, task)
    learning_agent.table = table
    learning_agent.epsilon = 0.0

    agents = [interactive_agent, learning_agent]

    while True:
        match_ended = False
        while not match_ended:
            for agent in agents:
                agent.see_result(domain.current_state())
                action = agent.choose_action(domain.current_state())
                domain.apply_action(action)
                print(str(domain.current_state()))
                if task.stateisfinal(domain.current_state()):
                    match_ended = True
                    for other_agent in agents:
                        if other_agent == agent:
                            pass
                        agent.see_result(domain.current_state())
                    domain.reset()
                    print(str(domain.current_state()))


table = train_agent(200000)
save_table(table)

query(table)
