import copy
import sys

import numpy as np
import scipy as scipy
from scipy import stats

from plotting import plot_evaluations
from tic_tac_toe import RandomAgent, WinTicTacToeTask, TicTacToeDomain, InteractiveAgent
from tic_tac_toe.learning_agent import LearningAgent

learning_agent_symbol = "X"
random_agent_symbol = "O"
evaluation_period = 200
evaluation_trials = 100
significance_level = 0.05


def main():
    num_evaluations = int(sys.argv[1])
    num_trials = int(sys.argv[2])
    assert num_trials > 1
    evaluations_mean = []
    evaluations_variance = []
    series = [i * evaluation_period for i in range(0, num_evaluations)]
    n = 0
    for i in range(0, num_trials):
        print("trial " + str(i))
        j = int(0)
        n += 1
        for (num_episodes, table) in train_agent(evaluation_period, num_evaluations):
            evaluation = evaluate(table)
            mean = None
            variance = None
            if j > len(evaluations_mean) - 1:
                evaluations_mean.append(0.0)
                evaluations_variance.append(0.0)
                mean = 0.0
                variance = 0.0
            else:
                mean = evaluations_mean[j]
                variance = evaluations_variance[j]

            delta = evaluation - mean
            mean += delta / n
            variance += delta * (evaluation - mean)

            evaluations_mean[j] = mean
            evaluations_variance[j] = variance
            j += 1

    evaluations_variance = [variance / (n - 1) for variance in evaluations_variance]

    confidences = []
    for (mean, variance) in zip(evaluations_mean, evaluations_variance):
        crit = scipy.stats.t.ppf(1.0 - significance_level, n - 1)
        width = crit * np.math.sqrt(variance) / np.math.sqrt(n)
        confidences.append(width)

    a = np.c_[series, evaluations_mean, evaluations_variance]
    np.savetxt("n" + str(num_trials) + "_experiment.csv", a, delimiter=",")

    plot_evaluations(series, evaluations_mean, confidences)


def evaluate(table) -> float:
    task = WinTicTacToeTask()
    domain = TicTacToeDomain(3)

    random_agent = RandomAgent(random_agent_symbol, domain, task)
    learning_agent = LearningAgent(learning_agent_symbol, domain, task)
    learning_agent.table = table
    learning_agent.epsilon = 0.0
    learning_agent.alpha = 0.0

    agents = [random_agent, learning_agent]

    win_count = 0

    for i in range(0, evaluation_trials):
        match_ended = False
        while not match_ended:
            for agent in agents:
                action = agent.choose_action(domain.current_state())
                domain.apply_action(action)
                if task.stateisfinal(domain.current_state()):
                    match_ended = True
                    final_state = domain.current_state()
                    domain.reset()
                    task_winner = task.winner(final_state)
                    if task_winner == learning_agent_symbol or task_winner is None:
                        win_count += 1

                    break

    return float(win_count) / float(evaluation_trials)


def train_agent(evaluation_period, num_stops):
    task = WinTicTacToeTask()
    domain = TicTacToeDomain(3)

    random_agent = RandomAgent(random_agent_symbol, domain, task)
    learning_agent = LearningAgent(learning_agent_symbol, domain, task)
    agents = [random_agent, learning_agent]

    stops = 0
    for i in range(0, evaluation_period * num_stops):
        if i % evaluation_period is 0:
            print(i)
            stops += 1
            yield i, copy.deepcopy(learning_agent.table)

        if num_stops == stops:
            return
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
                    for other_agent in unseen_agents:
                        other_agent.see_result(final_state)
                        other_agent.prepare_for_new_episode(domain.current_state())

                    break


def interactive(table):
    task = WinTicTacToeTask()
    domain = TicTacToeDomain(3)

    interactive_agent = InteractiveAgent(random_agent_symbol, domain, task)
    learning_agent = LearningAgent(learning_agent_symbol, domain, task)
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


main()
