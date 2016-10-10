import copy
import sys

import numpy as np
import scipy as scipy
import scipy.stats

from agent.sarsa_agent import SarsaAgent
from gridworld import ReachExit, GridWorld, GridItem

evaluation_period = 1000
evaluation_trials = 100
significance_level = 0.05


def main():
    num_evaluations = int(sys.argv[1])
    num_trials = int(sys.argv[2])
    experiment_num = int(sys.argv[3])

    def save(name, results):
        data = np.c_[results]
        np.savetxt("results/n" + str(num_trials) + "_" + name + ".csv", data, fmt=["%d", "%f", "%f", "%f"],
                   delimiter=",")

    if experiment_num == 0:
        book_results = run_evaluations(num_trials, num_evaluations)
        save("standard", book_results)


def run_evaluations(num_trials, num_evaluations,
                    initial_value=0.5,
                    epsilon=0.1,
                    alpha=0.2,
                    ):
    assert num_trials > 1
    evaluations_mean = []
    evaluations_variance = []
    series = [i * evaluation_period for i in range(0, num_evaluations)]
    n = 0
    for i in range(0, num_trials):
        print("trial " + str(i))
        j = int(0)
        n += 1
        for (num_episodes, table) in train_agent(evaluation_period,
                                                 num_evaluations,
                                                 initial_value=initial_value,
                                                 epsilon=epsilon,
                                                 alpha=alpha):
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

    evaluations_variance = [variance / (n - 1) for variance in
                            evaluations_variance]

    confidences = []
    for (mean, variance) in zip(evaluations_mean, evaluations_variance):
        crit = scipy.stats.t.ppf(1.0 - significance_level, n - 1)
        width = crit * np.math.sqrt(variance) / np.math.sqrt(n)
        confidences.append(width)

    return series, evaluations_mean, evaluations_variance, confidences


def evaluate(table) -> float:
    domain = GridWorld(10, 10)
    domain.map[9][9] = GridItem.exit
    task = ReachExit(domain)
    agent = SarsaAgent(domain, task)
    agent.table = table
    agent.epsilon = 0.0
    agent.alpha = 0.0

    cumulative_rewards = []
    for i in range(0, evaluation_trials):
        terminated = False
        max_steps = 200
        current_step = 0
        while not terminated:
            current_step += 1
            agent.act()

            if task.stateisfinal(domain.get_current_state()) or current_step > max_steps:
                terminated = True
                domain.reset()
                cumulative_rewards.append(agent.get_cumulative_reward())
                agent.episode_ended(domain.get_current_state())

    return np.mean(cumulative_rewards)


def train_agent(evaluation_period, num_stops, initial_value=0.5,
                epsilon=0.1,
                alpha=0.2):
    """
    Trains an agent, periodically yielding the agent's q-table
    :param evaluation_period:
    :param num_stops:
    :param initial_value:
    :param epsilon:
    :param alpha:
    :return:
    """
    domain = GridWorld(10, 10)
    domain.map[9][9] = GridItem.exit

    task = ReachExit(domain)

    agent = SarsaAgent(domain, task, epsilon=epsilon, alpha=alpha)

    stops = 0
    for i in range(0, evaluation_period * num_stops):
        if i % evaluation_period is 0:
            print(i)
            stops += 1
            yield i, copy.deepcopy(agent.state_action_value_table)

        if num_stops == stops:
            return
        terminated = False
        while not terminated:
            agent.act()
            # print(domain.get_current_state())
            if task.stateisfinal(domain.get_current_state()):
                agent.episode_ended(domain.get_current_state())
                domain.reset()
                break


main()
