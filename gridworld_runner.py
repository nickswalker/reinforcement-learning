import copy
import sys

import numpy as np
import scipy as scipy
import scipy.stats

from agent.gradient_descent_sarsa import TrueOnlineSarsaLambda
from agent.sarsa_agent import SarsaAgent
from gridworld import ReachExit, GridWorld

evaluation_period = 50
significance_level = 0.05


def main():
    experiment_num = int(sys.argv[1])
    num_evaluations = int(sys.argv[2])
    num_trials = int(sys.argv[3])

    def save(name, results):
        data = np.c_[results]
        np.savetxt("results/n" + str(num_trials) + "_" + name + ".csv", data, fmt=["%d", "%f", "%f", "%f"],
                   delimiter=",")

    if experiment_num == 0:
        standard_results = run_experiment(num_trials, num_evaluations)
        save("standard", standard_results)
    if experiment_num == 1:
        expected_results = run_experiment(num_trials, num_evaluations, expected=True)
        save("expected", expected_results)
    if experiment_num == 2:
        approximation_results = run_experiment(num_trials, num_evaluations, approximation=True)
    if experiment_num == 3:
        true_online_results = run_experiment(num_trials, num_evaluations, true_online=True)


def run_experiment(num_trials, num_evaluations,
                   initial_value=0.5,
                   epsilon=0.1,
                   alpha=0.2,
                   expected=False,
                   true_online=True
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
                                                 alpha=alpha,
                                                 expected=expected,
                                                 true_online=true_online):
            evaluation = evaluate(table, true_online=true_online)
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


def evaluate(table, true_online=False) -> float:
    domain = GridWorld(10, 10)
    domain.place_exit(9, 9)
    task = ReachExit(domain)
    if true_online:
        agent = TrueOnlineSarsaLambda(domain, task, epsilon=0.0, alpha=0.0)
    else:
        agent = SarsaAgent(domain, task, epsilon=0.0, alpha=0.0)
    agent.value_function = table


    agent = SarsaAgent(domain, task)

    cumulative_rewards = []
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
                alpha=0.2,
                expected=False, true_online=False):
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
    domain.place_exit(9, 9)
    task = ReachExit(domain)

    if true_online:
        agent = TrueOnlineSarsaLambda(domain, task, expected)
    else:
        agent = SarsaAgent(domain, task, epsilon=epsilon, alpha=alpha, expected=expected)

    stops = 0
    for i in range(0, evaluation_period * num_stops):
        if i % evaluation_period is 0:
            print(i)
            stops += 1
            yield i, copy.deepcopy(agent.value_function)

        if num_stops == stops:
            return
        terminated = False
        while not terminated:
            agent.act()
            # print(domain.current_state())
            if task.stateisfinal(domain.get_current_state()):
                final_state = domain.get_current_state()
                domain.reset()
                terminated = True



if __name__ == '__main__':
    main()
