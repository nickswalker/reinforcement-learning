import copy
import sys
from typing import Tuple

import numpy as np
import scipy as scipy
import scipy.stats

from agent.q_learning import QLearning
from agent.sarsa_agent import SarsaAgent
from agent.true_online_sarsa_lambda import TrueOnlineSarsaLambda
from gridworld import ReachExit, GridWorld, Task, Domain, Direction

evaluation_period = 50
significance_level = 0.05

stochasticity = 0.0


def main():
    experiment_num = int(sys.argv[1])
    num_evaluations = int(sys.argv[2])
    num_trials = int(sys.argv[3])
    global stochasticity
    stochasticity = float(sys.argv[4])

    def save(name, results):
        data = np.c_[results]
        np.savetxt("results/" + str(stochasticity) + "/n" + str(num_trials) + "_" + name + ".csv", data,
                   fmt=["%d", "%f", "%f", "%f"],
                   delimiter=",")

    if experiment_num == 0:
        factory = agent_factory(q_learning=True)
        q_learning_results = run_experiment(num_trials, num_evaluations, factory)
        save("Q-learning", q_learning_results)
    if experiment_num == 1:
        factory = agent_factory()
        standard_results = run_experiment(num_trials, num_evaluations, factory)
        save("Sarsa", standard_results)
    if experiment_num == 2:
        factory = agent_factory(expected=True)
        expected_results = run_experiment(num_trials, num_evaluations, factory)
        save("Expected Sarsa", expected_results)
    if experiment_num == 4:
        factory = agent_factory(true_online=True, lmbda=0.10)
        true_online_results = run_experiment(num_trials, num_evaluations, factory)
        save("True Online Sarsa λ=0.1", true_online_results)
    if experiment_num == 5:
        factory = agent_factory(true_online=True, lmbda=0.50)
        true_online_sarsa_lambda = run_experiment(num_trials, num_evaluations, factory)
        save("True Online Sarsa λ=0.5", true_online_sarsa_lambda)
    if experiment_num == 5:
        factory = agent_factory(true_online=True, lmbda=0.90)
        true_online_sarsa_lambda = run_experiment(num_trials, num_evaluations, factory)
        save("True Online Sarsa λ=0.9", true_online_sarsa_lambda)


def run_experiment(num_trials, num_evaluations,
                   agent_factory
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
                                                 agent_factory):
            evaluation = evaluate(table, agent_factory)
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


def plot_trajectory(trajectory):
    result_grid = trajectory[0][0].map
    for state, action in trajectory:
        if action.direction == Direction.up:
            act_string = "↑"
        if action.direction == Direction.right:
            act_string = "→"
        if action.direction == Direction.down:
            act_string = "↓"
        if action.direction == Direction.left:
            act_string = "←"
        result_grid[state.y][state.x] = act_string

    result = "___________\n"
    for y in reversed(range(0, len(result_grid))):
        result += "|"
        for x in range(0, len(result_grid[0])):
            result += " " + str(result_grid[y][x])
        result += "|\n"
    result += "------------"
    return result


def evaluate(table, agent_factory) -> float:
    domain, task = configure_gridworld()
    agent = agent_factory(domain, task)
    agent.value_function = table
    agent.epsilon = 0.0
    agent.alpha = 0.0
    cumulative_reward = 0.0
    terminated = False
    max_steps = 200
    current_step = 0

    trajectory = []

    while not terminated:
        current_step += 1
        agent.act()

        trajectory.append((agent.previousstate, agent.previousaction))

        if task.stateisfinal(domain.get_current_state()) or current_step > max_steps:
            terminated = True
            domain.reset()
            cumulative_reward = agent.get_cumulative_reward()
            agent.episode_ended()

    # print(plot_trajectory(trajectory))
    print(cumulative_reward)
    return cumulative_reward


def train_agent(evaluation_period, num_stops, agent_factory):
    """
    Trains an agent, periodically yielding the agent's q-table
    :param evaluation_period:
    :param num_stops:
    :param initial_value:
    :param epsilon:
    :param alpha:
    :return:
    """
    domain, task = configure_gridworld()
    agent = agent_factory(domain, task)

    stops = 0
    for i in range(0, evaluation_period * num_stops):
        if i % evaluation_period is 0:
            print(i)
            stops += 1
            yield i, copy.deepcopy(agent.value_function)

        if num_stops == stops:
            return
        terminated = False
        max_steps = 200
        current_step = 0
        while not terminated:
            current_step += 1
            agent.act()

            if task.stateisfinal(domain.get_current_state()):
                final_state = domain.get_current_state()
                agent.episode_ended()
                domain.reset()
                terminated = True


def configure_gridworld() -> Tuple[Domain, Task]:
    domain = GridWorld(10, 7, agent_x_start=0, agent_y_start=3, wind=True,
                       wind_strengths=[0, 0, 0, 1, 1, 1, 2, 1, 1, 0], stochasticity=stochasticity)
    domain.place_exit(7, 3)
    task = ReachExit(domain)
    return domain, task


def agent_factory(initial_value=0.5,
                  epsilon=0.1,
                  alpha=0.2,
                  lmbda=0.95,
                  expected=False, true_online=False, q_learning=False):
    def generate_agent(domain, task):
        if true_online:
            agent = TrueOnlineSarsaLambda(domain, task, expected, lamb=lmbda)
        elif q_learning:
            agent = QLearning(domain, task)
        else:
            agent = SarsaAgent(domain, task, epsilon=epsilon, alpha=alpha, expected=expected)
        return agent

    return generate_agent


if __name__ == '__main__':
    main()
