import sys

import numpy as np

from tic_tac_toe_plotting import Plot


def main():
    figure_num = int(sys.argv[1])
    n = 200

    stoch = (figure_num * 2) / 10.0

    plot = Plot('Performance over time, stochasticity=' + str(stoch), "Episodes",
                "Reward")

    def plot_experiment(name, n):
        try:
            series, evaluations_mean, variances, confidences = np.loadtxt(
                'results/' + str(stoch) + '/n' + str(n) + '_' + name + '.csv',
                delimiter=",", dtype=("int,float,float,float"),
                unpack=True)
            plot.plot_evaluations(series, evaluations_mean, variances, confidences,
                                  name)
        except Exception:
            pass

    if figure_num in range(0, 6):
        plot_experiment("Q-learning", n)
        plot_experiment("Sarsa", n)
        plot_experiment("Expected Sarsa", n)
        plot_experiment("True Online Sarsa λ=0.1", n)
        plot_experiment("True Online Sarsa λ=0.5", n)
        plot_experiment("True Online Sarsa λ=0.8", n)

    plot.save(figure_num)


main()
