import sys

import numpy as np

from tic_tac_toe_plotting import Plot


def main():
    figure_num = int(sys.argv[1])
    n = 2
    plot = Plot('Performance over time, n=' + str(n) + ' p=' + str(0.1), "Steps", "Reward")

    def plot_experiment(name, n):
        series, evaluations_mean, variances, confidences = np.loadtxt('results/n' + str(n) + '_' + name + '.csv',
                                                                      delimiter=",", dtype=("int,float,float,float"),
                                                                      unpack=True)
        plot.plot_evaluations(series, evaluations_mean, variances, confidences,
                              name)

    if figure_num == 0:
        plot_experiment("standard", n)
    if figure_num == 1:
        plot_experiment("expected", n)

    plot.save(figure_num)


main()
