import sys

import numpy as np

from tic_tac_toe_plotting import Plot


def main():
    figure_num = int(sys.argv[1])
    n = 10
    plot = Plot('Performance over time, n=' + str(n) + ' p=' + str(0.1))

    def plot_experiment(name, n):
        series, evaluations_mean, variances, confidences = np.loadtxt('results/n' + str(n) + '_' + name + '.csv',
                                                                      delimiter=",")
        plot.plot_evaluations(series, evaluations_mean, variances, confidences,
                              name)

    if figure_num == 0:
        plot_experiment("standard", n)
    elif figure_num == 1:
        plot_experiment("standard", n)
        plot_experiment("optimistic", n)
        plot_experiment("pessimistic", n)
    elif figure_num == 2:
        plot_experiment("standard", n)
        plot_experiment("epsilon", n)
    elif figure_num == 3:
        plot_experiment("standard", n)
        plot_experiment("backtrace", n)
    elif figure_num == 4:
        plot_experiment("standard", n)
        plot_experiment("random-tie-breaking", n)
    elif figure_num == 5:
        plot_experiment("standard", n)
        plot_experiment("self-play", n)

    plot.save(figure_num)

main()
