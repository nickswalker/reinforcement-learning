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
            clean_name = name.replace("λ", "Lambda")
            plot.plot_evaluations(series, evaluations_mean, variances, confidences,
                                  clean_name)
        except Exception:
            pass

    if figure_num in range(0, 6):
        plot_experiment("Q-learning", n)
        plot_experiment("Sarsa", n)
        plot_experiment("Expected Sarsa", n)
        plot_experiment("True Online Sarsa λ=0.1", n)
        plot_experiment("True Online Sarsa λ=0.5", n)
        plot_experiment("True Online Sarsa λ=0.8", n)
    elif figure_num == 6:
        global stoch
        stoch = 1.0
        plot = Plot('Performance over time, stochasticity=' + str(stoch), "Episodes",
                    "Reward")
        plot_experiment("True Online Sarsa λ=0.0", n)

    elif figure_num == 7:
        global stoch
        stoch = 1.0
        plot = Plot('Performance over time, stochasticity=' + str(stoch), "Episodes",
                    "Reward")
        plot_experiment("Q-learning", 201)

    plot.save(figure_num)


main()
