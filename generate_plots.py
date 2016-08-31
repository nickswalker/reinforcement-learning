import numpy as np

from plotting import Plot


def main():
    plot = Plot()

    series, evaluations_mean, variances, confidences = np.loadtxt("n10book.txt")
    plot.plot_evaluations(series, evaluations_mean, variances, confidences,
                          "book")


main()
