from matplotlib import pyplot as pp
from matplotlib.backends.backend_pdf import PdfPages


class Plot:
    def __init__(self):
        self.ax = pp.axes()
        self.ax.set_xlabel('Games spent learning')
        self.ax.set_ylabel('Percentage win or draw')

    def plot_evaluations(self, series, evaluations, variances, confidences,
                         label):
        pp.errorbar(series, evaluations, yerr=confidences, label=label)

    def save(self):
        pdf = PdfPages('plot.pdf')

        pdf.savefig()

        pdf.close()
