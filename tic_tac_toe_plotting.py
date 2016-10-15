from matplotlib import pyplot as pp
from matplotlib.backends.backend_pdf import PdfPages


class Plot:
    def __init__(self, title: str, x_label: str, y_label: str):
        self.ax = pp.axes()
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.ax.set_ylim([-200, 50])
        self.markers = ["s", "o", "D", "^", "8", "h"]

    def plot_evaluations(self, series, evaluations, variances, confidences,
                         label):
        pp.errorbar(series, evaluations, yerr=confidences, label=label, marker=self.markers.pop())

    def save(self, name):
        pp.legend(loc="lower right")
        pdf = PdfPages('results/' + str(name) + '.pdf')

        pdf.savefig()

        pdf.close()
