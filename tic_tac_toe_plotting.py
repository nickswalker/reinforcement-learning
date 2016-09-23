import matplotlib
from matplotlib import pyplot as pp
from matplotlib.backends.backend_pdf import PdfPages

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

class Plot:
    def __init__(self, title):
        self.ax = pp.axes()
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        self.ax.set_xlabel('Games spent learning')
        self.ax.set_ylabel('Percentage win or draw')
        self.markers = {"s", "o", "D", "^", "8", "h"}

    def plot_evaluations(self, series, evaluations, variances, confidences,
                         label):
        pp.errorbar(series, evaluations, yerr=confidences, label=label, marker=self.markers.pop())

    def save(self, name):
        pp.legend(loc="lower right")
        pdf = PdfPages('results/' + str(name) + '.pdf')

        pdf.savefig()

        pdf.close()
