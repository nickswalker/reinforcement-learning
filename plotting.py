from matplotlib import pyplot as pp
from matplotlib.backends.backend_pdf import PdfPages


def plot(winner):
    to_plot = {}
    for i in range(0, len(winner)):
        item = winner[i]
        list = to_plot.get(item, [])
        list.append(i)
        to_plot[item] = list

    markers = {}
    markers["X"] = "x"
    markers["O"] = "o"
    markers[None] = "D"

    position = {}
    position["X"] = 0
    position["O"] = 0.5
    position[None] = -0.5

    for (key, value) in to_plot.items():
        pp.scatter(value, len(value) * [position[key]], marker=markers[key], s=80)
    pp.show()


def plot_evaluations(series, evaluations, confidences):
    pp.errorbar(series, evaluations, yerr=confidences)
    ax = pp.axes()
    ax.set_xlabel('Games spent learning')
    ax.set_ylabel('Percentage win or draw')

    pdf = PdfPages('multipage.pdf')

    pdf.savefig()

    pdf.close()
