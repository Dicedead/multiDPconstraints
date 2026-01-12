from base.definitions import *
from base.tradeoff_function import TradeOffFunction
from multi_dp_mixture.piecewise_affine import DIAGONAL

COLOR_1 = '#377eb8'
COLOR_2 = '#ff7f00'
COLOR_3 = '#4daf4a'

COLORBLIND_FRIENDLY_PALETTE =  \
        [COLOR_1, COLOR_2, COLOR_3,
         '#f781bf', '#a65628', '#984ea3',
         '#999999', '#e41a1c', '#dede00']


def plot_multiple_functions(
        f_arr: List[TradeOffFunction],
        labels: List[str],
        linestyles: List[str] = None,
        colors: List[str] = None,
        orders: List[int] = None,
        start=0,
        end=1,
        num_points=100,
        save_to: str = None
):
    """
    Plots multiple functions on the same graph, providing a visual comparison
    between a list of given function objects and their respective labels.

    :param f_arr: A list of PiecewiseAffine objects, where each object represents
                  a function to be plotted.
    :type f_arr: List[PiecewiseAffine]
    :param labels: A list of labels corresponding to each function in f_arr,
                   which will be used for the plot's legend.
    :type labels: List[str]
    :param linestyles: A list of linestyles to be used for each function in f_arr.
    :type linestyles: List[str], optional. Defaults to solid style for all functions.
    :param colors: A list of colors to be used for each function in f_arr.
    :type colors: List[str], optional. Defaults to the colorblind palette above.
    :param start: First point to plot. Defaults to 0.
    :type start: float, optional
    :param end: Last point to plot. Defaults to 1.
    :type end: float, optional
    :param num_points: The granularity of the plot, specifying the number of
                       sample points to generate within the range [start, end].
                       Defaults to 100.
    :type num_points: int, optional
    :param save_to: path where to save the figure. Defaults to None. If given,
                    does not display figure and only saves it to folder.
    :type save_to: str, optional
    :return: None
    """
    assert len(f_arr) == len(labels)

    if linestyles is None:
        linestyles = ["solid"] * len(f_arr)

    if colors is None:
        colors = COLORBLIND_FRIENDLY_PALETTE[:len(f_arr)]

    if orders is None:
        orders = range(len(f_arr))


    x = np.linspace(start, end, num_points)
    fig = plt.figure()
    ax = fig.add_subplot()
    for f, label, linestyle, color, order in zip(f_arr, labels, linestyles, colors, orders):
        plt.plot(x, f(x), label=label, linestyle=linestyle, color=color, zorder=order)

    plt.plot(x, DIAGONAL(x), "k--")
    plt.legend()
    ax.set_aspect('equal', adjustable='box')
    ax.set_autoscale_on(False)
    plt.xlabel("$\\beta_I$")
    plt.ylabel("$\\beta_{II}}$")
    #plt.tight_layout()

    if save_to is not None:
        plt.savefig(save_to, bbox_inches='tight',pad_inches = 0)
    else:
        plt.show()

    plt.close()

def plot_one_function(f: TradeOffFunction, label: str, start=0, end=1, num_points=100):
    plot_multiple_functions([f], [label], start, end, num_points)