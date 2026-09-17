from base.definitions import *
from base.real_function import RealFunction
from base.tradeoff_function import TradeOffFunction
from multi_dp_mixture.piecewise_affine import DIAGONAL
from scipy.optimize import differential_evolution
from scipy.integrate import trapezoid

COLOR_1 = '#377eb8'
COLOR_2 = '#ff7f00'
COLOR_3 = '#4daf4a'
COLOR_4 = '#f781bf'
COLOR_5 = '#a65628'
COLOR_6 = '#984ea3'
COLOR_7 = '#999999'
COLOR_8 = '#e41a1c'
COLOR_9 = '#dede00'

COLORBLIND_FRIENDLY_PALETTE =  \
        [COLOR_1, COLOR_2, COLOR_3,
         COLOR_4, COLOR_5, COLOR_6,
         COLOR_7, COLOR_8, COLOR_9]

COLOR_PALETTE = COLORBLIND_FRIENDLY_PALETTE

_DPI = 200
_FIGSIZE = (5, 5)
_PLOTS_FOLDER = "../plots/"

plt.rcParams['text.usetex'] = True

def title_to_asset(title: str, extension: str = ".png", plots_folder: str = _PLOTS_FOLDER) -> str:
    """
    Preprocess title to save matplotlib figure as png in the correct folder.

    :param title: title of figure
    :type title: str

    :param extension: extension of figure
    :type extension: str

    :param plots_folder: folder to save figures in
    :type plots_folder: str

    :return: prepend folder and append .png
    :rtype: str
    """
    return plots_folder + title + extension

def _figsize_to_tikz_size(figsize: int):
    return str((figsize+1)*55)

def plot_multiple_functions(
        f_arr: List[TradeOffFunction],
        labels: List[str] = None,
        linestyles: List[str] = None,
        colors: List[str] = None,
        orders: List[int] = None,
        start=0,
        end=1,
        num_points=100,
        save_to: str = None,
        show_legend=True
):
    """
    Plots multiple functions on the same graph, providing a visual comparison
    between a list of given function objects and their respective labels.

    :param f_arr: A list of PiecewiseAffine objects, where each object represents
                  a function to be plotted.
    :type f_arr: List[PiecewiseAffine]
    :param labels: A list of labels corresponding to each function in f_arr,
                   which will be used for the plot's legend. If not provided,
                   no legend shown.
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
    show_legend = labels is not None
    assert not show_legend or len(f_arr) == len(labels)

    if not show_legend:
        labels = [""] * len(f_arr)

    if linestyles is None:
        linestyles = ["solid"] * len(f_arr)

    if colors is None:
        colors = COLOR_PALETTE[:len(f_arr)]

    if orders is None:
        orders = range(len(f_arr))


    x = np.linspace(start, end, num_points)
    fig = plt.figure(figsize=_FIGSIZE, dpi=_DPI)
    ax = fig.add_subplot()
    for f, label, linestyle, color, order in zip(f_arr, labels, linestyles, colors, orders):
        plt.plot(x, np.clip(f(x), 0, 1), label=label, linestyle=linestyle, color=color, zorder=order)

    plt.plot(x, DIAGONAL(x), "k--")
    ax.set_aspect('equal', adjustable='box')
    ax.set_autoscale_on(False)
    plt.xlabel(r"$\beta\textsubscript{I}$")
    plt.ylabel(r"$\beta\textsubscript{II}$")

    if show_legend:
        plt.legend()

    if save_to is not None:
        plt.savefig(title_to_asset(save_to), bbox_inches='tight',pad_inches = 0)
        m2t.save(title_to_asset(save_to, ".tex", _PLOTS_FOLDER + "tikz/"),
                 axis_width=_figsize_to_tikz_size(_FIGSIZE[0]),
                 axis_height=_figsize_to_tikz_size(_FIGSIZE[1]),
                 )
    else:
        plt.show()

    plt.close()

def plot_one_function(f: TradeOffFunction, label: str, start=0, end=1, num_points=100):
    plot_multiple_functions([f], [label], start, end, num_points)


def linf_distance(f: RealFunction, g: RealFunction, tol=1e-7) -> float:
    """
    Computes the Linf norm of f - g on [0, 1] using Differential Evolution.

    :param f: The first function.
    :type f: RealFunction

    :param g: The second function.
    :type g: RealFunction

    :param tol: The tolerance for the optimization algorithm.
    :type tol: float

    :return: The maximum absolute difference between f and g on [0, 1].
    """
    def objective(x):
        return -abs(f(x[0]) - g(x[0]))
    bounds = [(0.0, 1.0)]
    result = differential_evolution(objective, bounds, tol=tol)
    return -result.fun


def l1_distance(f: RealFunction, g: RealFunction, grid_points=1e5) -> float:
    """
    Calculate the L1 distance between two real-valued functions by with the trapezoid rule.

    :param f: The first function.
    :type f: RealFunction

    :param g: The second function.
    :type g: RealFunction

    :param grid_points: The number of points in the grid used to approximate
        the integral. Defaults to 100,000.

    :return: The L1 distance as a float.
    """
    x_grid = np.linspace(0.0, 1.0, int(grid_points))
    y_diff = np.abs(f(x_grid) - g(x_grid))
    return trapezoid(y_diff, x_grid)