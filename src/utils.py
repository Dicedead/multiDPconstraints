from definitions import *
from multi_dp_mixture.piecewise_affine import DIAGONAL

def plot_multiple_functions(
        f_arr: List[TradeOffFunction],
        labels: List[str],
        start=0,
        end=1,
        num_points=100
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
    :type start: First point to plot. Defaults to 0.
    :type start: float, optional
    :type end: Last point to plot. Defaults to 1.
    :type end: float, optional
    :param num_points: The granularity of the plot, specifying the number of
                       sample points to generate within the range [start, end].
                       Defaults to 100.
    :type num_points: int, optional
    :return: None
    """
    assert len(f_arr) == len(labels)

    x = np.linspace(start, end, num_points)
    fig = plt.figure()
    ax = fig.add_subplot()
    for f, label in zip(f_arr, labels):
        plt.plot(x, f(x), label=label)

    plt.plot(x, DIAGONAL(x), "k--")
    plt.legend()
    ax.set_aspect('equal', adjustable='box')
    ax.set_autoscale_on(False)

    plt.show()