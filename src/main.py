import numpy as np

from base.tradeoff_function import TradeOffFunction
from base.utils import plot_multiple_functions
from f_dp_approximation.gaussian_tradeoff import GaussianTradeoff
from f_dp_approximation.laplace_tradeoff import LaplaceTradeoff
from main_theorems.heterogeneous_version import privacy_region_composition_double_dp_heterogeneous_comp
from main_theorems.other_composition_theorems import privacy_region_composition_exact
from multi_dp_mixture.dp_functions import SingleEpsDeltaTradeoff, MultiEpsDeltaTradeoff


def png(title: str, plots_folder: str = "../plots/") -> str:
    """
    Preprocess title to save matplotlib figure as png in the correct folder.

    :param title: title of figure
    :type title: str

    :param plots_folder: folder to save figures in
    :type plots_folder: str

    :return: prepend folder and append .png
    :rtype: str
    """
    return plots_folder + title + ".png"


def mixture_example():
    """
    Plot an example of a mixture of trade-off functions.
    """
    alpha_1 = 0.5
    alpha_2 = 1 - alpha_1
    eps_1 = 1.3
    delta_1 = 0.0
    eps_2 = 0.5
    delta_2 = 0.2
    f1 = SingleEpsDeltaTradeoff(eps_1, delta_1)
    f2 = SingleEpsDeltaTradeoff(eps_2, delta_2)
    f = TradeOffFunction.weighted_infimal_convolution([alpha_1, alpha_2], [f1, f2])

    plot_multiple_functions([f1, f2, f],
                                            [f"$({eps_1},{delta_1})$-DP",
                                             f"$({eps_2},{delta_2})$-DP",
                                             f"Mixture, weights ({alpha_1}, {alpha_2})"
                                             ],
                            save_to=png("mixture_example")
                            )

def main_theorem_example():
    """
    Plot an instance of the double-DP main theorem's result compared to the corresponding
    single-DP exact composition trade-off functions.
    """
    eps_1 = 1.2
    delta_1 = 0.0
    eps_2 = 0.6
    delta_2 = 0.2
    k = 3

    f1 = privacy_region_composition_double_dp_heterogeneous_comp(eps_1, delta_1, eps_2, delta_2, k)
    fo = MultiEpsDeltaTradeoff([eps_1, eps_2], [delta_1, delta_2])
    f_dp_1 = privacy_region_composition_exact(eps_1, delta_1, k)
    f_dp_2 = privacy_region_composition_exact(eps_2, delta_2, k)

    plot_multiple_functions(
        [
         fo,
         f1,
         f_dp_1,
         f_dp_2
         ],
        [
         f"$({eps_1},{delta_1})$ and $({eps_2},{delta_2})$ DP",
         f"Double DP ${k}$ comp.",
         f"$({eps_1},{delta_1})$-DP {k} comp.",
         f"$({eps_2},{delta_2})$-DP {k} comp."
         ],
        save_to=png("theorem_1_example")
    )


def gaussian_tradeoff_approx():
    """
    Plot the double-DP lower and upper approximations of the gaussian tradeoff
    function.
    """
    mu = 1
    g_mu = GaussianTradeoff(mu)
    g_mu_approx_below = g_mu.approx_from_below()
    g_mu_approx_above = g_mu.approx_from_above()
    plot_multiple_functions(
        [
            g_mu,
            g_mu_approx_below,
            g_mu_approx_above,
        ],
        [
            f"${float(mu):.2}-GDP$",
            f"Approx below",
            "Approx above",
        ],
        save_to=png("gaussian_approx")
    )

def gaussian_compos_approx():
    """
    Plot the double-DP lower and upper approximations of the gaussian trade-off composition.
    """
    mu = 0.05
    k = 20
    mu_composed = np.sqrt(k) * mu
    g_mu = GaussianTradeoff(mu)
    g_mu_composed = GaussianTradeoff(mu_composed)
    g_mu_approx_below = g_mu.approx_from_below()
    g_mu_approx_above = g_mu.approx_from_above()

    eps_ls = g_mu_approx_below.get_eps_list()
    delta_ls = g_mu_approx_below.get_delta_list()
    g_mu_composed_below = privacy_region_composition_double_dp_heterogeneous_comp(
        eps_ls[0], delta_ls[0], eps_ls[1], delta_ls[1], k
    )

    eps_ls = g_mu_approx_above.get_eps_list()
    delta_ls = g_mu_approx_above.get_delta_list()
    g_mu_composed_above = privacy_region_composition_double_dp_heterogeneous_comp(
        eps_ls[0], delta_ls[0], eps_ls[1], delta_ls[1], k
    )

    plot_multiple_functions(
        [
            g_mu_composed,
            g_mu_composed_below,
            g_mu_composed_above
        ],
        [
            f"${float(mu_composed):.2}$-GDP ({k}-composition of ${float(mu):.2}$-GDP)",
            "Comp. approx from below",
            "Comp. approx from above"
        ],
        save_to=png("gaussian_compos_approx")
    )

def laplace_tradeoff_approx():
    """
    Plot the double-DP lower and upper approximations of the Laplace trade-off composition.
    """
    eps = 1
    laplace_eps = LaplaceTradeoff(eps)
    lap_eps_approx_below = laplace_eps.approx_from_below()
    laps_eps_approx_above = laplace_eps.approx_from_above()
    plot_multiple_functions(
        [
            laplace_eps,
            lap_eps_approx_below,
            laps_eps_approx_above
        ],
        [
            f"$Laplace({eps})-DP$",
            "Approx below",
            "Approx above",
        ],
        save_to=png("laplace_tradeoff")
    )


mixture_example()
gaussian_tradeoff_approx()
gaussian_compos_approx()
main_theorem_example()
