import numpy as np

from base.tradeoff_function import TradeOffFunction
from base.utils import plot_multiple_functions
from f_dp_approximation.gaussian_tradeoff import GaussianTradeoff
from f_dp_approximation.laplace_tradeoff import LaplaceTradeoff
from main_theorems.heterogeneous_version import privacy_region_composition_double_dp_heterogeneous_comp, \
    privacy_region_composition_heterogeneous
from main_theorems.other_composition_theorems import (privacy_region_composition_exact, tv_of_eps_delta,
                                                      privacy_region_dp_composition_total_var,
                                                      privacy_region_approx_heterogeneous_composition_multi_slacks)
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


def mixture_example(alpha_1,eps_1, delta_1, eps_2, delta_2, title):
    """
    Plot an example of a mixture of trade-off functions.
    """
    alpha_2 = 1 - alpha_1
    f1 = SingleEpsDeltaTradeoff(eps_1, delta_1)
    f2 = SingleEpsDeltaTradeoff(eps_2, delta_2)
    f = TradeOffFunction.weighted_infimal_convolution([alpha_1, alpha_2], [f1, f2])

    plot_multiple_functions([f1, f2, f],
                                            [
                                             f"$({eps_1},{delta_1})$-DP",
                                             f"$({eps_2},{delta_2})$-DP",
                                             f"Mixture, weights ({alpha_1}, {alpha_2})"
                                             ],
                            save_to=png(title)
                            )

def heterogeneous_comparison(eps_1, eps_2, x, y, delta_slack_ls, title):
    """
    Plot the approximation of the heterogeneous composition of two single-DP mechanisms compared to
    the exact region.
    """
    f_ours = privacy_region_composition_heterogeneous(eps_1, eps_2, x, y)
    eps_ls = [eps_1] * x + [eps_2] * y
    delta_ls = [0] * (x+y)
    f_approx = privacy_region_approx_heterogeneous_composition_multi_slacks(eps_ls, delta_ls, delta_slack_ls)

    plot_multiple_functions(
        [f_ours, f_approx],
        [f"Theorem 1", f"Prior work"],
        save_to=png(title)
    )

def main_theorem_comparison(eps_1, delta_1, eps_2, delta_2, k, title):
    """
    Plot an instance of the double-DP main theorem's result compared to the corresponding
    single-DP exact composition and the DPTV region.
    """

    if delta_1 > delta_2:
        delta_1, delta_2 = delta_2, delta_1
        eps_1, eps_2 = eps_2, eps_1

    f1 = privacy_region_composition_double_dp_heterogeneous_comp(eps_1, delta_1, eps_2, delta_2, k)
    f_dp_1 = privacy_region_composition_exact(eps_1, delta_1, k)
    f_dp_2 = privacy_region_composition_exact(eps_2, delta_2, k)
    f_dp_single = TradeOffFunction.intersection([f_dp_1, f_dp_2])

    induced_tv = tv_of_eps_delta(eps_2, delta_2)
    f_dptv = privacy_region_dp_composition_total_var(eps_1, delta_1, induced_tv, k)
    f_dptv = TradeOffFunction.intersection([f_dptv, f_dp_2])

    plot_multiple_functions(
        [
         f1,
         f_dp_single,
         f_dptv
         ],
        [
         f"Theorems 2-3, $k = {k}$",
         f"Remark 1, $k = {k}$",
         f"Remark 2, $k = {k}$"
         ],
        save_to=png(title)
    )

def main_theorem_comparison_two_ks(eps_1, delta_1, eps_2, delta_2, k1, k2, title):
    """
    Plot an instance of the double-DP main theorem's result compared to the corresponding
    single-DP exact composition and the DPTV region.
    """

    if delta_1 > delta_2:
        delta_1, delta_2 = delta_2, delta_1
        eps_1, eps_2 = eps_2, eps_1

    f_double_dp_1 = privacy_region_composition_double_dp_heterogeneous_comp(eps_1, delta_1, eps_2, delta_2, k1)
    f_double_dp_2 = privacy_region_composition_double_dp_heterogeneous_comp(eps_1, delta_1, eps_2, delta_2, k2)

    f_dp_1_k1 = privacy_region_composition_exact(eps_1, delta_1, k1)
    f_dp_2_k1 = privacy_region_composition_exact(eps_2, delta_2, k1)
    f_dp_single_1 = TradeOffFunction.intersection([f_dp_1_k1, f_dp_2_k1])

    f_dp_1_k2 = privacy_region_composition_exact(eps_1, delta_1, k2)
    f_dp_2_k2 = privacy_region_composition_exact(eps_2, delta_2, k2)
    f_dp_single_2 = TradeOffFunction.intersection([f_dp_1_k2, f_dp_2_k2])

    induced_tv = tv_of_eps_delta(eps_2, delta_2)
    f_dptv_k1 = privacy_region_dp_composition_total_var(eps_1, delta_1, induced_tv, k1)
    f_dptv_k1 = TradeOffFunction.intersection([f_dptv_k1, f_dp_2_k1])
    f_dptv_k2 = privacy_region_dp_composition_total_var(eps_1, delta_1, induced_tv, k2)
    f_dptv_k2 = TradeOffFunction.intersection([f_dptv_k2, f_dp_2_k2])

    plot_multiple_functions(
        [
         f_double_dp_1,
         f_dp_single_1,
         f_dptv_k1,
         f_double_dp_2,
         f_dp_single_2,
         f_dptv_k2
         ],
        [
         f"Theorems 2-3, $k = {k1}$",
         f"Remark 1, $k = {k1}$",
         f"Remark 2, $k = {k1}$",
         f"Theorems 2-3, $k = {k2}$",
         f"Remark 1, $k = {k2}$",
         f"Remark 2, $k = {k2}$"
         ],
        [
            "solid",
            "dashed",
            "dashed",
            "solid",
            "dashed",
            "dashed",
        ],
        save_to=png(title)
    )

def main_theorem_example(eps_1, delta_1, eps_2, delta_2, k_ls, title):
    """
    Plot an instance of the main theorem for multiple values of k.
    """
    if delta_1 > delta_2:
        delta_1, delta_2 = delta_2, delta_1
        eps_1, eps_2 = eps_2, eps_1

    f_double_dp = MultiEpsDeltaTradeoff([eps_1, eps_2], [delta_1, delta_2])

    f_comp = []
    for k in k_ls:
        f_comp.append(privacy_region_composition_double_dp_heterogeneous_comp(eps_1, delta_1, eps_2, delta_2, k))

    plot_multiple_functions(
        [f_double_dp] + f_comp,
        [f"({eps_1},{delta_1}) and ({eps_2},{delta_2}) DP"] + [f"{k}-composition" for k in k_ls],
        save_to=png(title)
    )


def gaussian_tradeoff_approx(mu, title):
    """
    Plot the double-DP lower and upper approximations of the gaussian tradeoff
    function.
    """
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
            f"{float(mu):.2}-GDP",
            f"Approx below",
            "Approx above",
        ],
        [
            "solid",
            "dashed",
            "dashed"
        ],
        save_to=png(title)
    )

def gaussian_compos_approx(mu, k, title):
    """
    Plot the double-DP lower and upper approximations of the gaussian trade-off composition.
    """
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
            f"{k}-composition of {float(mu):.2}-GDP",
            f"{k}-comp. lower approx",
            f"{k}-comp. upper approx"
        ],
        [
            "solid",
            "dashed",
            "dashed"
        ],
        save_to=png(title)
    )

def gaussian_tradeoff_and_compos_approx(mu, k, title):
    """
    Plot the double-DP lower and upper approximations of the gaussian trade-off composition as well as
    the gaussian tradeoff function itself.
    """
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
            g_mu,
            g_mu_approx_below,
            g_mu_approx_above,
            g_mu_composed,
            g_mu_composed_below,
            g_mu_composed_above
        ],
        [
            f"{float(mu):.2}-GDP",
            f"Lower approx of {float(mu):.2}-GDP",
            f"Upper approx of {float(mu):.2}-GDP",
            f"{float(mu_composed):.2}-GDP ({k}-comp. of {float(mu):.2}-GDP)",
            f"{k}-comp. lower approx",
            f"{k}-comp. upper approx"
        ],
        [
            "solid",
            "dashed",
            "dashed",
            "solid",
            "dashed",
            "dashed"
        ],
        save_to=png(title)
    )

def gaussian_compos_approx_two_compos(mu, k1, k2, title):
    """
    Plot the double-DP lower and upper approximations of the gaussian trade-off composition for 2 values of k.
    """
    mu_composed_k1 = np.sqrt(k1) * mu
    mu_composed_k2 = np.sqrt(k2) * mu

    g_mu = GaussianTradeoff(mu)

    g_mu_composed_k1 = GaussianTradeoff(mu_composed_k1)
    g_mu_composed_k2 = GaussianTradeoff(mu_composed_k2)

    g_mu_approx_below = g_mu.approx_from_below()
    g_mu_approx_above = g_mu.approx_from_above()

    eps_ls = g_mu_approx_below.get_eps_list()
    delta_ls = g_mu_approx_below.get_delta_list()
    g_mu_composed_below = privacy_region_composition_double_dp_heterogeneous_comp(
        eps_ls[0], delta_ls[0], eps_ls[1], delta_ls[1], k1
    )
    g_mu_composed_below_2 = privacy_region_composition_double_dp_heterogeneous_comp(
        eps_ls[0], delta_ls[0], eps_ls[1], delta_ls[1], k2
    )

    eps_ls = g_mu_approx_above.get_eps_list()
    delta_ls = g_mu_approx_above.get_delta_list()
    g_mu_composed_above = privacy_region_composition_double_dp_heterogeneous_comp(
        eps_ls[0], delta_ls[0], eps_ls[1], delta_ls[1], k1
    )
    g_mu_composed_above_2 = privacy_region_composition_double_dp_heterogeneous_comp(
        eps_ls[0], delta_ls[0], eps_ls[1], delta_ls[1], k2
    )

    plot_multiple_functions(
        [
            g_mu_composed_k1,
            g_mu_composed_below,
            g_mu_composed_above,
            g_mu_composed_k2,
            g_mu_composed_below_2,
            g_mu_composed_above_2
        ],
        [
            f"{k1}-composition of {float(mu):.2}-GDP",
            f"{k1}-comp. lower approx",
            f"{k1}-comp. upper approx",
            f"{k2}-composition of {float(mu):.2}-GDP",
            f"{k2}-comp. lower approx",
            f"{k2}-comp. upper approx",
        ],
        [
            "solid",
            "dashed",
            "dashed",
            "solid",
            "dashed",
            "dashed"
        ],
        save_to=png(title)
    )


def gaussian_compos_approx_tradeoff_and_two_compos(mu, k1, k2, title):
    """
    Plot the double-DP lower and upper approximations of the gaussian trade-off composition for 2 values of k
    along with the approximation of the gaussian trade_off itself.
    """
    mu_composed_k1 = np.sqrt(k1) * mu
    mu_composed_k2 = np.sqrt(k2) * mu

    g_mu = GaussianTradeoff(mu)

    g_mu_composed_k1 = GaussianTradeoff(mu_composed_k1)
    g_mu_composed_k2 = GaussianTradeoff(mu_composed_k2)

    g_mu_approx_below = g_mu.approx_from_below()
    g_mu_approx_above = g_mu.approx_from_above()

    eps_ls = g_mu_approx_below.get_eps_list()
    delta_ls = g_mu_approx_below.get_delta_list()
    g_mu_composed_below = privacy_region_composition_double_dp_heterogeneous_comp(
        eps_ls[0], delta_ls[0], eps_ls[1], delta_ls[1], k1
    )
    g_mu_composed_below_2 = privacy_region_composition_double_dp_heterogeneous_comp(
        eps_ls[0], delta_ls[0], eps_ls[1], delta_ls[1], k2
    )

    eps_ls = g_mu_approx_above.get_eps_list()
    delta_ls = g_mu_approx_above.get_delta_list()
    g_mu_composed_above = privacy_region_composition_double_dp_heterogeneous_comp(
        eps_ls[0], delta_ls[0], eps_ls[1], delta_ls[1], k1
    )
    g_mu_composed_above_2 = privacy_region_composition_double_dp_heterogeneous_comp(
        eps_ls[0], delta_ls[0], eps_ls[1], delta_ls[1], k2
    )

    plot_multiple_functions(
        [
            g_mu,
            g_mu_approx_below,
            g_mu_approx_above,
            g_mu_composed_k1,
            g_mu_composed_below,
            g_mu_composed_above,
            g_mu_composed_k2,
            g_mu_composed_below_2,
            g_mu_composed_above_2
        ],
        [
            f"{float(mu):.2}-GDP",
            f"Lower approx of {float(mu):.2}-GDP",
            f"Upper approx of {float(mu):.2}-GDP",
            f"{k1}-composition of {float(mu):.2}-GDP",
            f"{k1}-comp. lower approx",
            f"{k1}-comp. upper approx",
            f"{k2}-composition of {float(mu):.2}-GDP",
            f"{k2}-comp. lower approx",
            f"{k2}-comp. upper approx",
        ],
        [
            "solid",
            "dashed",
            "dashed",
            "solid",
            "dashed",
            "dashed",
            "solid",
            "dashed",
            "dashed"
        ],
        save_to=png(title)
    )


def laplace_tradeoff_approx(eps, title):
    """
    Plot the double-DP lower and upper approximations of the Laplace trade-off composition.
    """
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
            f"Laplace({eps})-DP",
            "Approx below",
            "Approx above",
        ],
        save_to=png(title)
    )


if __name__ == "__main__":
    heterogeneous_comparison(eps_1=0.6,eps_2=0.4,x=3,y=2,delta_slack_ls=[0.001], title="heterogeneous_comparison")
    mixture_example(alpha_1 = 0.5, eps_1 = 1.3, delta_1 = 0.0, eps_2 = 0.5, delta_2 = 0.2, title="mixture_example")
    gaussian_tradeoff_approx(mu=1, title="gaussian_approx")
    gaussian_compos_approx(k=20, mu=0.05, title="gaussian_compos_approx")
    gaussian_compos_approx(k=3, mu=1, title="gaussian_compos_approx_2")
    gaussian_compos_approx_two_compos(k1=10, k2=3, mu=1, title="gaussian_2_compos")
    gaussian_tradeoff_and_compos_approx(k=3, mu=1, title="gaussian_tradeoff_and_compos_approx")
    main_theorem_comparison(eps_1 = 1.2, delta_1 = 0.0, eps_2 = 0.6, delta_2 = 0.2, k = 3, title="theorem_1_comparison")
    main_theorem_example(eps_1 = 1.2, delta_1 = 0.0, eps_2 = 0.6, delta_2 = 0.2, k_ls = [2, 3, 10, 20],
                         title="theorem_1_example")
    main_theorem_example(eps_1 = 0.3, delta_1 = 0.0, eps_2 = 0.15, delta_2 = 0.02, k_ls = [2, 3, 10, 20],
                         title="theorem_1_example_small_region")
    main_theorem_comparison_two_ks(eps_1 = 1.2, delta_1 = 0.0, eps_2 = 0.6, delta_2 = 0.2,
                                   k1=3, k2=10, title="theorem_1_comparison_two_ks")
    main_theorem_comparison_two_ks(eps_1 = 0.3, delta_1 = 0.0, eps_2 = 0.15, delta_2 = 0.02, k1=3, k2=20,
                                  title="theorem_1_comparison_two_ks_small_region")
    gaussian_compos_approx_tradeoff_and_two_compos(k1=3, k2=10, mu=1, title="gaussian_tradeoff_and_2_compos")
