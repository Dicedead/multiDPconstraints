import numpy as np

from base.tradeoff_function import TradeOffFunction
from base.utils import plot_multiple_functions, COLOR_1, COLOR_2, COLOR_3, COLOR_4, COLOR_5
from f_dp_approximation.approximations import l1_multi_dp_approx_above, l1_multi_dp_approx_below, \
    linf_multi_dp_approx_below, linf_multi_dp_approx_above
from f_dp_approximation.smooth_approximation.gaussian_tradeoff import GaussianTradeoff
from f_dp_approximation.smooth_approximation.laplace_tradeoff import LaplaceTradeoff
from f_dp_approximation.smooth_approximation.vmf_tradeoff import VonMisesFisherTradeoff
from main_theorems.heterogeneous_composition_paper import heter_comp_generalized
from main_theorems.heterogeneous_version import privacy_region_composition_double_dp_heterogeneous_comp, \
    privacy_region_composition_heterogeneous_two_constraints, privacy_region_composition_multi_dp
from main_theorems.other_composition_theorems import (privacy_region_composition_exact, tv_of_eps_delta,
                                                      privacy_region_dp_composition_total_var,
                                                      privacy_region_approx_heterogeneous_composition_multi_slacks)
from multi_dp_mixture.dp_functions import SingleEpsDeltaTradeoff, MultiEpsDeltaTradeoff

dotted_custom = (0, (1, 1))


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


def mixture_example(alpha_1, eps_1, delta_1, eps_2, delta_2, title):
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
                            [
                                "dashed",
                                "dashed",
                                "solid"
                            ],
                            save_to=png(title)
                            )


def heterogeneous_comparison(eps_1, eps_2, x, y, delta_slack_ls, title):
    """
    Plot the approximation of the heterogeneous composition of two single-DP mechanisms compared to
    the exact region.
    """
    f_ours = privacy_region_composition_heterogeneous_two_constraints(eps_1, eps_2, x, y)
    eps_ls = [eps_1] * x + [eps_2] * y
    delta_ls = [0] * (x + y)
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
            f_dptv_k1,
            f_dp_single_1,
            f_double_dp_2,
            f_dptv_k2,
            f_dp_single_2
        ],
        [
            f"Theorems 2-3, $k = {k1}$",
            f"Remark 2, $k = {k1}$",
            f"Remark 1, $k = {k1}$",
            f"Theorems 2-3, $k = {k2}$",
            f"Remark 2, $k = {k2}$",
            f"Remark 1, $k = {k2}$"
        ],
        [
            "solid",
            "dashed",
            dotted_custom,
            "solid",
            "dashed",
            dotted_custom,
        ],
        [
            COLOR_1,
            COLOR_1,
            COLOR_1,
            COLOR_2,
            COLOR_2,
            COLOR_2,
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
    g_mu_approx_below = g_mu.l1_smooth_approx_2dp_below()
    g_mu_approx_above = g_mu.l1_smooth_approx_2dp_above()
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
    g_mu_approx_below = g_mu.l1_smooth_approx_2dp_below()
    g_mu_approx_above = g_mu.l1_smooth_approx_2dp_above()

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
    g_mu_approx_below = g_mu.l1_smooth_approx_2dp_below()
    g_mu_approx_above = g_mu.l1_smooth_approx_2dp_above()

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

    g_mu_approx_below = g_mu.l1_smooth_approx_2dp_below()
    g_mu_approx_above = g_mu.l1_smooth_approx_2dp_above()

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

    g_mu_approx_below = g_mu.l1_smooth_approx_2dp_below()
    g_mu_approx_above = g_mu.l1_smooth_approx_2dp_above()

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
            g_mu_approx_above,
            g_mu_approx_below,
            g_mu_composed_k1,
            g_mu_composed_above,
            g_mu_composed_below,
            g_mu_composed_k2,
            g_mu_composed_above_2,
            g_mu_composed_below_2,
        ],
        [
            f"{float(mu):.2}-GDP",
            f"Upper approx of {float(mu):.2}-GDP",
            f"Lower approx of {float(mu):.2}-GDP",
            f"{k1}-composition of {float(mu):.2}-GDP",
            f"{k1}-comp. upper approx",
            f"{k1}-comp. lower approx",
            f"{k2}-composition of {float(mu):.2}-GDP",
            f"{k2}-comp. upper approx",
            f"{k2}-comp. lower approx",
        ],
        [
            "solid",
            dotted_custom,
            "dashed",
        ] * 3,
        [COLOR_1] * 3 + [COLOR_2] * 3 + [COLOR_3] * 3,
        save_to=png(title)
    )


def laplace_tradeoff_approx(eps, title):
    """
    Plot the double-DP lower and upper approximations of the Laplace trade-off composition.
    """
    laplace_eps = LaplaceTradeoff(eps)
    lap_eps_approx_below = laplace_eps.l1_smooth_approx_2dp_below()
    laps_eps_approx_above = laplace_eps.l1_smooth_approx_2dp_above()
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


def smooth_vs_nonsmooth_above_2dp_approx_gaussian(mu, title):
    """
    Compare 2-DP approximations of GDP from above, assuming smoothness vs not assuming smoothness.
    """
    gaussian = GaussianTradeoff(mu)
    smooth_approx = gaussian.l1_smooth_approx_2dp_above()
    nonsmooth_approx = l1_multi_dp_approx_above(gaussian, 2)

    plot_multiple_functions(
        [
            gaussian,
            nonsmooth_approx,
            smooth_approx
        ],
        [
            "Gaussian",
            "Non-smooth Approx",
            "Smooth Approx"
        ],
        [
            "solid",
            "dashed",
            "dotted"
        ],
        [
            COLOR_1,
            COLOR_2,
            COLOR_3
        ],
        save_to=png(title)
    )


def smooth_vs_nonsmooth_below_2dp_approx_gaussian(mu, title):
    """
    Compare 2-DP approximations of GDP from below, assuming smoothness vs not assuming smoothness.
    """
    gaussian = GaussianTradeoff(mu)
    smooth_approx = gaussian.l1_smooth_approx_2dp_below()
    nonsmooth_approx = l1_multi_dp_approx_below(gaussian, 2)
    better_approx = l1_multi_dp_approx_below(gaussian, 3)

    plot_multiple_functions(
        [
            gaussian,
            nonsmooth_approx,
            smooth_approx,
            better_approx
        ],
        [
            "Gaussian",
            "Non-smooth Approx",
            "Smooth Approx",
            "Better Approx"
        ],
        [
            "solid",
            "dashed",
            "dotted",
            "dashed"
        ],
        [
            COLOR_1,
            COLOR_2,
            COLOR_3,
            COLOR_4
        ],
        save_to=png(title)
    )


def two_dp_constraints():
    eps_1 = 1.2
    delta_1 = 0.0
    eps_2 = 0.6
    delta_2 = 0.15
    f1 = SingleEpsDeltaTradeoff(eps_1, delta_1)
    f2 = SingleEpsDeltaTradeoff(eps_2, delta_2)
    title = "two_dp_constraints"

    plot_multiple_functions([f1, f2],
                            [
                                f"$({eps_1},{delta_1})$-DP",
                                f"$({eps_2},{delta_2})$-DP",
                            ],
                            [
                                "solid",
                                "solid",
                            ],
                            save_to=png(title)
                            )


def gaussian_n_dp_approx():
    """
    Compare n-DP approximations of GDP.
    """
    mu = 1.
    n = 2
    gaussian = GaussianTradeoff(mu)
    below_approx = l1_multi_dp_approx_below(gaussian, n)
    above_approx = l1_multi_dp_approx_above(gaussian, n)

    eps_below = below_approx.get_eps_list()
    delta_below = below_approx.get_delta_list()

    eps_above = above_approx.get_eps_list()
    delta_above = above_approx.get_delta_list()

    f_below = [SingleEpsDeltaTradeoff(eps, delta) for eps, delta in zip(eps_below, delta_below)]
    f_above = [SingleEpsDeltaTradeoff(eps, delta) for eps, delta in zip(eps_above, delta_above)]

    f_arr = f_below + f_above + [gaussian]
    colors = [COLOR_3] * len(f_below) + [COLOR_2] * len(f_above) + [COLOR_1]
    linestyles = ["dashed"] * len(f_below) + ["dashed"] * len(f_above) + ["solid"]
    title = f"gaussian_{n}_dp_approx"

    plot_multiple_functions(
        f_arr=f_arr,
        colors=colors,
        linestyles=linestyles,
        save_to=png(title)
    )

    plot_multiple_functions(
        f_arr=[gaussian, below_approx, above_approx],
        labels=["1-GDP", f"{n}-DP approx below", f"{n}-DP approx above"],
        colors=[COLOR_1, COLOR_3, COLOR_2],
        linestyles=["solid", "dashed", "dashed"],
        save_to=png(title + "_maxed")
    )


def heterogeneous_plots():
    eps_1 = 1.2
    eps_2 = 0.6
    delta_1 = 0
    delta_2 = 0.15

    x = 2
    y = 3

    f_no_delta = privacy_region_composition_heterogeneous_two_constraints(eps_1, eps_2, x, y)
    eps_ls = [eps_1] * x + [eps_2] * y
    delta_ls = [delta_1] * x + [delta_2] * y

    f_1 = SingleEpsDeltaTradeoff(eps_1, delta_1)
    f_2 = SingleEpsDeltaTradeoff(eps_2, delta_2)
    f_2_no_delta = SingleEpsDeltaTradeoff(eps_2, 0)

    f_with_delta = heter_comp_generalized(eps_ls, delta_ls, False)

    title = "heterogeneous"

    plot_multiple_functions(
        [f_1, f_2_no_delta, f_no_delta],
        labels=[f"{eps_1}-DP", f"{eps_2}-DP", f"({x},{y})-composition of {eps_1}-DP & {eps_2}-DP"],
        save_to=png(title + "_no_delta")
    )

    plot_multiple_functions(
        [f_no_delta],
        labels=[f"({x},{y})-composition of {eps_1}-DP and {eps_2}-DP"],
        save_to=png(title + "_only_no_delta")
    )

    plot_multiple_functions(
        [f_1, f_2, f_with_delta],
        labels=[f"{(eps_1, delta_1)}-DP (a)", f"{(eps_2, delta_2)}-DP (b)", f"({x},{y})-composition of (a) and (b)"],
        save_to=png(title + "_with_delta")
    )


def mixture_test():
    eps_1 = 1.2
    eps_2 = 0.6
    delta_1 = 0.0
    delta_2 = 0.0

    alpha_1 = 0.5
    alpha_2 = 1 - alpha_1
    f1 = SingleEpsDeltaTradeoff(eps_1, delta_1)
    f2 = SingleEpsDeltaTradeoff(eps_2, delta_2)
    f = TradeOffFunction.weighted_infimal_convolution([alpha_1, alpha_2], [f1, f2])

    title = "mixture_test"

    plot_multiple_functions([f1, f2, f],
                            [
                                f"$({eps_1},{delta_1})$-DP",
                                f"$({eps_2},{delta_2})$-DP",
                                f"Mixture, weights ({alpha_1}, {alpha_2})"
                            ],
                            [
                                "dashed",
                                "dashed",
                                "solid"
                            ],
                            save_to=png(title)
                            )


def doubledp_and_multidp_coincide(eps_1, delta_1, eps_2, delta_2, k_ls, title):
    """
    Plot an instance of the main theorems for multiple values of k.
    """
    if delta_1 > delta_2:
        delta_1, delta_2 = delta_2, delta_1
        eps_1, eps_2 = eps_2, eps_1

    f_double_dp = MultiEpsDeltaTradeoff([eps_1, eps_2], [delta_1, delta_2])

    f_comp = []
    f_multi = []
    for k in k_ls:
        f_comp.append(privacy_region_composition_double_dp_heterogeneous_comp(eps_1, delta_1, eps_2, delta_2, k))
        f_multi.append(privacy_region_composition_multi_dp([eps_1, eps_2], [delta_1, delta_2], k))

    plot_multiple_functions(
        [f_double_dp] + f_comp + f_multi,
        [f"({eps_1},{delta_1}) and ({eps_2},{delta_2}) DP"] + [f"{k}-double" for k in k_ls] + [f"{k}-multi" for k in
                                                                                               k_ls],
        ["solid"] + ["solid"] * len(f_comp) + ["dashed"] * len(f_multi),
        save_to=png(title)
    )


def multidp_example_multi_vs_double(eps_ls, delta_ls, k, title):
    f_double = privacy_region_composition_double_dp_heterogeneous_comp(eps_ls[0], delta_ls[0], eps_ls[1], delta_ls[1],
                                                                       k)
    f_triple = privacy_region_composition_multi_dp(eps_ls, delta_ls, k)

    plot_multiple_functions(
        [f_double, f_triple],
        [f"{k}-double", f"{k}-multi"],
        ["solid", "dashed"],
        save_to=png(title)
    )


def laplace_multidp_comp_approx(eps, n, k, title):
    f_lap = LaplaceTradeoff(eps)

    f_below = l1_multi_dp_approx_below(f_lap, n)
    f_above = l1_multi_dp_approx_above(f_lap, n)

    f_lap_comp = LaplaceTradeoff(eps)
    f_below_comp = privacy_region_composition_multi_dp(f_below.get_eps_list(), f_below.get_delta_list(), k)
    f_above_comp = privacy_region_composition_multi_dp(f_above.get_eps_list(), f_above.get_delta_list(), k)

    plot_multiple_functions(
        [
            f_lap_comp,
            f_below_comp,
            f_above_comp
        ],
        [
            f"Laplace({eps})-DP",
            f"{n}-DP {k}-comp. approx below",
            f"{n}-DP {k}-comp. approx above"
        ],

        ["solid",
         "dashed",
         "dashed"
         ],
        save_to=png(title)
    )


def laplace_n_dp_approx(n, mu=1):
    """
    Compare n-DP approximations of Laplace-DP.
    """
    lap = LaplaceTradeoff(mu)
    below_approx = l1_multi_dp_approx_below(lap, n)
    above_approx = l1_multi_dp_approx_above(lap, n)

    eps_below = below_approx.get_eps_list()
    delta_below = below_approx.get_delta_list()

    eps_above = above_approx.get_eps_list()
    delta_above = above_approx.get_delta_list()

    f_below = [SingleEpsDeltaTradeoff(eps, delta) for eps, delta in zip(eps_below, delta_below)]
    f_above = [SingleEpsDeltaTradeoff(eps, delta) for eps, delta in zip(eps_above, delta_above)]

    f_arr = f_below + f_above + [lap]
    colors = [COLOR_3] * len(f_below) + [COLOR_2] * len(f_above) + [COLOR_1]
    linestyles = ["dashed"] * len(f_below) + ["dashed"] * len(f_above) + ["solid"]
    title = f"laplace_{n}_dp_approx"

    plot_multiple_functions(
        f_arr=f_arr,
        colors=colors,
        linestyles=linestyles,
        save_to=png(title)
    )

    plot_multiple_functions(
        f_arr=[lap, below_approx, above_approx],
        labels=[f"Laplace({mu})-DP", f"{n}-DP approx below", f"{n}-DP approx above"],
        colors=[COLOR_1, COLOR_3, COLOR_2],
        linestyles=["solid", "dashed", "dashed"],
        save_to=png(title + "_maxed")
    )


def subsampled_dp_test(eps=3, delta=0.1, p=0.2, title="subsampled_dp_test"):
    f = SingleEpsDeltaTradeoff(eps, delta)
    f_subsampled = f.subsampled(p)

    plot_multiple_functions(
        [f, f_subsampled],
        [f"({eps},{delta})-DP", f"({eps},{delta})-DP, subsampled"],
        ["solid", "dashed"],
        save_to=png(title)
    )


def subsampled_gaussian_test(mu=1.8, p=0.35, title="subsampled_gaussian_test"):
    gaussian = GaussianTradeoff(mu)
    gaussian_subsampled = TradeOffFunction.subsampled(gaussian, p)

    plot_multiple_functions(
        [gaussian, gaussian_subsampled],
        [f"Gaussian({mu})-DP", f"Gaussian({mu})-DP, subsampled"],
        ["solid", "dashed"],
        save_to=png(title)
    )


def subsampled_laplace_approx(n, mu=1, p=0.2, title="laplace_subsampled_n_dp_approx"):
    """
    Compare n-DP approximations of subsampled Laplace-DP.
    """
    lap = LaplaceTradeoff(mu)
    lap_subs = lap.subsampled(p)
    below_approx = l1_multi_dp_approx_below(lap_subs, n)
    above_approx = l1_multi_dp_approx_above(lap_subs, n)

    plot_multiple_functions(
        f_arr=[lap, lap_subs, below_approx, above_approx],
        labels=[f"Lap({mu})", f"{p}-subsampled Lap({mu})", f"{n}-DP approx below", f"{n}-DP approx above"],
        colors=[COLOR_4, COLOR_1, COLOR_3, COLOR_2],
        linestyles=["dotted", "solid", "dashed", "dashed"],
        save_to=png(title)
    )


def subsampled_laplace_comp_approx(n, k, mu=1, p=0.2, title="laplace_subsampled_comp_n_dp_approx"):
    """
    Compare n-DP approximations of composed subsampled Laplace mechanisms.
    """
    lap = LaplaceTradeoff(mu)
    lap_subs = lap.subsampled(p)
    f_below = l1_multi_dp_approx_below(lap_subs, n)
    f_above = l1_multi_dp_approx_above(lap_subs, n)

    f_below_comp = privacy_region_composition_multi_dp(f_below.get_eps_list(), f_below.get_delta_list(), k)
    f_above_comp = privacy_region_composition_multi_dp(f_above.get_eps_list(), f_above.get_delta_list(), k)

    plot_multiple_functions(
        f_arr=[lap_subs, f_below, f_above, f_below_comp, f_above_comp],
        labels=[f"Subsampled Lap({mu})", f"{n}-DP approx below", f"{n}-DP approx above", f"{k}-comp. approx below",
                f"{k}-comp. approx above"],
        colors=[COLOR_1, COLOR_3, COLOR_2, COLOR_3, COLOR_2],
        linestyles=["solid", "dashed", "dashed", "dotted", "dotted"],
        save_to=png(title)
    )

def subsampled_vmf(p, dimensions=3., kappa=2., max_angle=np.cos(np.pi / 4), title="vmf_subsampled"):
    """
    Compare VMF and subsampled VMF.
    """
    vmf = VonMisesFisherTradeoff(dimensions, kappa, max_angle)
    vmf_subs = vmf.subsampled(p)

    plot_multiple_functions(
        f_arr=[vmf, vmf_subs],
        labels=[f"VMF", f"{p}-subsampled VMF"],
        colors=[COLOR_1, COLOR_2],
        linestyles=["solid", "solid"],
        save_to=png(title)
    )


def subsampled_vmf_n_dp_approx(n, p, dimensions=3., kappa=2., max_angle=np.cos(np.pi / 4), title="vmf_n_dp_approx"):
    """
    Compare n-DP approximations of subsampled VMF.
    """
    vmf = VonMisesFisherTradeoff(dimensions, kappa, max_angle)
    vmf_subs = vmf #.subsampled(p)
    below_approx = l1_multi_dp_approx_below(vmf_subs, n)
    above_approx = l1_multi_dp_approx_above(vmf_subs, n)

    plot_multiple_functions(
        f_arr=[vmf, vmf_subs, below_approx, above_approx],
        labels=[f"VMF", f"{p}-subsampled VMF", f"{n}-DP approx below", f"{n}-DP approx above"],
        colors=[COLOR_4, COLOR_1, COLOR_3, COLOR_2],
        linestyles=["dotted", "solid", "dashed", "dashed"],
        save_to=png(title)
    )

def subsampled_gaussian_n_dp_comp_test(n, k, mu=1., p=0.2, title="subsampled_gaussian_comp_test"):
    gaussian = GaussianTradeoff(mu)
    gaussian_subsampled = TradeOffFunction.subsampled(gaussian, p)

    f_below = l1_multi_dp_approx_below(gaussian_subsampled, n)
    f_above = l1_multi_dp_approx_above(gaussian_subsampled, n)

    f_below_comp = privacy_region_composition_multi_dp(f_below.get_eps_list(), f_below.get_delta_list(), k)
    f_above_comp = privacy_region_composition_multi_dp(f_above.get_eps_list(), f_above.get_delta_list(), k)

    plot_multiple_functions(
        f_arr=[gaussian_subsampled, f_below, f_above, f_below_comp, f_above_comp],
        labels=[f"Subsampled G({mu})", f"{n}-DP approx below", f"{n}-DP approx above", f"{k}-comp. approx below",
                f"{k}-comp. approx above"],
        colors=[COLOR_1, COLOR_3, COLOR_2, COLOR_3, COLOR_2],
        linestyles=["solid", "dashed", "dashed", "dotted", "dotted"],
        save_to=png(title)
    )

def laplace_tradeoff_approx_multip_norms(n, eps=1., title="laplace_tradeoff_approx_multip_norms"):
    """
    Plot the n-DP lower and upper L1/Linf approximations of the Laplace trade-off function.
    """
    laplace_eps = LaplaceTradeoff(eps)
    l1_eps_approx_below = l1_multi_dp_approx_below(laplace_eps, n)
    l1_eps_approx_above = l1_multi_dp_approx_above(laplace_eps, n)
    linf_eps_approx_below = linf_multi_dp_approx_below(laplace_eps, n)
    linf_eps_approx_above = linf_multi_dp_approx_above(laplace_eps, n)


    plot_multiple_functions(
        [
            laplace_eps,
            l1_eps_approx_below,
            l1_eps_approx_above,
            linf_eps_approx_below,
            linf_eps_approx_above
        ],
        [
            f"Laplace({eps})-DP",
            "L1 approx below",
            "L1 approx above",
            "Linf approx below",
            "Linf approx above"
        ],
        [
            "dotted",
            "solid",
            "solid",
            "dashed",
            "dashed",
        ],
        [
            COLOR_1,
            COLOR_2,
            COLOR_2,
            COLOR_3,
            COLOR_3
        ],
        save_to=png(title)
    )

def gaussian_tradeoff_approx_multip_norms(n, k, mu=1., title="gaussian_tradeoff_approx_multip_norms"):
    """
    Plot the n-DP lower and upper L1/Linf approximations of the Gaussian trade-off comparison.
    """
    gaussian = GaussianTradeoff(mu)

    l1_eps_approx_below = l1_multi_dp_approx_below(gaussian, n)
    l1_eps_approx_above = l1_multi_dp_approx_above(gaussian, n)

    linf_eps_approx_below = linf_multi_dp_approx_below(gaussian, n)
    linf_eps_approx_above = linf_multi_dp_approx_above(gaussian, n)

    l1_below_comp = privacy_region_composition_multi_dp(l1_eps_approx_below.get_eps_list(), l1_eps_approx_below.get_delta_list(), k)
    l1_above_comp = privacy_region_composition_multi_dp(l1_eps_approx_above.get_eps_list(), l1_eps_approx_above.get_delta_list(), k)

    linf_below_comp = privacy_region_composition_multi_dp(linf_eps_approx_below.get_eps_list(), linf_eps_approx_below.get_delta_list(), k)
    linf_above_comp = privacy_region_composition_multi_dp(linf_eps_approx_above.get_eps_list(), linf_eps_approx_above.get_delta_list(), k)


    plot_multiple_functions(
        [
            gaussian,
            l1_eps_approx_below,
            l1_eps_approx_above,
            linf_eps_approx_below,
            linf_eps_approx_above,
            l1_below_comp,
            l1_above_comp,
            linf_below_comp,
            linf_above_comp,
        ],
        [
            f"Gaussian({mu})-DP",
            "L1 approx below",
            "L1 approx above",
            "Linf approx below",
            "Linf approx above",
            "L1 comp. approx below",
            "L1 comp. approx above",
            "Linf comp. approx below",
            "Linf comp. approx above",
        ],
        [
            "dotted",
            "solid",
            "solid",
            "dashed",
            "dashed",
            "solid",
            "solid",
            "dashed",
            "dashed",
        ],
        [
            COLOR_1,
            COLOR_2,
            COLOR_2,
            COLOR_3,
            COLOR_3,
            COLOR_4,
            COLOR_4,
            COLOR_5,
            COLOR_5
        ],
        save_to=png(title)
    )


def subsampled_gaussian_tradeoff_approx_multip_norms(n, k, mu=1., p=0.2, title="subs_gaussian_tradeoff_approx_multip_norms"):
    """
    Plot the n-DP lower and upper L1/Linf approximations of the Gaussian trade-off comparison.
    """
    gaussian = GaussianTradeoff(mu).subsampled(p)

    l1_eps_approx_below = l1_multi_dp_approx_below(gaussian, n)
    l1_eps_approx_above = l1_multi_dp_approx_above(gaussian, n)

    linf_eps_approx_below = linf_multi_dp_approx_below(gaussian, n)
    linf_eps_approx_above = linf_multi_dp_approx_above(gaussian, n)

    l1_below_comp = privacy_region_composition_multi_dp(l1_eps_approx_below.get_eps_list(), l1_eps_approx_below.get_delta_list(), k)
    l1_above_comp = privacy_region_composition_multi_dp(l1_eps_approx_above.get_eps_list(), l1_eps_approx_above.get_delta_list(), k)

    linf_below_comp = privacy_region_composition_multi_dp(linf_eps_approx_below.get_eps_list(), linf_eps_approx_below.get_delta_list(), k)
    linf_above_comp = privacy_region_composition_multi_dp(linf_eps_approx_above.get_eps_list(), linf_eps_approx_above.get_delta_list(), k)


    plot_multiple_functions(
        [
            gaussian,
            l1_eps_approx_below,
            l1_eps_approx_above,
            linf_eps_approx_below,
            linf_eps_approx_above,
            l1_below_comp,
            l1_above_comp,
            linf_below_comp,
            linf_above_comp,
        ],
        [
            f"{p}-subsampled G({mu})-DP",
            "L1 approx below",
            "L1 approx above",
            "Linf approx below",
            "Linf approx above",
            "L1 comp. approx below",
            "L1 comp. approx above",
            "Linf comp. approx below",
            "Linf comp. approx above",
        ],
        [
            "dotted",
            "solid",
            "solid",
            "dashed",
            "dashed",
            "solid",
            "solid",
            "dotted",
            "dotted",
        ],
        [
            COLOR_1,
            COLOR_2,
            COLOR_2,
            COLOR_3,
            COLOR_3,
            COLOR_4,
            COLOR_4,
            COLOR_5,
            COLOR_5
        ],
        save_to=png(title)
    )


if __name__ == "__main__":
    # heterogeneous_comparison(eps_1=0.6,eps_2=0.4,x=3,y=2,delta_slack_ls=[0.001], title="heterogeneous_comparison")
    # mixture_example(alpha_1 = 0.5, eps_1 = 1.3, delta_1 = 0.0, eps_2 = 0.5, delta_2 = 0.2, title="mixture_example")
    # gaussian_tradeoff_approx(mu=1, title="gaussian_approx")
    # gaussian_compos_approx(k=20, mu=0.05, title="gaussian_compos_approx")
    # gaussian_compos_approx(k=3, mu=1, title="gaussian_compos_approx_2")
    # gaussian_compos_approx_two_compos(k1=10, k2=3, mu=1, title="gaussian_2_compos")
    # gaussian_tradeoff_and_compos_approx(k=3, mu=1, title="gaussian_tradeoff_and_compos_approx")
    # main_theorem_comparison(eps_1 = 1.2, delta_1 = 0.0, eps_2 = 0.6, delta_2 = 0.2, k = 3, title="theorem_1_comparison")
    # main_theorem_example(eps_1 = 1.2, delta_1 = 0.0, eps_2 = 0.6, delta_2 = 0.2, k_ls = [2, 3, 10, 20],
    #                      title="theorem_1_example")
    # main_theorem_example(eps_1 = 0.3, delta_1 = 0.0, eps_2 = 0.15, delta_2 = 0.02, k_ls = [2, 3, 10, 20],
    #                      title="theorem_1_example_small_region")
    # main_theorem_comparison_two_ks(eps_1 = 1.2, delta_1 = 0.0, eps_2 = 0.6, delta_2 = 0.2,
    #                                k1=3, k2=10, title="theorem_1_comparison_two_ks")
    # main_theorem_comparison_two_ks(eps_1 = 0.3, delta_1 = 0.0, eps_2 = 0.15, delta_2 = 0.02, k1=3, k2=20,
    #                               title="theorem_1_comparison_two_ks_small_region")
    # gaussian_compos_approx_tradeoff_and_two_compos(k1=3, k2=10, mu=1, title="gaussian_tradeoff_and_2_compos")
    # gaussian_compos_approx_tradeoff_and_two_compos(k1=3, k2=10, mu=0.05, title="gaussian_tradeoff_and_2_compos_small")
    # gaussian_tradeoff_and_compos_approx(mu=0.05, k=3, title="gaussian_tradeoff_and_compos_approx_small_single_reg")
    # smooth_vs_nonsmooth_above_2dp_approx_gaussian(mu=1, title="smooth_vs_nonsmooth_2dp_approx_gaussian_above")
    # smooth_vs_nonsmooth_below_2dp_approx_gaussian(mu=1., title="smooth_vs_nonsmooth_2dp_approx_gaussian_below")
    # mixture_test()
    # doubledp_and_multidp_coincide(eps_1=0.3, delta_1=0.0, eps_2=0.15, delta_2=0.02, k_ls=[5],
    #                               title="double_and_multi_comparison")
    # multidp_example_multi_vs_double([0.3, 0.15, 0.04, 0.01], [0.0, 0.02, 0.05, 0.06], 5, "triple_vs_double")
    # laplace_multidp_comp_approx(1, 3, 4, "laplace_3dp_comp_approx")
    # laplace_multidp_comp_approx(1, 2, 4, "laplace_2dp_comp_approx")
    # laplace_n_dp_approx(4)
    # subsampled_dp_test(eps = 2, delta = 0.1, p = 0.2, title = "subsampled_dp_test")
    # subsampled_gaussian_test(mu = 1.8, p = 0.35, title = "subsampled_gaussian_test")
    # subsampled_laplace_approx(3)
    # subsampled_laplace_comp_approx(3, 9)
    # subsampled_vmf(0.2, dimensions=3., kappa=2., max_angle=np.cos(np.pi / 4))
    # subsampled_gaussian_n_dp_comp_test(4, 9, mu=1., p=0.2, title="subsampled_gaussian_n_dp_comp_test")
    laplace_tradeoff_approx_multip_norms(2)
    gaussian_tradeoff_approx_multip_norms(3, 6, mu=0.5)
    subsampled_gaussian_tradeoff_approx_multip_norms(3, 9)