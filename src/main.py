from f_dp_approximation.gaussian_tradeoff import GaussianTradeoff
from f_dp_approximation.laplace_tradeoff import LaplaceTradeoff
from multi_dp_mixture.dp_functions import MultiEpsDeltaTradeoff, SingleEpsDeltaTradeoff
from multi_dp_mixture.piecewise_affine import PiecewiseAffine, DIAGONAL
from imports import *
from main_theorems.combinatorial_version import privacy_region_composition_double_dp_combinatorial
from main_theorems.heterogeneous_version import privacy_region_composition_double_dp_heterogeneous_comp, \
    privacy_region_composition_heterogeneous
from main_theorems.other_composition_theorems import *
from utils import plot_multiple_functions, plot_one_function


def multi_dp_test():
    eps_s = [0.8, 0.25]
    delta_s = [0.1, 0.2]
    f = MultiEpsDeltaTradeoff(eps_s, delta_s)
    f.to_plot()

def add_test():
    f = 0.5 * (SingleEpsDeltaTradeoff(0.6, 0.5) + SingleEpsDeltaTradeoff(0.5, 0.2))
    f.to_plot()

def convex_conj_test():
    eps_s = [0.8, 0.25]
    delta_s = [0.1, 0.2]
    f = MultiEpsDeltaTradeoff(eps_s, delta_s)
    f.to_plot()
    f.convex_conjugate().to_plot()

def double_convex_conj_is_identity_test():
    eps_s = [0.8, 0.25, 0]
    delta_s = [0.1, 0.2, 0.65]
    f = MultiEpsDeltaTradeoff(eps_s, delta_s)
    f.to_plot()
    f_conj = f.convex_conjugate()
    f_double_conj = f_conj.convex_conjugate()
    f_double_conj.to_plot()

def first_addition_test():
    alpha_1 = 0.5
    alpha_2 = 1 - alpha_1
    eps_1 = 1.3
    delta_1 = 0.0
    eps_2 = 0.5
    delta_2 = 0.2
    f1 = SingleEpsDeltaTradeoff(eps_1, delta_1)
    f2 = SingleEpsDeltaTradeoff(eps_2, delta_2)
    f = PiecewiseAffine.weighted_infimal_convolution([alpha_1, alpha_2], [f1, f2])

    plot_multiple_functions([f1, f2, f],
                                            [f"$({eps_1},{delta_1})$-DP",
                                             f"$({eps_2},{delta_2})$-DP",
                                             f"Mixture, weights ({alpha_1}, {alpha_2})"
                                             ])

def two_theorems_match():
    eps_1 = 1.2
    delta_1 = 0.0
    eps_2 = 0.6
    delta_2 = 0.2
    k = 3

    f1 = privacy_region_composition_double_dp_heterogeneous_comp(eps_1, delta_1, eps_2, delta_2, k)
    f2 = privacy_region_composition_double_dp_combinatorial(eps_1, delta_1, eps_2, delta_2, k)
    fo = MultiEpsDeltaTradeoff([eps_1, eps_2], [delta_1, delta_2])
    f_dp_1 = privacy_region_composition_exact(eps_1, delta_1, k)
    f_dp_2 = privacy_region_composition_exact(eps_2, delta_2, k)

    plot_multiple_functions(
        [
         #fo,
         f1,
         f2,
         #f_dp_1,
         #f_dp_2
         ],
        [
         #f"$({eps_1},{delta_1})$ and $({eps_2},{delta_2})$ DP",
         f"Heterogeneous ${k}$ comp.",
         f"Combinatorial ${k}$ comp.",
         #f"$({eps_1},{delta_1})$-DP {k} comp.",
         #f"$({eps_2},{delta_2})$-DP {k} comp."
         ]
    )

def privacy_region_heter_sanity_check():
    eps_1 = 1.3
    delta_1 = 0.0
    eps_2 = 0.5
    delta_2 = 0.2
    x = 3
    y = 0

    f_comp = privacy_region_composition_heterogeneous(eps_1, eps_2, x, y)
    f_original = privacy_region_composition_exact(eps_1, 0, x)

    plot_multiple_functions([f_comp, f_original],["Heterogeneous new", "Original"])

def gaussian_tradeoff_approx():
    # eps = 1
    # delta = 0.02
    # mu = GaussianTradeoff.compute_mu_from_eps_delta(eps, delta)
    mu = 1
    g_mu = GaussianTradeoff(mu)
    g_mu_approx_below = g_mu.approx_from_below()
    g_mu_approx_above = g_mu.approx_from_above()
    #default_eps_delta = SingleEpsDeltaTradeoff(eps, delta)
    plot_multiple_functions(
        [
            g_mu,
            g_mu_approx_below,
            g_mu_approx_above,
            #default_eps_delta
        ],
        [
            f"${float(mu):.2}-GDP$",
            f"Approx below",
            "Approx above",
            #"Default eps delta guarantee"
        ]
    )

def gaussian_compos_approx():
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
            f"${float(mu):.2}-GDP$ comp. {k} times $= {float(mu_composed):.2}-GDP$",
            "Comp. approx from below",
            "Comp. approx from above"
        ]
    )

def laplace_tradeoff_approx():
    eps = 1
    laplace_eps = LaplaceTradeoff(eps)
    lap_eps_approx_below = laplace_eps.approx_from_below()
    laps_eps_approx_above = laplace_eps.approx_from_above()
    # default_eps_delta = SingleEpsDeltaTradeoff(eps, delta)
    plot_multiple_functions(
        [
            laplace_eps,
            lap_eps_approx_below,
            laps_eps_approx_above
            # default_eps_delta
        ],
        [
            f"$Laplace({eps})-DP$",
            "Approx below",
            "Approx above",
            # "Default eps delta guarantee"
        ]
    )


two_theorems_match()
# laplace_tradeoff_approx()
# gaussian_tradeoff_approx()
# gaussian_compos_approx()
