from multi_dp_mixture.dp_functions import MultiEpsDeltaTradeoff, SingleEpsDeltaTradeoff
from multi_dp_mixture.piecewise_affine import PiecewiseAffine
from imports import *
from main_theorems.combinatorial_version import privacy_region_composition_double_dp_combinatorial
from main_theorems.heterogeneous_version import privacy_region_composition_double_dp_heterogeneous_comp, \
    privacy_region_composition_heterogeneous, privacy_region_composition_heterogeneous_fails_sanity_check
from main_theorems.other_composition_theorems import *

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
    f1 = SingleEpsDeltaTradeoff(1.3, 0.1)
    f2 = SingleEpsDeltaTradeoff(0.5, 0.2)
    f = PiecewiseAffine.weighted_infimal_convolution([alpha_1, alpha_2], [f1, f2])

    PiecewiseAffine.plot_multiple_functions([f1, f2, f], ["$(1.3,0.1)$-DP", "$(0.5,0.2)$-DP", "Mixture"])

def two_theorems_match():
    eps_1 = 1.31
    delta_1 = 0.1
    eps_2 = 0.5
    delta_2 = 0.2
    k = 3

    f1 = privacy_region_composition_double_dp_heterogeneous_comp(eps_1, delta_1, eps_2, delta_2, k)
    f2 = privacy_region_composition_double_dp_combinatorial(eps_1, delta_1, eps_2, delta_2, k)
    fo = MultiEpsDeltaTradeoff([eps_1, eps_2], [delta_1, delta_2])
    f_dp_1 = privacy_region_composition_exact(eps_1, delta_1, k)
    f_dp_2 = privacy_region_composition_exact(eps_2, delta_2, k)

    PiecewiseAffine.plot_multiple_functions(
        [
         #fo,
         f1,
         f2,
         f_dp_1,
         f_dp_2
         ],
        [
        #f"$({eps_1},{delta_1})$ and $({eps_2},{delta_2})$ DP",
         f"Heterogeneous ${k}$ comp.",
         f"Combinatorial ${k}$ comp.",
         f"$({eps_1},{delta_1})$-DP {k} comp.",
         f"$({eps_2},{delta_2})$-DP {k} comp."
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
    f_comp_2 = privacy_region_composition_heterogeneous_fails_sanity_check(eps_1, eps_2, x, y)
    f_original = privacy_region_composition_exact(eps_1, 0, x)

    PiecewiseAffine.plot_multiple_functions([f_comp, f_comp_2, f_original],
                                            ["Heterogeneous new", "Heterogeneous fails sanity check", "Original"])
    PiecewiseAffine.plot_multiple_functions([f_comp, f_original],["Heterogeneous new", "Original"])

two_theorems_match()
# privacy_region_heter_sanity_check()
