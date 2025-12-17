from multi_dp_mixture.dp_functions import MultiEpsDeltaTradeoff, SingleEpsDeltaTradeoff
from multi_dp_mixture.piecewise_affine import PiecewiseAffine
from imports import *
from main_theorems.combinatorial_version import privacy_region_composition_double_dp_combinatorial
from main_theorems.heterogeneous_version import privacy_region_composition_double_dp_heterogeneous_comp

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
    eps_1 = 1.3
    delta_1 = 0.1
    eps_2 = 0.5
    delta_2 = 0.2
    k = 1

    f1 = privacy_region_composition_double_dp_heterogeneous_comp(eps_1, delta_1, eps_2, delta_2, k)
    f2 = privacy_region_composition_double_dp_combinatorial(eps_1, delta_1, eps_2, delta_2, k)
    fo = MultiEpsDeltaTradeoff([eps_1, eps_2], [delta_1, delta_2])

    PiecewiseAffine.plot_multiple_functions([fo, f1, f2], ["Original double DP guarantee", "Heterogeneous", "Combinatorial"])

two_theorems_match()
