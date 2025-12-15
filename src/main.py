from multi_dp_mixture.dp_functions import MultiEpsDeltaTradeoff, SingleEpsDeltaTradeoff
from multi_dp_mixture.piecewise_affine import PiecewiseAffine

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
    f_conj.to_plot()
    f_double_conj = f.convex_conjugate().convex_conjugate()
    f_double_conj.to_plot()

def first_addition_test():
    alpha_1 = 0.3
    alpha_2 = 1 - alpha_1
    f1 = SingleEpsDeltaTradeoff(0.5, 0.0)
    f2 = SingleEpsDeltaTradeoff(0.5, 0.4)
    f = PiecewiseAffine.weighted_infimal_convolution([alpha_1, alpha_2], [f1, f2])
    f1.to_plot()
    f2.to_plot()
    f.to_plot()

first_addition_test()