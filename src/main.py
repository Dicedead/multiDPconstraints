from multi_dp_mixture.dp_functions import MultiEpsDeltaTradeoff, SingleEpsDeltaTradeoff

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
    f = SingleEpsDeltaTradeoff(0.6, 0.5)
    f.to_plot()
    f_double_conj = f.convex_conjugate().convex_conjugate()
    f_double_conj.to_plot()

double_convex_conj_is_identity_test()