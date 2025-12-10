from multi_dp_mixture.dp_functions import MultiEpsDeltaTradeoff

eps_s = [0.8, 0.25]
delta_s = [0.1, 0.2]
f = MultiEpsDeltaTradeoff(eps_s, delta_s)
f.to_plot()