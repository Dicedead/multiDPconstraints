from definitions import *
from multi_dp_mixture.dp_functions import MultiEpsDeltaTradeoff, SingleEpsDeltaTradeoff
from multi_dp_mixture.piecewise_affine import PiecewiseAffine


def privacy_region_composition_double_dp_combinatorial(eps_1, delta_1, eps_2, delta_2, k) -> PiecewiseAffine:
    """
    Computes the privacy region composition of k doubly (eps,delta)-DP constrained
    mechanisms combinatorially.

    :param eps_1: First epsilon parameter value.
    :type eps_1: float, > 0
    :param delta_1: First delta parameter value.
    :type delta_1: float in [0,1]
    :param eps_2: Second epsilon parameter value.
    :type eps_2: float, > 0
    :param delta_2: Second delta parameter value.
    :type delta_2: float in [0,1]
    :param k: Total number of mechanisms to compose.
    :type k: int, >= 1
    :return: A compositional trade-off function derived from combinations of mechanisms.
    :rtype: PiecewiseAffine
    """

    def compute_b_set(u, v):
        b_set = []
        for a in range(0, k+1):
            for b in range(0, k+1):
                for c in range(0, k+1):
                    d = k - a - b - c
                    if d >= 0 and (a + k - d - u - v) * eps_1 + (b + v - c - u) * eps_2 > 0:
                        b_set.append((a, b, c, d))
        return b_set

    def multinomial(*ks):
        n = sum(ks)
        result = 1
        remaining = n
        for k in ks:
            result *= math.comb(remaining, k)
            remaining -= k
        return result

    assert eps_1 > 0
    assert eps_2 > 0
    assert 0 <= delta_1 <= 1
    assert 0 <= delta_2 <= 1
    assert k >= 1

    if delta_1 > delta_2:
        delta_1, delta_2 = delta_2, delta_1
        eps_1, eps_2 = eps_2, eps_1

    exp_eps_1 = np.exp(eps_1)
    exp_eps_2 = np.exp(eps_2)

    assert (1-delta_1) * (1+exp_eps_2) < (1-delta_2) * (1+exp_eps_1)

    ratio = eps_1 / eps_2

    for p in range(1, k+1):
        for q in range(1, k+1):
            assert ratio != p/q

    eps_ls = []
    delta_ls = []

    alpha_num = (1-delta_1) * exp_eps_2 - (1-delta_2) * exp_eps_1 + (delta_2 - delta_1)
    alpha_denom = (exp_eps_2 - exp_eps_1) * (1-delta_1)
    alpha = alpha_num / alpha_denom

    for v in range(0, k+1):
        lower_u_bound = int(np.ceil((k * eps_1 - v * (eps_1 - eps_2))/(eps_1 + eps_2)))
        for u in range(lower_u_bound, k+1):
            eps_u_v = eps_1 * (u + v - k) + eps_2 * (u - v)
            b_set_u_v = compute_b_set(u, v)
            delta_u_v = 0.

            for a, b, c, d in b_set_u_v:
                multi = multinomial(a, b, c, d)
                first_factor = ((1-alpha) / (exp_eps_1 + 1)) ** (a + d)
                second_factor = (alpha/(exp_eps_2 + 1)) ** (b + c)
                third_factor_term_1 = np.exp(a * eps_1 + b * eps_2)
                third_factor_term_2 = np.exp((d+u+v-k) * eps_1 + (c+u-v) * eps_2)
                third_factor = third_factor_term_1 - third_factor_term_2
                delta_u_v += multi * first_factor * second_factor * third_factor

            eps_ls.append(eps_u_v)
            delta_ls.append(delta_u_v)

    f_intersect = MultiEpsDeltaTradeoff(eps_ls, delta_ls)
    f_01 = SingleEpsDeltaTradeoff(0, 1)
    weight_f_intersect = (1-delta_1) ** k

    return PiecewiseAffine.weighted_infimal_convolution(
        [weight_f_intersect, 1-weight_f_intersect], [f_intersect, f_01]
    )
