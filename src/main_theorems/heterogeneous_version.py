from definitions import *
from multi_dp_mixture.dp_functions import MultiEpsDeltaTradeoff, SingleEpsDeltaTradeoff
from multi_dp_mixture.piecewise_affine import PiecewiseAffine


def privacy_region_composition_heterogeneous(eps_1, eps_2, x, y) -> PiecewiseAffine:
    """
    Computes privacy region corresponding to the composition of x (eps_1,0)-DP mechanisms
    with y (eps_2, 0)-DP mechanisms.

    :param eps_1: Privacy parameter epsilon for the first kind of mechanism.
    :type eps_1: float. > 0
    :param eps_2: Privacy parameter epsilon for the second kind of mechanism.
    :type eps_2: float, > 0
    :param x: Number of first kind of mechanisms.
    :type x: int
    :param y: Number of second kind of mechanisms.
    :type y: int
    :return: A compositional trade-off function derived from combinations of mechanisms.
    :rtype: PiecewiseAffine
    """
    def compute_epsilon_from_ab(a, b):
        return eps_1 * (2 * a - x) + eps_2 * (2 * b - y)

    def compute_delta_from_ab(a_star, b_star):
        delta = 0

        b_0 = int(np.ceil((a_star-x)*(eps_1/eps_2) + b_star))
        for b in range(b_0, y+1):
            a_0 = int(np.ceil((b_star-b)*(eps_2/eps_1) + a_star))
            for a in range(a_0, x+1):
                first_term = np.exp(a * eps_1 + b * eps_2)
                second_term = np.exp(2 * (a_star * eps_1 + b_star * eps_2) - (a * eps_1 + b * eps_2))
                delta += first_term - second_term

        first_factor = (1/(np.exp(eps_1)+1)) ** x
        second_factor = (1/(np.exp(eps_2)+1)) ** y
        delta = delta * first_factor * second_factor
        return delta

    def compute_a_set():
        a_set = []
        for a in range(0,x+1):
            for b in range(0,y+1):
                if compute_epsilon_from_ab(a, b) >= 0:
                    a_set.append((a,b))
        return a_set

    assert eps_1 > 0
    assert eps_2 > 0
    assert x >= 0
    assert y >= 0

    if eps_2 > eps_1:
        eps_1, eps_2 = eps_2, eps_1

    a_set = compute_a_set()
    eps_ls = [compute_epsilon_from_ab(a, b) for a, b in a_set]
    delta_ls = [compute_delta_from_ab(a, b) for a, b in a_set]

    return MultiEpsDeltaTradeoff(eps_ls, delta_ls)

def privacy_region_composition_double_dp_heterogeneous_comp(eps_1, delta_1, eps_2, delta_2, k) -> PiecewiseAffine:
    """
    Computes the privacy region composition of k doubly (eps,delta)-DP constrained
    mechanisms by decomposing the composition into a sum of compositions of
    heterogeneous mechanisms.

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

    assert eps_1 > 0
    assert eps_2 > 0
    assert 0 <= delta_1 <= 1
    assert 0 <= delta_2 <= 1
    assert k >= 1

    exp_eps_1 = np.exp(eps_1)
    exp_eps_2 = np.exp(eps_2)

    if delta_1 > delta_2:
        delta_1, delta_2 = delta_2, delta_1
        eps_1, eps_2 = eps_2, eps_1

    assert (1-delta_1) * (1+exp_eps_2) < (1-delta_2) * (1+exp_eps_1)

    alpha_num = (1-delta_1) * exp_eps_2 - (1-delta_2) * exp_eps_1 + (delta_2 - delta_1)
    alpha_denom = (exp_eps_2 - exp_eps_1) * (1-delta_1)
    alpha = alpha_num / alpha_denom

    heterogeneous_weight = (1-delta_1) ** k
    weights = [1-heterogeneous_weight]
    functions = [SingleEpsDeltaTradeoff(0, 1)]

    for i in range(k+1):
        weight = heterogeneous_weight * sp.comb(k, i) * (alpha ** i) * ((1-alpha) ** (k-i))
        weights.append(weight)
        functions.append(privacy_region_composition_heterogeneous(eps_1, eps_2, i, k))

    return PiecewiseAffine.weighted_infimal_convolution(weights, functions)