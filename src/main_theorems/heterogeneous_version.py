import numpy as np

from base.definitions import *
from base.tradeoff_function import TradeOffFunction
from main_theorems.heterogeneous_composition_paper import heter_comp_generalized
from multi_dp_mixture.dp_functions import MultiEpsDeltaTradeoff, SingleEpsDeltaTradeoff, \
    get_all_slopes_intercepts_from_eps_delta_ls, slope_to_eps, intercept_to_delta
from multi_dp_mixture.piecewise_affine import PiecewiseAffine, keep_useful_lines


def privacy_region_composition_heterogeneous_two_constraints(
        eps_1,
        eps_2,
        x,
        y,
        return_eps_deltas: bool = False
) -> MultiEpsDeltaTradeoff:
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
    :param return_eps_deltas: If True, returns the list of epsilon and delta values.
    :type return_eps_deltas: bool
    :return: A compositional trade-off function derived from combinations of mechanisms.
    :rtype: PiecewiseAffine
    """
    def compute_epsilon_from_ab(a, b):
        return eps_1 * (x - 2 * a) + eps_2 * (y - 2 * b)

    def compute_delta_from_ab(a_star, b_star):
        delta = 0
        slope = -np.exp(compute_epsilon_from_ab(a_star, b_star))

        for b in range(0, y+1):
            lower_a = max(int(np.ceil((y-b_star-b)*(eps_2/eps_1) + (x-a_star))), 0)
            for a in range(lower_a, x+1):
                first_term = np.exp(a * eps_1 + b * eps_2)
                second_term = slope * np.exp((x - a) * eps_1 + (y - b) * eps_2)

                delta += sps.comb(x, a, exact=True) * sps.comb(y, b, exact=True) * (first_term + second_term)

        first_factor = (1/(np.exp(eps_1)+1)) ** x
        second_factor = (1/(np.exp(eps_2)+1)) ** y
        delta = delta * first_factor * second_factor
        return delta

    def compute_ab_star():
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
        x, y = y, x

    a_set = compute_ab_star()
    eps_ls = [compute_epsilon_from_ab(a_star, b_star) for a_star, b_star in a_set]
    delta_ls = [compute_delta_from_ab(a_star, b_star) for a_star, b_star in a_set]

    f = MultiEpsDeltaTradeoff(eps_ls, delta_ls)

    if return_eps_deltas:
        eps_delta_ls = list(zip(eps_ls, delta_ls))
        return f, eps_delta_ls

    return f


def privacy_region_composition_double_dp_heterogeneous_comp(eps_1, delta_1, eps_2, delta_2, k) -> TradeOffFunction:
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

    if delta_1 > delta_2:
        delta_1, delta_2 = delta_2, delta_1
        eps_1, eps_2 = eps_2, eps_1

    exp_eps_1 = np.exp(eps_1)
    exp_eps_2 = np.exp(eps_2)

    assert (1-delta_1) * (1+exp_eps_2) < (1-delta_2) * (1+exp_eps_1)

    alpha_num = (1-delta_1) * exp_eps_2 - (1-delta_2) * exp_eps_1 + (delta_2 - delta_1)
    alpha_denom = (exp_eps_2 - exp_eps_1) * (1-delta_1)
    alpha = alpha_num / alpha_denom

    heterogeneous_weight = (1-delta_1) ** k
    weights = [1-heterogeneous_weight]
    functions: List[MultiEpsDeltaTradeoff] = [SingleEpsDeltaTradeoff(0, 1)]

    for i in range(0, k+1):
        weight = heterogeneous_weight * sps.comb(k, i) * ((1 - alpha) ** i) * (alpha ** (k-i))
        weights.append(weight)
        functions.append(privacy_region_composition_heterogeneous_two_constraints(eps_1, eps_2, i, k - i))

    return TradeOffFunction.weighted_infimal_convolution(weights, functions)


def __weak_compositions(n, k):
    """
    Yields all vectors of length n with non-negative integer components summing to k.
    """
    # Edge cases
    if n == 0:
        if k == 0:
            yield ()
        return

    for c in itertools.combinations(range(k + n - 1), n - 1):
        extended_c = (-1,) + c + (k + n - 1,)
        yield tuple(extended_c[i + 1] - extended_c[i] - 1 for i in range(n))

def __multinomial(lst):
    lst = list(lst)
    res, i = 1, sum(lst)
    i0 = lst.index(max(lst))
    for a in lst[:i0] + lst[i0+1:]:
        for j in range(1,a+1):
            res *= i
            res //= j
            i -= 1
    return res

def privacy_region_composition_multi_dp(eps_ls: List[float], delta_ls: List[float], k: int) -> TradeOffFunction:
    """
    Computes the privacy region composition of k multi-DP constrained
    mechanisms by decomposing the composition into a sum of compositions of
    heterogeneous mechanisms.

    :param eps_ls: List of epsilon parameters.
    :type eps_ls: List[float]
    :param delta_ls: List of delta parameters.
    :type delta_ls: List[float], same size as eps_ls
    :param k: Total number of mechanisms to compose.
    :type k: int, >= 1
    :return: A compositional trade-off function derived from combinations of mechanisms.
    :rtype: PiecewiseAffine
    """
    assert len(eps_ls) == len(delta_ls)

    # Keep only the active constraints and sort them by decreasing epsilon and increasing delta
    slopes, intercepts = get_all_slopes_intercepts_from_eps_delta_ls(
        np.array(eps_ls), np.array(delta_ls), with_inverses=False
    )
    useful_slopes, useful_intercepts = keep_useful_lines(slopes, intercepts)
    eps_ls_reduced = slope_to_eps(useful_slopes)
    delta_ls_reduced = intercept_to_delta(useful_intercepts)

    n = len(eps_ls_reduced)

    delta_1 = delta_ls_reduced[0]
    delta_tilde = 1 - ((1 - delta_1) ** k)

    sigmas = np.zeros_like(eps_ls_reduced)
    sigmas[0] = 1
    for i in range(1, n):
        exp_eps_i_1 = np.exp(eps_ls_reduced[i-1])
        exp_eps_i = np.exp(eps_ls_reduced[i])
        delta_i_1 = delta_ls_reduced[i-1]
        delta_i = delta_ls_reduced[i]

        sigmas[i] = exp_eps_i_1 * (1-delta_i) - exp_eps_i * (1-delta_i_1) + (delta_i_1 - delta_i)
        sigmas[i] /= (exp_eps_i_1 - exp_eps_i)
        sigmas[i] /= (1-delta_1)

    alphas = np.zeros_like(sigmas)
    for i in range(n-1):
        alphas[i] = sigmas[i] - sigmas[i+1]
    alphas[-1] = sigmas[-1]

    weights = [delta_tilde]
    functions: List[MultiEpsDeltaTradeoff] = [SingleEpsDeltaTradeoff(0, 1)]

    for j in list(__weak_compositions(n, k)):
        j = np.array(j)
        weight = (1-delta_tilde) * __multinomial(j) * np.prod([alphas[i] ** j[i] for i in range(n)])
        weights.append(weight)

        eps_j_list = np.concatenate([np.repeat(eps_ls_reduced[i], j[i]) for i in range(n)])
        functions.append(heter_comp_generalized(eps_j_list, np.zeros_like(eps_j_list)))

    return TradeOffFunction.weighted_infimal_convolution(weights, functions)
