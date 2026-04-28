from base.definitions import *
from base.tradeoff_function import TradeOffFunction
from main_theorems.heterogeneous_version import privacy_region_composition_heterogeneous
from multi_dp_mixture.dp_functions import MultiEpsDeltaTradeoff, SingleEpsDeltaTradeoff
from multi_dp_mixture.piecewise_affine import PiecewiseAffine


def heter_comp_generalized(epsilons, deltas, return_eps_deltas: bool = False) -> MultiEpsDeltaTradeoff:
    """
    Computes all exact (eps_g, delta_g) pairs forming the privacy profile or heterogeneous composition of k
    differentially private mechanisms.

    :param epsilons: List of k epsilon parameters.
    :type epsilons: List[float]
    :param deltas: List of k delta parameters.
    :type deltas: List[float]
    :param return_eps_deltas: If True, also returns the list of epsilon and delta value pairs.
    :type return_eps_deltas: bool
    :return: MultiEpsDeltaTradeoff object representing the privacy profile of the heterogeneous composition.
    """

    assert len(epsilons) == len(deltas), "Lengths of epsilons and deltas must match."
    k = len(epsilons)

    prod_1_minus_delta = math.prod(1. - d for d in deltas)
    delta_base = 1 - prod_1_minus_delta

    denom_lhs = math.prod(1 + math.exp(e) for e in epsilons)

    subsets_data = []
    for S in itertools.product([False, True], repeat=k):
        # represent subsets by a length k boolean array, eps_i in the set <=> S[i] is True

        sum_in_S = sum(e for i, e in enumerate(epsilons) if S[i])
        sum_notin_S = sum(e for i, e in enumerate(epsilons) if not S[i])

        w = sum_in_S - sum_notin_S

        term_sum_in_s = math.exp(sum_in_S) / denom_lhs
        term_sum_notin_s = math.exp(sum_notin_S) / denom_lhs

        subsets_data.append((w, term_sum_in_s, term_sum_notin_s))

    critical_eps_g = sorted([w for w, _, _ in subsets_data if w >= -1e-12]) # keep only eps_g large enough + sort

    # deduplicate floating point values to prevent redundant calculations
    unique_eps_g = []
    for e in critical_eps_g:
        if not unique_eps_g or (e - unique_eps_g[-1]) > 1e-9:
            unique_eps_g.append(max(0., e))  # avoid -0.0

    eps_delta_ls = []
    for eps_g in unique_eps_g:
        lhs_sum = 0.
        for w, term_sum_in_s, term_sum_notin_s in subsets_data:
            if w > eps_g:
                lhs_sum += term_sum_in_s - math.exp(eps_g) * term_sum_notin_s

        delta_g = delta_base + lhs_sum * prod_1_minus_delta
        eps_delta_ls.append((eps_g, delta_g))

    eps_ls = [eps for (eps, delta) in eps_delta_ls]
    delta_ls = [delta for (eps, delta) in eps_delta_ls]
    f = MultiEpsDeltaTradeoff(eps_ls, delta_ls)

    if return_eps_deltas:
        return f, eps_delta_ls

    return f


if __name__ == "__main__":
    eps_1 = 1.3
    eps_2 = 0.5
    x = 3
    y = 2
    eps = [eps_1] * x + [eps_2] * y
    dls = [0] * len(eps)

    _, new_imp = heter_comp_generalized(eps, dls, return_eps_deltas=True)

    _, current_imp = privacy_region_composition_heterogeneous(eps_1, eps_2, x, y, True)
    current_imp.sort(key=lambda x: x[0])
    new_imp.sort(key=lambda x: x[0])

    print(current_imp)
    print(new_imp)
