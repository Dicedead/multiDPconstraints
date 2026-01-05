from base.definitions import *
from multi_dp_mixture.dp_functions import MultiEpsDeltaTradeoff


def tv_of_eps_delta(eps: float, delta: float) -> float:
    """
    Maximum total variation of (eps,delta)-DP mechanism.

    :param eps: Epsilon parameter of the differentially private mechanism.
    :type eps: float
    :param delta: Delta parameter of the differentially private mechanism.
    :type delta: float
    :return: Max total variation of the mechanism.
    :rtype: float
    """
    return delta + (1-delta) * (np.exp(eps) - 1) / (np.exp(eps) + 1)


def privacy_region_composition_exact(eps: float, delta: float, k: int) -> MultiEpsDeltaTradeoff:
    """
    Compute the differential privacy composition region corresponding to the improved result for the composition
    of differentially private mechanisms.

    :param eps: Epsilon parameter of the differentially private mechanisms being composed.
    :type eps: float
    :param delta: Delta parameter of the differentially private mechanisms being composed.
    :type delta: float
    :param k: Number of composed mechanisms.
    :type k: int
    :return: Piecewise affine function representing the composition region.
    :rtype: PiecewiseAffine
    """
    assert eps >= 0
    assert 0 <= delta <= 1
    assert k >= 0

    eps_ls = []
    delta_ls = []
    for i in range(int(np.floor(k/2)+1)):
        eps_prime = (k - 2 * i) * eps
        delta_tmp = sum([sps.comb(k, l) * (np.exp((k-l) * eps) - np.exp((k-2*i+l) * eps)) for l in range(i)])
        delta_tmp /= (1+np.exp(eps)) ** k
        delta_prime = 1 - ((1 - delta) ** k) * (1 - delta_tmp)
        eps_ls.append(eps_prime)
        delta_ls.append(delta_prime)

    return MultiEpsDeltaTradeoff(eps_ls, delta_ls)

def privacy_region_dp_composition_total_var(
        eps: float,
        delta: float,
        eta: float,
        k: int
) -> MultiEpsDeltaTradeoff:
    """
    Compute the differential privacy composition region corresponding to the improved result for the composition
    of differentially private mechanisms, accounting for the total variation of the considered mechanisms.

    :param eps: Epsilon parameter of the differentially private mechanisms being composed.
    :type eps: float
    :param delta: Delta parameter of the differentially private mechanisms being composed.
    :type delta: float
    :param eta: Total variation of the considered mechanisms.
    :type eta: float
    :param k: Number of composed mechanisms.
    :type k: int
    :return: Piecewise affine function representing the composition region.
    :rtype: PiecewiseAffine
    """
    assert eps >= 0
    assert delta >= 0
    assert k >= 0
    assert eta >= 0

    alpha = 1 - (eta - delta) * (1 + np.exp(eps)) / ((1 - delta) * (np.exp(eps) - 1))

    eps_ls = []
    delta_ls = []
    for j in range(k+1):
        eps_prime = j * eps
        delta_tmp = sum(
            [ sps.comb(k, a) * sum(
                [
                    sps.comb(k-a, l) * (((1-alpha)/(1+np.exp(eps))) ** (k-a)) * \
                    (alpha ** a) * (np.exp((k-l-a)*eps) - np.exp((l+j)*eps))
                    for l in range(int(np.ceil((k-j-a)/2.)))
                ]
            )
                for a in range(k-j)
              ]
        )
        delta_prime = max(0., 1 - ((1 - delta) ** k) * (1 - delta_tmp))
        eps_ls.append(eps_prime)
        delta_ls.append(delta_prime)


    return MultiEpsDeltaTradeoff(eps_ls, delta_ls)