import numpy as np

from base.tradeoff_function import TradeOffFunction
from multi_dp_mixture.piecewise_affine import PiecewiseAffine
from base.definitions import *

def eps_to_slope(eps: Array) -> Array:
    """
    Converts a given epsilon value to a slope value using the negative exponential function.

    :param eps: The epsilon value to be converted.
    :type eps: Array
    :return: The computed slope after applying the transformation.
    :rtype: Array
    """
    return -np.exp(eps)

def slope_to_eps(slope: Array) -> Array:
    """
    Convert a slope value to a corresponding epsilon value used in calculations.


    :param slope: The slope value to be converted into an epsilon.
    :type slope: Array
    :return: The computed epsilon value obtained from the given slope.
    :rtype: Array
    """
    return np.clip(np.log(-slope), 0, None)

def delta_to_intercept(delta: Array) -> Array:
    """
    Converts a given delta to its corresponding intercept value.

    :param delta: The delta value to be converted.
    :type delta: Array
    :return: The calculated intercept value.
    :rtype: Array
    """
    return 1-delta

def intercept_to_delta(inter: Array) -> Array:
    """
    Converts the given intercept value to its corresponding delta value.

    :param inter: The intercept value.
    :type inter: Array
    :return: The resulting delta value.
    :rtype: Array
    """
    return np.clip(1-inter, 0, 1)

def get_all_slopes_intercepts_from_eps_delta_ls(eps_ls: Array, delta_ls: Array, with_inverses=True) -> Tuple[Array, Array]:
    """
    Extracts all slopes and intercepts from epsilon and delta inputs.

    :param eps_ls: Array of epsilon values used for calculating the initial slopes
    :type eps_ls: Array
    :param delta_ls: List or array of delta values used for calculating the initial intercepts
    :type delta_ls: Array
    :param with_inverses: Whether to add the inverse slopes and intercepts or not
    :type with_inverses: bool
    :return: Tuple of all slopes then all intercepts
    :rtype: Tuple[Array, Array]
    """

    initial_slopes = eps_to_slope(eps_ls)
    inverse_slopes = 1 / initial_slopes

    initial_intercepts = delta_to_intercept(delta_ls)
    inverse_intercepts = -initial_intercepts / initial_slopes

    if with_inverses:
        slopes = np.concatenate([initial_slopes, inverse_slopes, np.r_[0.]])
        intercepts = np.concatenate([initial_intercepts, inverse_intercepts, np.r_[0.]])
    else:
        slopes = initial_slopes
        intercepts = initial_intercepts

    return slopes, intercepts

class MultiEpsDeltaTradeoff(PiecewiseAffine, TradeOffFunction):

    def __init__(self, eps_ls: Array, delta_ls: Array):
        """
        Represents the tradeoff function of a mechanism with multiple (epsilon, delta)-DP constraints.

        :ivar eps_ls: Array of epsilon values used for the tradeoff computation.
        :type eps_ls: Array
        :ivar delta_ls: Array of delta values used for the tradeoff computation.
        :type delta_ls: Array
        """

        self._eps_ls = np.array(eps_ls.copy())
        self._delta_ls = np.array(delta_ls.copy())
        self._eps_ls.flags.writeable = False
        self._delta_ls.flags.writeable = False
        slopes, intercepts = get_all_slopes_intercepts_from_eps_delta_ls(self._eps_ls, self._delta_ls)

        super().__init__(slopes, intercepts, domain_start=0., domain_end=1., bounded=True)

        self._fixed_point = self.__compute_fixed_point()

    def get_eps_list(self) -> Array:
        return self._eps_ls

    def get_delta_list(self) -> Array:
        return self._delta_ls

    def fixed_point(self) -> float:
        return self._fixed_point

    def subgradient(self, x: float, tol=1e-9) -> float:
        return self.subgradient(x, tol)

    def __compute_fixed_point(self) -> float:
        candidates = self._intercepts / (1 - self._slopes)
        values = np.abs(self(candidates) - candidates)
        return candidates[np.argmin(values)]

    @staticmethod
    def from_slopes_and_offsets(slopes: Array, offsets: Array) -> 'MultiEpsDeltaTradeoff':
        return MultiEpsDeltaTradeoff(
            slope_to_eps(np.array(slopes)),
            intercept_to_delta(np.array(offsets))
        )

class SingleEpsDeltaTradeoff(MultiEpsDeltaTradeoff):
    def __init__(self, eps: float, delta: float):
        """
        Represents a single (epsilon, delta)-DP tradeoff curve.

        :ivar eps: The epsilon value characterising the privacy tradeoff.
        :type eps: Float
        :ivar delta: The delta value characterising the privacy tradeoff.
        :type delta: Float
        """
        self._eps = eps
        self._delta = delta
        super().__init__([eps], [delta])

    def get_eps(self):
        return self._eps

    def get_delta(self):
        return self._delta
