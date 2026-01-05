from base.tradeoff_function import TradeOffFunction
from multi_dp_mixture.piecewise_affine import PiecewiseAffine
from base.definitions import *

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

        initial_slopes = -np.exp(self._eps_ls)
        inverse_slopes = 1/initial_slopes


        initial_intercepts = 1-self._delta_ls
        inverse_intercepts = -initial_intercepts/initial_slopes

        slopes = np.concatenate([initial_slopes, inverse_slopes, np.r_[0.]])
        intercepts = np.concatenate([initial_intercepts, inverse_intercepts, np.r_[0.]])

        super().__init__(slopes, intercepts, domain_start=0., domain_end=1., bounded=True)

    def get_eps_list(self) -> Array:
        return self._eps_ls

    def get_delta_list(self) -> Array:
        return self._delta_ls

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
