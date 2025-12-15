from multi_dp_mixture.piecewise_affine import PiecewiseAffine
from definitions import *

class MultiEpsDeltaTradeoff(PiecewiseAffine):
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

        initial_slopes = -np.exp(self._eps_ls)
        inverse_slopes = 1/initial_slopes


        initial_intercepts = 1-self._delta_ls
        inverse_intercepts = -initial_intercepts/initial_slopes

        slopes = np.concatenate([initial_slopes, inverse_slopes])
        intercepts = np.concatenate([initial_intercepts, inverse_intercepts])

        super().__init__(slopes, intercepts, domain_start=0., domain_end=1., bounded=True)


class SingleEpsDeltaTradeoff(MultiEpsDeltaTradeoff):
    def __init__(self, eps: float, delta: float):
        """
        Represents a single (epsilon, delta)-DP tradeoff curve.


        :ivar eps: The epsilon value characterising the privacy tradeoff.
        :type eps: Float
        :ivar delta: The delta value characterising the privacy tradeoff.
        :type delta: Float
        """
        super().__init__([eps], [delta])

