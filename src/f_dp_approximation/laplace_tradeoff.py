from f_dp_approximation.smooth_tradeoff_function import SmoothTradeOffFunction
from base.definitions import *


class LaplaceTradeoff(SmoothTradeOffFunction):
    """
    Represent a tradeoff function T(Laplace(0,1), Laplace(eps,1)). Note that this is not a strictly convex function
    and twice differentiable function, hence we do not expect the approximation to always work.
    """

    def __init__(self, eps: float):
        """
        Initializes an instance of the class with the specified privacy budget parameter.

        :param eps: The privacy budget parameter. Must be a non-negative float.
        :type eps: float
        """
        assert eps >= 0
        self._eps = eps
        self._exp_eps = np.exp(eps)
        self._cutoff = np.exp(-eps)/2
        super().__init__()

    def __call__(self, x: Array) -> Array:
        return spt.laplace.cdf(spt.laplace.ppf(1 - x) - self._eps)

    @staticmethod
    def __pdf_prime(x):
        return -np.sign(x) * spt.laplace.pdf(x)

    def derivative_at(self, x: Array) -> Array:
        quantile = spt.laplace.ppf(1 - x)
        return -spt.laplace.pdf(quantile - self._eps) / spt.laplace.pdf(quantile)

    def second_derivative_at(self, x: Array) -> Array:
        quantile = spt.laplace.ppf(1 - x)
        denom_sq = spt.laplace.pdf(quantile)
        num_first_term = self.__pdf_prime(quantile - self._eps)
        num_second_term = spt.laplace.pdf(quantile - self._eps) * self.__pdf_prime(quantile) / denom_sq
        num = num_first_term - num_second_term
        return num / (denom_sq * denom_sq)

    def fixed_point(self) -> float:
        return np.exp(-self._eps/2)/2 # 0.3032653298563167