from f_dp_approximation.tradeoff_function import SmoothTradeOffFunction
from definitions import *


class LaplaceTradeoff(SmoothTradeOffFunction):
    # TODO write that this is supposed to fail because Laplace trade-off function is not strictly convex nor smooth

    def __init__(self, eps: float):
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