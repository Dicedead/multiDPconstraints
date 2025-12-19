from definitions import *
from f_dp_approximation.tradeoff_function import SmoothTradeOffFunction


class GaussianTradeoff(SmoothTradeOffFunction):
    """
    Represent a tradeoff function T(N(mu,1), N(0,1)).
    """

    def __init__(self, mu: float):
        """
        Parameter mu is the mean of the underlying Gaussian distribution compared against the
        standard normal distribution.

        :param mu: float
        """
        self._mu = mu
        super().__init__()

    @staticmethod
    def compute_mu_from_eps_delta(eps: float, delta: float) -> float:
        """
        Compute the parameter mu based on the given epsilon and delta in order for
        the corresponding tradeoff function to achieve at least (eps, delta)-DP.

        :param eps: Positive float value representing the epsilon parameter.
        :param delta: Positive float value within the interval (0, 1] representing the
            delta parameter.
        :return: The computed mu as a float value.
        """
        assert 1 >= delta > 0
        assert eps >= 0
        return eps/np.sqrt(2 * np.log(5/(4 * delta)))

    def __call__(self, x: Array) -> Array:
        return spt.norm.cdf(spt.norm.ppf(1 - x) - self._mu)

    @staticmethod
    def __pdf_prime(x):
        return (-x/np.sqrt(2*np.pi)) * np.exp(-(x**2)/2)

    def derivative_at(self, x: Array) -> Array:
        quantile = spt.norm.ppf(1 - x)
        return -spt.norm.pdf(quantile - self._mu) / spt.norm.pdf(quantile)

    def second_derivative_at(self, x: Array) -> Array:
        quantile = spt.norm.ppf(1 - x)
        denom_sq = spt.norm.pdf(quantile)
        num_first_term = self.__pdf_prime(quantile - self._mu)
        num_second_term = spt.norm.pdf(quantile - self._mu) * self.__pdf_prime(quantile) / denom_sq
        num = num_first_term - num_second_term
        return num / (denom_sq * denom_sq)
