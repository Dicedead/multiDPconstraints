from scipy.integrate import quad
from scipy.stats import beta as beta_dist
from scipy.special import iv, gamma
from base.smooth_tradeoff_function import SmoothTradeOffFunction
from base.definitions import *


class VonMisesFisherTradeoff(SmoothTradeOffFunction):
    """
    Represents a von Mises-Fisher-based smooth tradeoff function.

    This class is not yet fully tested. Should be considered experimental.
    """

    def __init__(self, n: int, kappa: float, d: float, grid_points: int = 2000):
        super().__init__()
        self._n = n
        self._kappa = kappa
        self._d = d

        nu = self._n / 2.0 - 1.0
        self._c_vmf = (self._kappa / 2.0) ** nu / (np.sqrt(np.pi) * gamma((self._n - 1) / 2.0) * iv(nu, self._kappa))

        self._max_w = np.sqrt(2 - 2 * self._d)
        self._c_vals = np.linspace(-self._max_w - 0.5, self._max_w + 0.5, grid_points)
        self._f_vals = np.clip([self._compute_f(c) for c in self._c_vals], 0, 1)
        self._p_vals = np.gradient(self._f_vals, self._c_vals)

    def _vmf_mix_pdf(self, t: float) -> float:
        return self._c_vmf * (1 - t ** 2) ** ((self._n - 3) / 2.0) * np.exp(self._kappa * t)

    def _compute_f(self, c: float) -> float:
        def integrand(t):
            if t <= -1.0 or t >= 1.0:
                return 0.0

            A = (1 - self._d) * t - c
            B = np.sqrt(1 - self._d ** 2) * np.sqrt(1 - t ** 2)

            if B <= 1e-12:
                rho_min = 1.0 if A > 0 else -1.0
            else:
                rho_min = np.clip(A / B, -1.0, 1.0)

            u = (rho_min + 1.0) / 2.0
            prob_rho = 1.0 - beta_dist.cdf(u, (self._n - 2) / 2.0, (self._n - 2) / 2.0)

            return self._vmf_mix_pdf(t) * prob_rho

        res = quad(integrand, -1, 1, epsabs=1e-4, epsrel=1e-4, limit=200, full_output=True)[0]
        return res

    def __call__(self, x: Array) -> Array:
        x = np.atleast_1d(x)
        out = np.zeros_like(x, dtype=float)

        for i, alpha in enumerate(x):
            if alpha <= 0.0:
                out[i] = 1.0
            elif alpha >= 1.0:
                out[i] = 0.0
            else:
                c_alpha = np.interp(alpha, self._f_vals, self._c_vals)
                out[i] = np.interp(-c_alpha, self._c_vals, self._f_vals)

        return out.reshape(x.shape)

    def derivative(self, x: Array) -> Array:
        x = np.atleast_1d(x)
        out = np.zeros_like(x, dtype=float)

        for i, alpha in enumerate(x):
            if alpha <= 0.0:
                c_alpha = self._max_w
            elif alpha >= 1.0:
                c_alpha = -self._max_w
            else:
                c_alpha = np.interp(alpha, self._f_vals, self._c_vals)

            out[i] = -np.exp(-self._kappa * c_alpha)

        return out.reshape(x.shape)

    def second_derivative(self, x: Array) -> Array:
        x = np.atleast_1d(x)
        out = np.zeros_like(x, dtype=float)

        for i, alpha in enumerate(x):
            if alpha <= 0.0 or alpha >= 1.0:
                out[i] = np.inf
                continue

            c_alpha = np.interp(alpha, self._f_vals, self._c_vals)
            p_c_alpha = np.interp(c_alpha, self._c_vals, self._p_vals)

            if p_c_alpha > 1e-12:
                out[i] = (self._kappa * np.exp(-self._kappa * c_alpha)) / p_c_alpha
            else:
                out[i] = np.inf

        return out.reshape(x.shape)