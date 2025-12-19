from definitions import *
from multi_dp_mixture.dp_functions import MultiEpsDeltaTradeoff, SingleEpsDeltaTradeoff


class SmoothTradeOffFunction(TradeOffFunction, ABC):
    """
    Defines a smooth trade-off function to compute fixed points, derivatives,
    approximations, and other transformation calculations.

    :ivar _cached_c: Caches the computed fixed-point value for efficiency.
    :type _cached_c: float or None
    """

    def __init__(self):
        self._cached_c: float | None = None

    @abstractmethod
    def __call__(self, x: Array) -> Array:
        """
        Evaluate the function at x.

        :param x: Evaluation point.
        :type x: Array
        :return: Function value at x.
        """
        pass

    @abstractmethod
    def derivative_at(self, x: Array) -> Array:
        """
        Evaluate the derivative of the function at x.

        :param x: Evaluation point.
        :type x: Array
        :return: Derivative value at x.
        """
        pass

    @abstractmethod
    def second_derivative_at(self, x: Array) -> Array:
        """
         Evaluate the second derivative of the function at x.

         :param x: Evaluation point.
         :type x: Array
         :return: Second derivative value at x.
         """
        pass


    def find_pivot_approx_above(self) -> float | None:
        return None

    def find_pivot_approx_below(self) -> float | None:
        return None

    def fixed_point(self) -> float:
        """
        Finds and returns the fixed point of a function by solving the equation f(x) = x,
        where f is defined by an implementation of this instance. Caches the result.

        :return: The fixed point of the function
        :rtype: float
        """
        if self._cached_c is not None:
            return self._cached_c

        self._cached_c = spo.root_scalar(
            f=lambda x:self(x) - x,
            fprime=self.derivative_at,
            fprime2=self.second_derivative_at,
            bracket=(0, 1),
            x0=1./2.
        ).root

        return self._cached_c

    def rotation_change(self, u: float) -> Callable[[Array], Array]:
        """
        45 deg rotation change of variable.

        :param u: Rotated evaluation point.
        :type u: float
        :return: Function to minimize to obtain the rotated evaluation point.
        :rtype: Callable[[Array], Array]
        """
        return lambda x: (x - self(x))/np.sqrt(2) - u

    def rotation_change_deriv(self) -> Callable[[Array], Array]:
        """
        45 deg rotation change of variable derivative.

        :param u: Rotated evaluation point.
        :type u: float
        :return: Derivative of the function to minimize to obtain the rotated evaluation point.
        :rtype: Callable[[Array], Array]
        """
        return lambda x: (1 - self.derivative_at(x))/np.sqrt(2)

    def rotation_change_second_deriv(self) -> Callable[[Array], Array]:
        """
        45 deg rotation change of variable second derivative.

        :param u: Rotated evaluation point.
        :type u: float
        :return: Second derivative of the function to minimize to obtain the rotated evaluation point.
        :rtype: Callable[[Array], Array]
        """
        return lambda x: - self.second_derivative_at(x)/np.sqrt(2)

    @staticmethod
    def compute_eps_from_alpha(alpha) -> float:
        """
        Compute epsilon guarantee from rotated alpha slope.

        :param alpha: Rotated slope.
        :type alpha: float
        :return: float
        """
        return np.log(-(alpha - 1)/(alpha + 1))

    @staticmethod
    def compute_delta_from_alpha_beta(alpha, beta) -> float:
        """
         Compute delta guarantee from rotated alpha slope and beta intercept.

         :param alpha: Rotated slope.
         :type alpha: float
         :param beta: Rotated intercept.
         :type beta: float
         :return: float
         """
        return 1 - np.sqrt(2) * beta/(alpha + 1)

    def approx_from_below(self, g: 'NormalRotation' = None) -> MultiEpsDeltaTradeoff:
        """
        Compute an approximation from below for the given tradeoff function.

        :param g: NormalRotation object representing a tradeoff mechanism. If not provided,
            a new `NormalRotation` instance will be created using the current object.
        :return: MultiEpsDeltaTradeoff object representing the approximated tradeoff curve.
        """
        if g is None:
            g = NormalRotation(self)

        t_approx_below = self.find_pivot_approx_below()
        if t_approx_below is not None:
            t_star = t_approx_below
        else:
            t_star = spo.root_scalar(
                f=g.preprocess_for_approx_below(),
                x0=g.get_z()/2,
                fprime=True
            ).root

        t_1 = (t_star + g.get_z())/2
        t_2 = t_star/2

        alpha_2 = g.derivative_at(t_2)
        beta_2 = g.call(t_2) - alpha_2 * t_2
        eps_2 = SmoothTradeOffFunction.compute_eps_from_alpha(alpha_2)
        delta_2 = SmoothTradeOffFunction.compute_delta_from_alpha_beta(alpha_2, beta_2)

        if np.abs(t_1 - t_star) < TOL:
            return SingleEpsDeltaTradeoff(eps_2, delta_2)

        alpha_1 = g.derivative_at(t_1)
        beta_1 = g.call(t_1) - alpha_1 * t_1
        eps_1 = SmoothTradeOffFunction.compute_eps_from_alpha(alpha_1)
        delta_1 = SmoothTradeOffFunction.compute_delta_from_alpha_beta(alpha_1, beta_1)

        return MultiEpsDeltaTradeoff([eps_1, eps_2], [delta_1, delta_2])

    def approx_from_above(self) -> MultiEpsDeltaTradeoff:
        """
        Computes an approximation from above for the tradeoff function, using provided
        methods to calculate derivatives and fixed points. Depending on conditions, it
        returns either a single epsilon-delta tradeoff or a multi-epsilon-delta tradeoff.

        :return: A `MultiEpsDeltaTradeoff` object containing the computed epsilon-delta
            tradeoff approximations.
        :rtype: MultiEpsDeltaTradeoff
        """
        offset = (self.fixed_point() - self(0))/self.fixed_point()

        t_approx_above = self.find_pivot_approx_above()
        if t_approx_above is not None:
            t_s = t_approx_above
        else:
            t_s = spo.root_scalar(
                f=lambda t: self.derivative_at(t) - offset,
                fprime=self.second_derivative_at,
                x0=self.fixed_point()/2
            ).root

        h_ts = t_s * self(0) + self.fixed_point() * (self(t_s) - self(0) - t_s)

        if h_ts > 0:
            eps = np.log((self(0) - self.fixed_point())/self.fixed_point())
            delta = 1-self(0)
            return SingleEpsDeltaTradeoff(eps, delta)

        else:
            eps_1 = np.log((self(0) - self(t_s))/t_s)
            delta_1 = 1-self(0)
            log_arg = (self.fixed_point()-self(t_s))/(t_s - self.fixed_point())
            eps_2 = np.log(log_arg)
            delta_2 = 1-self.fixed_point()-self.fixed_point() * log_arg
            return MultiEpsDeltaTradeoff([eps_1, eps_2], [delta_1, delta_2])


class NormalRotation:

    """
    45 degree rotation of a smooth trade-off function.
    """

    def __init__(self, f: SmoothTradeOffFunction):
        self._f = f
        self._z = -f(0)/np.sqrt(2)

    def get_z(self):
        """
        Left bound of the rotation interval.
        :return: float
        """
        return self._z

    def preprocess_for_approx_below(self):
        """
        Prepare function to find parameter t for approximation from below.

        :return: Function to evaluate in root finding problem, and its derivative.
        """
        def eval_and_deriv(t):
            u1 = (t + self._z)/2
            u2 = t/2
            x1 = self.invert_u(u1)
            x2 = self.invert_u(u2)

            g_prime_u1 = self.derivative_at(u1, x1)
            g_prime_u2 = self.derivative_at(u2, x2)

            g_1 = self.call(u1, x1) + g_prime_u1 * (t-self._z)/2
            g_2 = self.call(u2, x2) + g_prime_u2 * t/2

            call = g_1 - g_2

            deriv_1 = g_prime_u1 + self.second_derivative_at(u1, x1) * (t-self._z)/4
            deriv_2 = g_prime_u2 + self.second_derivative_at(u2, x2) * t/4

            deriv = deriv_1 - deriv_2

            return call, deriv

        return eval_and_deriv

    def invert_u(self, u: float) -> float:
        """
        Invert the input value `u` to find the corresponding root of the equation
        defined by the function rotation change and its derivatives.

        :param u: Input value to be inverted.
        :type u: float
        :return: The root found for the corresponding input value `u`.
        :rtype: float
        """
        return spo.root_scalar(
            f=self._f.rotation_change(u),
            x0=self._f.fixed_point()/2,
            fprime=self._f.rotation_change_deriv(),
            fprime2=self._f.rotation_change_second_deriv()
        ).root

    def call(self, u: Array, x_u: Array = None) -> Array:
        """
        Evaluate the rotated function at `u` using the precomputed `x_u` values, or compute them if not provided.

        :param u: Input array or a single value used for computation.
        :param x_u: Optional precomputed array or single value to bypass
            the default computation of `x_u`, root of u.
        :return: Rotated function value at u
        """
        if x_u is None:
            x_u = [self.invert_u(ui) for ui in u] if type(u) is Array else self.invert_u(u)
        return (x_u + self._f(x_u))/np.sqrt(2)

    def __call__(self, u: Array) -> Array:
        return self.call(u)

    def derivative_at(self, u: Array, x_u: Array = None) -> Array:
        """
        Evaluate the derivative of rotated function at `u` using the precomputed `x_u` values,
        or compute them if not provided.

        :param u: Input array or a single value used for computation.
        :param x_u: Optional precomputed array or single value to bypass
            the default computation of `x_u`, root of u.
        :return: Rotated function value at u
        """
        if x_u is None:
            x_u = [self.invert_u(ui) for ui in u] if type(u) is Array else self.invert_u(u)
        f_prime_x_u = self._f.derivative_at(x_u)
        return (1+f_prime_x_u)/(1-f_prime_x_u)

    def second_derivative_at(self, u: Array, x_u: Array = None) -> Array:
        """
        Evaluate the second derivative of rotated function at `u` using the precomputed `x_u` values,
        or compute them if not provided.

        :param u: Input array or a single value used for computation.
        :param x_u: Optional precomputed array or single value to bypass
            the default computation of `x_u`, root of u.
        :return: Rotated function value at u
        """
        if x_u is None:
            x_u = [self.invert_u(ui) for ui in u] if type(u) is Array else self.invert_u(u)
        f_prime_x_u = self._f.derivative_at(x_u)
        f_second_x_u = self._f.second_derivative_at(x_u)
        return (2 * np.sqrt(2) * f_second_x_u)/((1-f_prime_x_u) ** 3)
