import numpy as np

from base.definitions import *
from base.real_function import RealFunction


class TradeOffFunction(RealFunction, ABC):
    """
    Represents an abstract tradeoff function.
    """

    def fixed_point(self) -> float:
        """
        Finds and returns the fixed point of a function by solving the equation f(x) = x,
        where f is defined by an implementation of this instance.

        By default, looks for the fixed point using the bisection method implemented with Scipy.

        :return: The fixed point of the function
        :rtype: float
        """

        c = spo.root_scalar(
            f=lambda x: self(x) - x,
            bracket=(0, 1),
            x0=1. / 2.
        ).root
        return c

    def rotation_change(self, u: float) -> Callable[[Array], Array]:
        """
        45 deg rotation change of variable.

        :param u: Rotated evaluation point.
        :type u: float
        :return: Function to find the root of to obtain the rotated evaluation point.
        :rtype: Callable[[Array], Array]
        """
        return lambda x: (x - self(x))/np.sqrt(2) - u

    def normal_rotation(self) -> 'NormalRotation':
        return NormalRotation(self)

    @abstractmethod
    def subgradient_at(self, x: float) -> float:
        pass

    @abstractmethod
    def __call__(self, x: Array) -> Array:
        pass

    @staticmethod
    def weighted_infimal_convolution(weights: Array, f_arr: List['TradeOffFunction']) -> 'TradeOffFunction':
        """
        Computes the weighted infimal convolution of a list of TradeOffFunction objects,
        given their corresponding weights.

        Note that we know that the infimal convolution of trade-off functions is a trade-off function, hence why
        this function outputs a TradeOffFunction object and not a ConvexFunction object.

        :param weights: Array containing the weights for the infimal convolution operation.
        :param f_arr: List of TradeOffFunction objects on which the weighted infimal
            convolution is to be performed.
        :return: Resulting TradeOffFunction after performing the weighted
            infimal convolution.
        """
        assert len(weights) == len(f_arr)

        weights = np.array(weights)
        mask = np.argwhere(weights > 0).reshape(-1)
        weights = [float(x) for x in weights[mask]]
        f_arr = [f_arr[i] for i in mask]

        f_star = weights[0] * f_arr[0].convex_conjugate()
        for i in range(1, len(weights)):
            f_star += weights[i] * f_arr[i].convex_conjugate()
        f_mixture = f_star.convex_conjugate()
        return f_mixture

    @staticmethod
    def intersection(f_arr: List['TradeOffFunction']) -> 'TradeOffFunction':

        class IntersectedTradeoffFunction(TradeOffFunction):

            def convex_conjugate(self) -> 'RealFunction':
                # it can be implemented, but we do not need it for now
                pass

            def __call__(self, x: Array) -> Array:
                return np.max(np.array([f(x) for f in f_arr]), axis=0)

            def fixed_point(self) -> float:
                candidates = np.array([f.fixed_point() for f in f_arr])
                values = np.abs(self(candidates) - candidates)
                return candidates[np.argmin(values)]

            def subgradient_at(self, x: float) -> float:
                return f_arr[np.argmax(np.array([f(x) for f in f_arr]))].subgradient_at(x)

        return IntersectedTradeoffFunction()

class NormalRotation:
    """
    45 degree rotation of a trade-off function.
    """

    def __init__(self, f: TradeOffFunction):
        self._f = f
        self._z = -f(0)/np.sqrt(2)

    def get_z(self):
        """
        Left bound of the rotation interval.
        :return: float
        """
        return self._z


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
            x0=self._f.fixed_point()/2
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

    def subgradient_at(self, u: float) -> Array:
        return NormalRotation.slope_forward_rotation(self._f.subgradient_at(self.invert_u(u)))

    def __call__(self, u: Array) -> Array:
        return self.call(u)

    @staticmethod
    def slope_forward_rotation(a: Array) -> Array:
        """
        Rotate the slope of a line in the original (y=0, x=0) coordinate system
        to the new (y=-x,y=x) coordinate system.

        :param a: slope in original coordinate system
        :type a: Array
        :return: rotated slope
        :rtype: Array
        """
        a = np.array(a)
        return (1+a)/(1-a)

    @staticmethod
    def slope_offset_forward_rotation(a: Array, b: Array) -> Tuple[Array,Array]:
        """
         Rotate the slope and offset of a line in the original (y=0, x=0) coordinate system
         to the new (y=-x,y=x) coordinate system.

         :param a: slope in original coordinate system
         :type a: Array
         :param b: offset in original coordinate system
         :type b: Array
         :return: rotated slope and offset
         :rtype: Tuple[Array,Array]
         """
        a = np.array(a)
        b = np.array(b)
        assert np.shape(a) == np.shape(b)
        alpha = NormalRotation.slope_forward_rotation(a)
        beta = b * (alpha - 1) / (a * np.sqrt(2))
        return alpha, beta

    @staticmethod
    def slope_inversion(alpha: Array) -> Array:
        """
        Rotate the slope of a line in the rotated (y=-x,y=x) coordinate system
        to the original (y=0, x=0) coordinate system.

        :param alpha: slope in rotated coordinate system
        :type alpha: Array
        :return: slope in original coordinate system
        :rtype: Array
        """
        alpha = np.array(alpha)
        return (alpha-1)/(alpha+1)

    @staticmethod
    def slope_offset_rotation_inversion(alpha: Array, beta: Array) -> Tuple[Array, Array]:
        """
        Rotate the slope and offset of a line in the rotated (y=-x,y=x) coordinate system
        to the original (y=0, x=0) coordinate system.

        :param alpha: slope in rotated coordinate system
        :type alpha: Array
        :param beta: offset in rotated coordinate system
        :type beta: Array
        :return: slope and offset in original coordinate system
        :rtype: Tuple[Array, Array]
        """
        alpha = np.array(alpha)
        beta = np.array(beta)
        assert np.shape(alpha) == np.shape(beta)
        a = NormalRotation.slope_inversion(alpha)
        b = np.sqrt(2) * beta/(alpha + 1)
        return a, b
