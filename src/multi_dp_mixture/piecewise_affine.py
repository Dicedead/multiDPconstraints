from src.definitions import *


class PiecewiseAffine:
    """
    Represents a piecewise affine function.

    This class models a piecewise affine function defined by a collection of
    linear equations, each specified by a slope and an intercept.
    """

    def __init__(self, slopes: Array, intercepts: Array):
        """
        Initializes the PiecewiseAffine object with slopes and intercepts.

        This constructor takes two lists representing slopes and intercepts of
        linear equations, converts them into numpy arrays, and stores them as
        internal attributes.

        :param slopes: List of slopes corresponding to the linear equations.
        :type slopes: Array
        :param intercepts: List of intercepts corresponding to the linear equations,
                           of the same length as the list of slopes.
        :type intercepts: Array
        """
        self._slopes = np.array(slopes.copy())
        self._inner_slopes = self._slopes.reshape(1, -1)

        self._intercepts = np.array(intercepts.copy())
        self._inner_intercepts = self._intercepts.reshape(1, -1)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the maximum value across a computed set of linear equations for each input element.

        :param x: Input array, evaluation point of the piecewise affine function.
        :type x: np.ndarray, last dimension of the same length as lists of slopes and intercepts.
        :return: The maximum computed value along the last axis of the transformed array.
        :rtype: np.ndarray
        """
        x = x.reshape(-1, 1)
        max_input = self._inner_slopes * x + self._inner_intercepts
        return np.max(max_input, axis=-1)

    def convex_conjugate(self) -> 'PiecewiseAffine':
        ## todo: implement convex conjugate of a piecewise affine function
        return self

    def __add__(self, other: 'PiecewiseAffine') -> 'PiecewiseAffine':
        ## todo: how to add two piecewise affine functions? probably just add up all possible slope / intercept pairs?
        ## is that too many?
        return self

    def __mul__(self, other: float) -> 'PiecewiseAffine':
        ## todo: check correctness
        return PiecewiseAffine(other * self._slopes, other * self._intercepts)

    def to_plot(self, lower_limit=0, upper_limit=1, num_points=100):
        x = np.linspace(lower_limit, upper_limit, num_points)
        fig = plt.figure()
        ax = fig.add_subplot()
        plt.plot(x, self(x))
        plt.plot(x, IDENTITY(x), "--")
        plt.plot(x, DIAGONAL(x), "--")
        ax.set_aspect('equal', adjustable='box')
        ax.set_autoscale_on(False)
        plt.show()

IDENTITY = PiecewiseAffine([ 1], [0])
DIAGONAL = PiecewiseAffine([-1], [1])