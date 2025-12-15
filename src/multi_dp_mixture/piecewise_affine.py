import numpy as np

from src.definitions import *

class PiecewiseAffine:
    """
    Represents a piecewise affine function.

    This class models a piecewise affine function defined by a collection of
    linear equations, each specified by a slope and an intercept.
    """

    def __init__(
            self,
            slopes: Array,
            intercepts: Array,
            domain_start: float = DEFAULT_DOMAIN_START,
            domain_end: float = DEFAULT_DOMAIN_END,
            bounded: bool = False
    ):
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

        def first_pair_is_lower(a1, b1, a2, b2):
            if not bounded:
                if a1 == a2:
                    return b1 <= b2
                return False

            if a1 <= a2:
                return a1 * domain_start + b1 <= a2 * domain_start + b2
            return a1 * domain_end + b1 <= a2 * domain_end + b2

        def keep_only_useful_pairs():
            new_slopes = []
            new_intercepts = []

            for first_slope, first_intercept in zip(slopes, intercepts):
                keep_pair = True
                for second_slope, second_intercept in zip(slopes, intercepts):
                    if first_slope != second_slope or first_intercept != second_intercept:
                        if first_pair_is_lower(first_slope, first_intercept, second_slope, second_intercept):
                            keep_pair = False

                if keep_pair:
                    new_slopes.append(first_slope)
                    new_intercepts.append(first_intercept)

            return new_slopes, new_intercepts

        def keep_only_unique_pairs(sorted_slope_and_offset_pairs):
            # currently not using the fact that they're sorted, applying naive algo instead
            unique_pairs = []
            for slope, offset in sorted_slope_and_offset_pairs:
                if (slope, offset) not in unique_pairs:
                    unique_pairs.append((slope, offset))

            return unique_pairs

        kept_slopes, kept_intercepts = keep_only_useful_pairs()

        pairs = list(zip(kept_slopes, kept_intercepts))
        pairs.sort(key=lambda pair: pair[0])
        pairs = keep_only_unique_pairs(pairs)

        kept_slopes, kept_intercepts = list(zip(*pairs))  # unzips the list of pairs into two lists

        self._slopes = np.array(kept_slopes)
        self._inner_slopes = self._slopes.reshape(1, -1)

        self._intercepts = np.array(kept_intercepts)
        self._inner_intercepts = self._intercepts.reshape(1, -1)

        self._domain_start = domain_start
        self._domain_end = domain_end
        self._bounded_domain = bounded

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
        """
        Computes the convex conjugate of the current piecewise affine function.

        The convex conjugate of a function f is the function defined as:
        f*(y) = sup_x(y * x - f(x)), where sup represents the supremum.

        This method computes the convex conjugate for piecewise affine functions. The
        result is also a piecewise affine function.

        :return: A new instance of `PiecewiseAffine` representing the convex conjugate.
        """
        if self._bounded_domain:
            breakpoints = [self._domain_start, self._domain_end]
        else:
            breakpoints = []

        for i in range(len(self._slopes)-1):
            offset_ip1 = self._intercepts[i+1]
            offset_i = self._intercepts[i]
            slope_ip1 = self._slopes[i+1]
            slope_i = self._slopes[i]

            t_i = -(offset_ip1 - offset_i) / (slope_ip1 - slope_i)
            breakpoints.append(t_i)

        new_intercepts = np.zeros(len(breakpoints))
        for idx, t_i in enumerate(breakpoints):
            new_intercepts[idx] = -self(np.r_[t_i])

        domain_start = -np.Infinity if self._bounded_domain else self._slopes[0]
        domain_end = np.Infinity if self._bounded_domain else self._slopes[-1]

        return PiecewiseAffine(
            breakpoints,
            new_intercepts,
            domain_start=domain_start,
            domain_end=domain_end,
            bounded=not self._bounded_domain
            )

    def __add__(self, other: 'PiecewiseAffine') -> 'PiecewiseAffine':
        """
        Performs addition of two `PiecewiseAffine` objects. The resulting object is a new
        instance with the slopes and intercepts added pairwise. The domain of the
        resulting object is defined based on the domains of the input objects.

        :param other: The `PiecewiseAffine` instance to add to `self`
        :type other: PiecewiseAffine

        :return: A new `PiecewiseAffine` object resulting from the addition operation
        :rtype: PiecewiseAffine
        """
        new_slopes = []
        new_intercepts = []

        for self_slope, self_intercept in zip(self._slopes, self._intercepts):
            for other_slope, other_intercept in zip(other._slopes, other._intercepts):
                new_slopes.append(self_slope + other_slope)
                new_intercepts.append(self_intercept + other_intercept)

        bounded = self._bounded_domain or other._bounded_domain

        return PiecewiseAffine(
            new_slopes,
            new_intercepts,
            domain_start=-np.infty if not bounded else max(self._domain_start, other._domain_start),
            domain_end=np.infty if not bounded else min(self._domain_end, other._domain_end),
            bounded=bounded
        )

    def __mul__(self, other: float) -> 'PiecewiseAffine':
        """
        Multiplies the current PiecewiseAffine instance by a scalar value. This operation
        scales the slopes and intercepts of the affine segments by the given scalar while
        preserving the domain.

        :param other: A scalar value to multiply the slopes and intercepts with.
        :type other: float
        :return: A new PiecewiseAffine instance where the slopes and intercepts are scaled
            by the given scalar.
        :rtype: PiecewiseAffine
        """
        return PiecewiseAffine(
            other * self._slopes,
            other * self._intercepts,
            self._domain_start,
            self._domain_end,
            self._bounded_domain
        )

    def __rmul__(self, other: float) -> 'PiecewiseAffine':
        return self.__mul__(other)

    def rescale(self, factor: float) -> 'PiecewiseAffine':
        return PiecewiseAffine(
            self._slopes / factor,
            self._intercepts,
            self._domain_start,
            self._domain_end,
            self._bounded_domain
        )

    def to_plot(self, num_points=100, start=DEFAULT_DOMAIN_START, end=DEFAULT_DOMAIN_END):
        """
        Generates and displays a plot for the given function over its domain.

        :param num_points: The number of points to use for the domain discretization.
        :type num_points: int
        :param start: If unbounded domain, first x to plot.
        :type start: float
        :param end: If unbounded domain, last x to plot.
        :type end: float
        :return: None
        """
        start = start if not self._bounded_domain else self._domain_start
        end = end if not self._bounded_domain else self._domain_end
        x = np.linspace(start, end, num_points)
        fig = plt.figure()
        ax = fig.add_subplot()
        plt.plot(x, self(x))

        if self._domain_start == DEFAULT_DOMAIN_START and self._domain_end == DEFAULT_DOMAIN_END:
            plt.plot(x, IDENTITY(x), "--")
            plt.plot(x, DIAGONAL(x), "--")
            ax.set_aspect('equal', adjustable='box')
            ax.set_autoscale_on(False)

        plt.show()

    @staticmethod
    def weighted_infimal_convolution(weights: Array, f_arr: List['PiecewiseAffine']) -> 'PiecewiseAffine':
        """
        Computes the weighted infimal convolution of a list of PiecewiseAffine functions,
        given their corresponding weights.

        :param weights: Array containing the weights for the infimal convolution operation.
        :param f_arr: List of PiecewiseAffine objects on which the weighted infimal
            convolution is to be performed.
        :return: Resulting PiecewiseAffine function after performing the weighted
            infimal convolution.
        """
        assert len(weights) == len(f_arr)

        f_star = weights[0] * f_arr[0].convex_conjugate()
        f_arr[0].convex_conjugate().to_plot()
        for i in range(1, len(weights)):
            f_star += weights[i] * f_arr[i].convex_conjugate()
            f_arr[i].convex_conjugate().to_plot()

        f_mixture = f_star.convex_conjugate()
        return f_mixture

    @staticmethod
    def plot_multiple_functions(f_arr: List['PiecewiseAffine'], labels: List[str], num_points=100):
        """
        Plots multiple functions on the same graph, providing a visual comparison
        between a list of given function objects and their respective labels.

        :param f_arr: A list of PiecewiseAffine objects, where each object represents
                      a function to be plotted.
        :type f_arr: List[PiecewiseAffine]
        :param labels: A list of labels corresponding to each function in f_arr,
                       which will be used for the plot's legend.
        :type labels: List[str]
        :param num_points: The granularity of the plot, specifying the number of
                           sample points to generate within the range [0, 1].
                           Defaults to 100.
        :type num_points: int, optional
        :return: None
        """
        assert len(f_arr) == len(labels)

        x = np.linspace(0,1, num_points)
        fig = plt.figure()
        ax = fig.add_subplot()
        for f, label in zip(f_arr, labels):
            plt.plot(x, f(x), label=label)


        plt.plot(x, IDENTITY(x), "k--")
        plt.plot(x, DIAGONAL(x), "k--")
        plt.legend()
        ax.set_aspect('equal', adjustable='box')
        ax.set_autoscale_on(False)

        plt.show()


IDENTITY = PiecewiseAffine([ 1], [0])
DIAGONAL = PiecewiseAffine([-1], [1])