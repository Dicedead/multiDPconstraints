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

        kept_slopes, kept_intercepts = keep_only_useful_pairs()

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
        if self._bounded_domain:
            breakpoints = [self._domain_start, self._domain_end]
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

            return PiecewiseAffine(
                breakpoints,
                new_intercepts,
                domain_start=DEFAULT_DOMAIN_START,
                domain_end=DEFAULT_DOMAIN_END,
                bounded=False
            )

        pairs = list(zip(self._slopes, self._intercepts))
        pairs.sort(key=lambda pair: pair[0])
        sorted_slopes, sorted_intercepts = list(zip(*pairs))  # unzips the list of pairs into two lists

        new_slopes = np.zeros(len(sorted_slopes)-1)
        new_intercepts = np.zeros(len(sorted_slopes)-1)

        for j in range(len(sorted_slopes)-1):
            b_j = sorted_intercepts[j]
            b_jp1 = sorted_intercepts[j+1]
            a_j = sorted_slopes[j]
            a_jp1 = sorted_slopes[j+1]

            b = -b_j + a_j * (b_jp1 - b_j) / (a_jp1 - a_j)
            a = -(b_jp1 - b_j) / (a_jp1 - a_j)

            new_slopes[j] = a
            new_intercepts[j] = b

        return PiecewiseAffine(
            new_slopes,
            new_intercepts,
            sorted_slopes[0],
            sorted_slopes[-1],
            not self._bounded_domain
        )

    def __add__(self, other: 'PiecewiseAffine') -> 'PiecewiseAffine':
        new_slopes = []
        new_intercepts = []

        i = 0
        for self_slope, self_intercept in zip(self._slopes, self._intercepts):
            for other_slope, other_intercept in zip(other._slopes, other._intercepts):
                new_slopes.append(self_slope + other_slope)
                new_intercepts.append(self_intercept + other_intercept)
                i += 1

        return PiecewiseAffine(
            new_slopes,
            new_intercepts,
            max(self._domain_start, other._domain_start),
            min(self._domain_end, other._domain_end),
            self._bounded_domain or other._bounded_domain
        )

    def __mul__(self, other: float) -> 'PiecewiseAffine':
        return PiecewiseAffine(
            other * self._slopes,
            other * self._intercepts,
            self._domain_start,
            self._domain_end,
            self._bounded_domain
        )

    def __rmul__(self, other: float) -> 'PiecewiseAffine':
        return self.__mul__(other)

    def to_plot(self, num_points=100):
        x = np.linspace(self._domain_start, self._domain_end, num_points)
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
        assert len(weights) == len(f_arr)

        f_star = weights[0] * (f_arr[0].convex_conjugate())
        for i in range(1, len(weights)):
            f_star = f_star +  (weights[i] * (f_arr[i].convex_conjugate()))
        f_star.to_plot()
        f_mixture = f_star.convex_conjugate()
        return f_mixture

    @staticmethod
    def plot_multiple_functions(f_arr: List['PiecewiseAffine'], labels: List[str], num_points=100):
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