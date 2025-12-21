from base.definitions import *
from base.convex_function import ConvexFunction


class TradeOffFunction(ConvexFunction, ABC):
    """
    Represents an abstract tradeoff function.
    """

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
