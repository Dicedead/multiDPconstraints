from base.definitions import *


class ConvexFunction(Callable[[Array], Array], ABC):
    """
    Represents an abstract convex function with its convex conjugate.
    """

    @abstractmethod
    def __call__(self, x: Array) -> Array:
        pass

    @abstractmethod
    def convex_conjugate(self) -> 'ConvexFunction':
        """
        Computes the convex conjugate of the current function.

        The convex conjugate of a function f is the function defined as:
        f*(y) = sup_x(y * x - f(x)), where sup represents the supremum.

        This is useful for computing the weighted infimal convolution.

        :return: A new instance of `ConvexFunction` representing the convex conjugate.
        """
        pass