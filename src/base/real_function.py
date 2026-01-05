from base.definitions import *


class RealFunction(Callable[[Array], Array], ABC):
    """
    Represents an abstract function of real parameter with its convex conjugate.
    """

    @abstractmethod
    def __call__(self, x: Array) -> Array:
        pass

    @abstractmethod
    def convex_conjugate(self) -> 'RealFunction':
        """
        Computes the convex conjugate of the current function.

        The convex conjugate of a function f is the function defined as:
        f*(y) = sup_x(y * x - f(x)), where sup represents the supremum.

        This is useful for computing the weighted infimal convolution.

        :return: A new instance of `RealFunction` representing the convex conjugate.
        """
        pass