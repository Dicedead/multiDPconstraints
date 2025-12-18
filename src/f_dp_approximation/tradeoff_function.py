from definitions import *

class SmoothTradeOffFunction(TradeOffFunction, ABC):

    @abstractmethod
    def __call__(self, x: Array) -> Array:
        pass

    @abstractmethod
    def fixed_point(self) -> float:
        pass

    @abstractmethod
    def derivative_at(self, x: Array) -> Array:
        pass

    @abstractmethod
    def second_derivative_at(self, x: Array) -> Array:
        pass

    def rotation_change(self, u: float) -> Callable[[Array], Array]:
        return lambda x: (x - self(x))/np.sqrt(2) - u

    def rotation_change_deriv(self) -> Callable[[Array], Array]:
        return lambda x: (1 - self.derivative_at(x))/np.sqrt(2)

    def rotation_change_second_deriv(self) -> Callable[[Array], Array]:
        return lambda x: - self.second_derivative_at(x)/np.sqrt(2)

class NormalRotation:

    def __init__(self, f: SmoothTradeOffFunction):
        self._f = f

    def invert_u(self, u: float) -> float:
        return spo.root_scalar(
            f=self._f.rotation_change(u),
            bracket=(0, self._f.fixed_point()),
            x0=self._f.fixed_point()/2,
            fprime=self._f.rotation_change_deriv(),
            fprime2=self._f.rotation_change_second_deriv()
        )

    def call(self, u: Array | float, x_u: Array | float = None) -> Array | float:
        if x_u is None:
            x_u = [self.invert_u(ui) for ui in u] if type(u) is Array else self.invert_u(u)
        return (x_u + self._f(x_u))/np.sqrt(2)

    def __call__(self, u: Array | float) -> Array | float:
        return self.call(u)

    def derivative_at(self, u: Array | float, x_u: Array | float = None) -> Array | float:
        if x_u is None:
            x_u = [self.invert_u(ui) for ui in u] if type(u) is Array else self.invert_u(u)
        f_prime_x_u = self._f.derivative_at(x_u)
        return (1+f_prime_x_u)/(1-f_prime_x_u)

    def second_derivative_at(self, u: Array | float, x_u: Array | float = None) -> Array | float:
        if x_u is None:
            x_u = [self.invert_u(ui) for ui in u] if type(u) is Array else self.invert_u(u)
        f_prime_x_u = self._f.derivative_at(x_u)
        f_second_x_u = self._f.second_derivative_at(x_u)
        return (2 * np.sqrt(2) * f_second_x_u)/((1-f_prime_x_u) ** 3)