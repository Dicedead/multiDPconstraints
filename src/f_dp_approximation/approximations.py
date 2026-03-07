from base.definitions import *
from base.tradeoff_function import TradeOffFunction
from multi_dp_mixture.dp_functions import MultiEpsDeltaTradeoff


def __upper_approx_golden_section(
        f: TradeOffFunction,
        n: int,
        tol=1e-6,
        max_iter=1000
) -> Array:
    """
    Finds the partition of [0, c] into n intervals minimizing the area
    of the n piece piecewise linear upper bound of a trade-off function f.
    Uses the golden section algorithm.

    :param f: The trade-off function to approximate from above.
    :type f: TradeOffFunction

    :param n: The number of intervals to partition [0, c].
    :type n: int

    :param tol: The tolerance for convergence.
    :type tol: float

    :param max_iter: The maximum number of iterations for the algorithm.
    :type max_iter: int

    :return: An array of length n containing the partition points.
    :rtype: Array
    """

    c = f.fixed_point()
    p = np.linspace(0, c, n + 1)

    phi = (np.sqrt(5) + 1) / 2
    resphi = 2 - phi

    for iteration in range(max_iter):
        max_change = 0.0

        for k in range(1, n):
            x_prev = p[k - 1]
            x_next = p[k + 1]

            def trapezoid_area(x_k):
                return (x_k - x_prev) * (f(x_prev) + f(x_k)) + \
                    (x_next - x_k) * (f(x_k) + f(x_next))

            a, b = x_prev, x_next
            x1 = a + resphi * (b - a)
            x2 = b - resphi * (b - a)

            f1 = trapezoid_area(x1)
            f2 = trapezoid_area(x2)

            while (b - a) > tol * 0.1: # Tighter tolerance for the 1D search
                if f1 < f2:
                    b = x2
                    x2 = x1
                    f2 = f1
                    x1 = a + resphi * (b - a)
                    f1 = trapezoid_area(x1)
                else:
                    a = x1
                    x1 = x2
                    f1 = f2
                    x2 = b - resphi * (b - a)
                    f2 = trapezoid_area(x2)

            x_new = (a + b) / 2

            # Track maximum change for convergence checking
            change = np.abs(p[k] - x_new)
            if change > max_change:
                max_change = change

            p[k] = x_new

        if max_change < tol:
            break

    return p

def multi_dp_approx_above(f: TradeOffFunction, n: int) -> MultiEpsDeltaTradeoff:
    """
    Compute the best L1 approximation of the trade-off function f from above by an n-DP trade-off function.

    :param f: trade-off function to approximate:
    :type f: TradeOffFunction

    :param n: number of DP degrees of freedom.
    :type n: int

    :return: MultiEpsDeltaTradeoff object representing the approximation.
    :rtype: MultiEpsDeltaTradeoff
    """
    p = __upper_approx_golden_section(f, n)
    p = np.array(sorted(np.unique(p)))

    slopes = np.zeros(len(p)-1)
    offsets = np.zeros(len(p)-1)

    i = 0
    while i < len(p)-1:
        slope = (f(p[i+1]) - f(p[i]))/(p[i+1] - p[i])
        offset = f(p[i]) - slope * p[i]

        slopes[i] = slope
        offsets[i] = offset
        i += 1

    return MultiEpsDeltaTradeoff.from_slopes_and_offsets(slopes, offsets)


