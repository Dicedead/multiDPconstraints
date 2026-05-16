from base.definitions import *
from base.tradeoff_function import TradeOffFunction, NormalRotation
from multi_dp_mixture.dp_functions import MultiEpsDeltaTradeoff
from scipy.optimize import brentq


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


def __recover_x_coordinates(u, f):
    u_array = np.atleast_1d(u)
    x_array = np.zeros_like(u_array, dtype=float)
    u_scaled = u_array * np.sqrt(2)
    for i, target in enumerate(u_scaled):
        x_array[i] = brentq(lambda x: x - f(x) - target, 0.0, f.fixed_point()+1e-6)
    return x_array


def __lower_approx_bisection(
        f,
        n,
        tol=1e-9,
        max_iter=1000,
        ternary_search_max_iter=70
):
    """
    Finds the partition P = (u0, u1, ..., un) that maximizes the
    Midpoint Riemann Sum S(P) for a convex, non-increasing function f,
    passing by its normal rotation.
    """

    g = NormalRotation(f)
    z = -f(0) / np.sqrt(2)

    mids = [i * z / n for i in range(n + 1)]
    mids = np.array(sorted(mids))

    def local_objective(xi, x_prev, x_next):
        """Calculates the sum of the two rectangles affected by xi."""
        area1 = (xi - x_prev) * g((xi + x_prev) / 2)
        area2 = (x_next - xi) * g((x_next + xi) / 2)
        return area1 + area2

    # 3. Coordinate Descent Loop
    for iteration in range(max_iter):
        prev_mids = list(mids)

        # Optimize each internal point x_1, ..., x_{n-1}
        for i in range(1, n):
            # Ternary search to find the best x[i] between x[i-1] and x[i+1]
            l, r = mids[i - 1], mids[i + 1]
            for _ in range(ternary_search_max_iter):  # High precision
                m1 = l + (r - l) / 3
                m2 = r - (r - l) / 3
                if local_objective(m1, mids[i - 1], mids[i + 1]) < local_objective(m2, mids[i - 1], mids[i + 1]):
                    l = m1
                else:
                    r = m2
            mids[i] = (l + r) / 2

        # Check for convergence (max shift in any point)
        diff = max(abs(mids[i] - prev_mids[i]) for i in range(n + 1))
        if diff < tol:
            break

    # mids_rotated = np.zeros_like(mids)
    # mids_rotated[1:] = __recover_x_coordinates(mids[1:], f)
    # return mids_rotated
    return mids

def multi_dp_approx_below(f: TradeOffFunction, n: int) -> MultiEpsDeltaTradeoff:
    """
    Compute the best L1 approximation of the trade-off function f from below by an n-DP trade-off function.

    :param f: trade-off function to approximate:
    :type f: TradeOffFunction

    :param n: number of DP degrees of freedom.
    :type n: int

    :return: MultiEpsDeltaTradeoff object representing the approximation.
    :rtype: MultiEpsDeltaTradeoff
    """
    u = __lower_approx_bisection(f, n)
    u = np.array(sorted(np.unique(u)))

    slopes = np.zeros(len(u) - 1)
    offsets = np.zeros(len(u) - 1)
    g = NormalRotation(f)

    i = 1
    while i < len(u):
        avg = (u[i] + u[i - 1]) / 2
        slope = g.subgradient_at(avg)
        offset = g(avg) - slope * avg

        slopes[i-1], offsets[i-1] = NormalRotation.slope_offset_rotation_inversion(slope, offset)
        i += 1

    return MultiEpsDeltaTradeoff.from_slopes_and_offsets(slopes, offsets)


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

    slopes = np.zeros(len(p) - 1)
    offsets = np.zeros(len(p) - 1)

    i = 0
    while i < len(p) - 1:
        slope = (f(p[i + 1]) - f(p[i])) / (p[i + 1] - p[i])
        offset = f(p[i]) - slope * p[i]

        slopes[i] = slope
        offsets[i] = offset
        i += 1

    return MultiEpsDeltaTradeoff.from_slopes_and_offsets(slopes, offsets)



