from base.definitions import *
from base.tradeoff_function import TradeOffFunction, NormalRotation
from multi_dp_mixture.dp_functions import MultiEpsDeltaTradeoff
from multi_dp_mixture.piecewise_affine import keep_useful_lines


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

            while (b - a) > tol * 0.1:
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

            change = np.abs(p[k] - x_new)
            if change > max_change:
                max_change = change

            p[k] = x_new

        if max_change < tol:
            break

    return p


def __lower_approx_midpoint_sum(f: TradeOffFunction, n: int, tol = SMALL_TOL):
    """
    Finds the partition P = (u0, u1, ..., un) that maximizes the
    Midpoint Riemann Sum S(P) for a convex, non-increasing function f,
    passing by its normal rotation.
    """

    g = f.normal_rotation()
    z = -f(0) / np.sqrt(2)
    if type(z) is not float:
        z = z.item() # ugly but works so far

    if n == 1:
        return np.array([z, 0.0])

    def objective(u_inner):
        u = np.concatenate(([z], u_inner, [0.]))
        widths = np.diff(u)
        midpoints = (u[:-1] + u[1:]) / 2.
        return -1 * np.sum([widths[idx] * g(mid) for idx, mid in enumerate(midpoints)])

    # initial guess
    u0 = np.linspace(z, 0, n + 1)[1:-1]

    def constraint(u_inner):
        u = np.concatenate(([z], u_inner, [0.]))
        return np.diff(u) # enforce u_i <= u_i+1

    cons = {'type': 'ineq', 'fun': constraint}
    bounds = [(z, 0.0) for _ in range(n - 1)]

    result = spo.minimize(
        objective,
        u0,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'disp': False, 'ftol': tol}
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    optimal_partition = np.concatenate(([z], result.x, [0.]))
    return optimal_partition

def l1_multi_dp_approx_below(f: TradeOffFunction, n: int) -> MultiEpsDeltaTradeoff:
    """
    Compute the best L1 approximation of the trade-off function f from below by an n-DP trade-off function.

    :param f: trade-off function to approximate:
    :type f: TradeOffFunction

    :param n: number of DP degrees of freedom.
    :type n: int

    :return: MultiEpsDeltaTradeoff object representing the approximation.
    :rtype: MultiEpsDeltaTradeoff
    """
    assert n >= 1

    u = __lower_approx_midpoint_sum(f, n)
    u = np.array(sorted(np.unique(u)))

    slopes = np.zeros(len(u) - 1)
    offsets = np.zeros(len(u) - 1)
    g = f.normal_rotation()

    i = 1
    while i < len(u):
        avg = (u[i] + u[i - 1]) / 2
        slope = g.subgradient(avg)
        offset = g(avg) - slope * avg

        slopes[i-1], offsets[i-1] = NormalRotation.slope_offset_rotation_inversion(slope, offset)
        i += 1
    return MultiEpsDeltaTradeoff.from_slopes_and_offsets(*keep_useful_lines(slopes, offsets))


def l1_multi_dp_approx_above(f: TradeOffFunction, n: int) -> MultiEpsDeltaTradeoff:
    """
    Compute the best L1 approximation of the trade-off function f from above by an n-DP trade-off function.

    :param f: trade-off function to approximate:
    :type f: TradeOffFunction

    :param n: number of DP degrees of freedom.
    :type n: int

    :return: MultiEpsDeltaTradeoff object representing the approximation.
    :rtype: MultiEpsDeltaTradeoff
    """
    assert n >= 1

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

    return MultiEpsDeltaTradeoff.from_slopes_and_offsets(*keep_useful_lines(slopes, offsets))


def linf_multi_dp_approx_above(f: TradeOffFunction, n: int):
    """
    Compute the best Linfinity approximation of the trade-off function f from above by an n-DP trade-off function.

    :param f: trade-off function to approximate:
    :type f: TradeOffFunction

    :param n: number of DP degrees of freedom.
    :type n: int

    :return: MultiEpsDeltaTradeoff object representing the approximation.
    :rtype: MultiEpsDeltaTradeoff
    """
    c_fixed = f.fixed_point()

    def obj_above(vars):
        return vars[-1]

    def constraints_above(vars):
        P = np.concatenate(([0], vars[:-1], [c_fixed]))
        max_err = vars[-1]
        cons = []

        for i in range(len(P) - 1):
            cons.append(P[i + 1] - P[i] - TOL) # t_i+1 < t_i constraint

        for i in range(1, n + 1):
            x_prev, x_curr = P[i - 1], P[i]

            if x_curr - x_prev < SMALL_TOL:
                cons.append(max_err)
                continue

            y_prev, y_curr = f(x_prev), f(x_curr)
            m = (y_curr - y_prev) / (x_curr - x_prev)
            c = y_prev - m * x_prev

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = spo.minimize_scalar(
                    lambda x: f(x) - (m * x + c),
                    bounds=(x_prev, x_curr),
                    method='bounded'
                )
            cons.append(max_err - (-res.fun))

        return np.array(cons)

    P_init = np.linspace(0, c_fixed, n + 1)[1:-1]
    P_init_in = np.append(P_init, 0.1)

    res_above = spo.minimize(
        obj_above, P_init_in, method='SLSQP',
        constraints={'type': 'ineq', 'fun': constraints_above},
        options={'disp': False, 'ftol': 1e-8, 'maxiter': 500}
    )
    P_star = np.concatenate(([0], res_above.x[:-1], [c_fixed]))

    slopes, intercepts = [], []
    for i in range(1, n + 1):
        m = (f(P_star[i]) - f(P_star[i - 1])) / (P_star[i] - P_star[i - 1])
        c = f(P_star[i - 1]) - m * P_star[i - 1]
        slopes.append(m)
        intercepts.append(c)

    return MultiEpsDeltaTradeoff.from_slopes_and_offsets(*keep_useful_lines(slopes, intercepts))


def linf_multi_dp_approx_below(f: TradeOffFunction, n: int):
    """
    Compute the best Linfinity approximation of the trade-off function f from below by an n-DP trade-off function.

    :param f: trade-off function to approximate:
    :type f: TradeOffFunction

    :param n: number of DP degrees of freedom.
    :type n: int

    :return: MultiEpsDeltaTradeoff object representing the approximation.
    :rtype: MultiEpsDeltaTradeoff
    """
    x_star = f.fixed_point()
    f_prime_func = f.subgradient

    def obj_below(vars):
        return vars[-1]

    def constraints_below(vars):
        P = vars[:-1]
        max_err = vars[-1]
        cons = []

        cons.append(P[0] - TOL)
        for i in range(n - 1):
            cons.append(P[i + 1] - P[i] - TOL)
        cons.append(x_star - P[-1] - TOL)

        subgrads = np.array([f_prime_func(t) for t in P])
        d_offsets = np.array([f(t) - subgrads[i] * t for i, t in enumerate(P)])

        cons.append(max_err - (f(0) - d_offsets[0]))

        for i in range(n - 1):
            if abs(subgrads[i] - subgrads[i + 1]) < SMALL_TOL:
                x_i = P[i]
            else:
                x_i = (d_offsets[i + 1] - d_offsets[i]) / (subgrads[i] - subgrads[i + 1])
            x_i = np.clip(x_i, 0, x_star)
            cons.append(max_err - (f(x_i) - (subgrads[i] * x_i + d_offsets[i])))

        x_n = d_offsets[-1] / (1 - subgrads[-1]) if abs(subgrads[-1] - 1.0) > SMALL_TOL else 0
        cons.append(max_err - (f(x_n) - x_n))

        return np.array(cons)

    P_init = np.linspace(0, x_star, n + 2)[1:-1]
    P_init_in = np.append(P_init, 0.1)

    res_below = spo.minimize(
        obj_below, P_init_in, method='SLSQP',
        constraints={'type': 'ineq', 'fun': constraints_below},
        options={'disp': False, 'ftol': 1e-8, 'maxiter': 500}
    )

    P_star = res_below.x[:-1]
    slopes = [f_prime_func(t) for t in P_star]
    intercepts = [f(t) - m * t for t, m in zip(P_star, slopes)]

    return MultiEpsDeltaTradeoff.from_slopes_and_offsets(*keep_useful_lines(slopes, intercepts))
