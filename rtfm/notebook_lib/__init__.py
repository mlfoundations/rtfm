import math

import scipy


def clopper_pearson(acc, n, alpha=0.05):
    """Estimate the confidence interval for a sampled Bernoulli random
    variable.
    `x` is the number of successes and `n` is the number trials (x <=
    n). `alpha` is the confidence level (i.e., the true probability is
    inside the confidence interval with probability 1-alpha). The
    function returns a `(low, high)` pair of numbers indicating the
    interval on the probability.
    """
    x = int(acc * n)
    b = scipy.stats.beta.ppf
    lo = b(alpha / 2, x, n - x + 1)
    hi = b(1 - alpha / 2, x + 1, n - x)
    return 0.0 if math.isnan(lo) else lo, 1.0 if math.isnan(hi) else hi


def plot_confidence_interval(
    ax,
    xy,
    x_int=None,
    y_int=None,
    color="#2187bb",
    alpha=0.5,
    width=0.001,
    linewidth=1,
    **kwargs
):
    """Plot a single (x,y) point with (possibly different) confidence
    intervals for the x- and y-values."""
    x, y = xy
    if x_int:
        xmax = max(x_int)
        xmin = min(x_int)
        ax.plot(
            x_int, [y, y], color=color, alpha=alpha, linewidth=linewidth
        )  # horizontal bars
        ax.plot(
            [xmin, xmin], [y - width, y + width], color=color, alpha=alpha
        )  # left CI "tail"
        ax.plot(
            [xmax, xmax], [y - width, y + width], color=color, alpha=alpha
        )  # right CI "tail"
    if y_int:
        ymax = max(y_int)
        ymin = min(y_int)
        ax.plot(
            [x, x], y_int, color=color, alpha=alpha, linewidth=linewidth
        )  # vertical bars
        ax.plot(
            [x - width, x + width], [ymax, ymax], color=color, alpha=alpha
        )  # top CI "tail"
        ax.plot(
            [x - width, x + width], [ymin, ymin], color=color, alpha=alpha
        )  # bottom CI "tail"

    ax.plot(x, y, color=color, alpha=alpha, **kwargs)
    return ax
