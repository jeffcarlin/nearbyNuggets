import numpy as np


# Function used to fit completeness vs. mag in Martin+2016 (PAndAS)
def completenessFunction(mags, mag50, rho, a):
    denom = 1 + np.exp((mags - mag50) / rho)
    return a / denom


# Solve function used to fit completeness vs. mag in Martin+2016 (PAndAS)
# mag = mag50 + rho ln(A/comp - 1)
def completenessMag(comp, mag50, rho, a):
    return mag50 + rho * np.log(a / comp - 1)


def median_interval(data, alpha=0.32):
    """
    Median including Bayesian credible interval.

    Parameters
    ----------
    data  : posterior samples
    alpha : 1 - confidence interval

    Returns
    -------
    [med,[lo, hi]] : median, lower, and upper percentiles

    """
    q = [100 * alpha / 2., 50, 100 * (1 - alpha / 2.)]
    lo, med, hi = np.percentile(data, q)
    return [med, [lo, hi]]
