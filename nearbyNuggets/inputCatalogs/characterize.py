import numpy as np
import scipy.stats as stats


def findNsigma(mags, errors, nsig=10.0, magbinsize=0.02, minmag=15.0, maxmag=29.0):
    # Note: errors can be either mag. errors or fractional flux errors
    errbins = np.arange(minmag, maxmag, magbinsize)
    errbin_medians = stats.binned_statistic(mags, errors, statistic='median', bins=errbins)
    minloc_err = np.nanargmin(np.abs(errbin_medians[0]-(1/nsig)))
    return errbin_medians[1][minloc_err]


def getMedianMagErrors(cat, magbinsize=0.2, minmag=17.0, maxmag=29.0,
                       magColumn='i0_bgfix', errColumn='ierr'):
    # Calculate the median mag errors as a function of magnitude:
    magbins = np.arange(minmag, maxmag, magbinsize)
    mederr = []
    magbin_centers = []
    for i in range(np.size(magbins)):
        inbin = (cat.dat[magColumn] < magbins[i]+magbinsize) & (cat.dat[magColumn] > magbins[i])
        mederr.append(np.nanmedian(cat.dat[inbin][errColumn]))
        magbin_centers.append(magbins[i]+magbinsize/2)

    magErrBins = magbin_centers
    magErrMedians = mederr

    return magErrBins, magErrMedians
