import numpy as np


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
