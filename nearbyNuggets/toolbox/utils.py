import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits


# Function used to fit completeness vs. mag in Martin+2016 (PAndAS)
def completenessFunction(mags, mag50, rho, a):
    denom = 1 + np.exp((mags - mag50) / rho)
    return a / denom


# Solve function used to fit completeness vs. mag in Martin+2016 (PAndAS)
# mag = mag50 + rho ln(A/comp - 1)
def completenessMag(comp, mag50, rho, a):
    return mag50 + rho * np.log(a / comp - 1)


def distToDmod(dist):
    """
    Convert distance to distance modulus.

    Parameters
    ----------
    distance  : in pc

    Returns
    -------
    dmod  : distance modulus

    """
    return 5.0*np.log10(dist)-5.0


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


def median_pos(sc_inp):
    """
    Recalculate the median position of a sample of stars.

    Parameters
    ----------
    sc_inp : `SkyCoord` object of input stars

    Returns
    -------
    sc_out : median RA, Dec of input stars (as SkyCoord object)

    """
    ra_out = np.median(sc_inp.ra.value)
    dec_out = np.median(sc_inp.dec.value)

    sc_out = SkyCoord(ra_out*u.deg, dec_out*u.deg)

    return(sc_out)


def rh_arcmin_to_pc(rh_arcmin, dist):
    """ Convert angular size in arcmin to parsecs at a given distance

    Parameters
    ----------
    rh_arcmin : `float`
        Size in arcminutes
    dist : `float`
        Distance to object in kpc
    """
    rh_pc = (dist*1e3)*((rh_arcmin*u.arcmin).to(u.rad))
    return rh_pc


def whichPatch(position, patchlist):
    """ Figure out which patch a given position appears in

    Parameters
    ----------
    position : `SkyCoord`
        Astropy SkyCoord object with the position to look up
    patchlist : `string`
        Path to the FITS file with RA, Dec limits of each patch
    """

    hdulist = fits.open(patchlist)
    patch_dat = hdulist[1].data

    ratmp = position.ra.value
    dectmp = position.dec.value
    # NOTE: ramin/ramax are backwards?
    matchpatch = np.where((patch_dat['ramax'] < ratmp) & (patch_dat['ramin'] > ratmp) &
                          (patch_dat['decmin'] < dectmp) & (patch_dat['decmax'] > dectmp))

    return patch_dat[matchpatch]['patch'][0]
