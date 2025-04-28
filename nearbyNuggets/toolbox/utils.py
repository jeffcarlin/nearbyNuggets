import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits, ascii
from scipy import interpolate


def calcIsoDist(iso=[], dmod=27.5, data_table=[]):
    """
    Calculate the distance (in color/mag) of all objects from an input
    isochrone.

    Parameters
    ----------
    iso : `list` of two `np.array`s
        Isochrone to filter on -- blue band first, red band second

    dmod : `float`
        Distance modulus to shift the isochrone to

    data_table : `Astropy.Table`
        Astropy table object containing two columns, which _must_ be named
        "gmag" and "imag".
    Returns
    -------
    isodist : `np.array`
        Array of CMD distance (in mags) of each point from the supplied
        isochrone.
    """

    # Isochrone:
    abs_g_iso, abs_i_iso = iso
    gi_iso = abs_g_iso - abs_i_iso
    g_iso = abs_g_iso + dmod
    i_iso = abs_i_iso + dmod

    # sort them so they are monotonically increasing (or spline interp fails)
    i_iso, gi_iso = zip(*sorted(zip(i_iso, gi_iso)))

    # spline interpolate the isochrone
    tck = interpolate.splrep(i_iso, gi_iso, s=0)
    yy = np.arange(np.min(i_iso), np.max(i_iso), 0.01)
    xx = interpolate.splev(yy, tck, der=0)

    df = data_table.to_pandas()

    def minDist(df):
        col = df['gmag'] - df['imag']
        return np.min(np.sqrt((col-xx)**2 + (df['imag']-yy)**2))

    isodist = df.apply(minDist, axis=1).to_numpy()

    return isodist


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


def dmodToDist(dmod):
    """
    Convert distance modulus to distance.

    Parameters
    ----------
    dmod  : distance modulus

    Returns
    -------
    distance  : in pc

    """
    return 10.0**((dmod+5)/5.0)


def getIsochrone(mh=-2.0, msrgb=True,
                 isofile='/Users/jcarlin/Dropbox/isochrones/padova/panstarrs/output494507502139.dat'):
    """
    Get a 10 Gyr Padova isochrone of a given metallicity. Currently uses a
    hard-coded file of PanSTARRS isochrones for only one age, but could be
    updated in the future.

    Parameters
    ----------
    mh : `float`
        Requested [M/H]

    msrgb : `bool`
        Return only the main sequence and RGB?

    Returns
    -------
    iso : `np.array`

    """
    mhvals = [-2.19174, -2.1, -2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4,
              -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4,
              -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
    if mh not in mhvals:
        print('Supplied [M/H] value not in list. Use one of these: ', mhvals)
        return
    iso_all = ascii.read(isofile, header_start=13, data_start=14)

    if msrgb:
        iso_msrgb = (iso_all['label'] <= 3) & (iso_all['label'] > 0)
    else:
        iso_msrgb = (iso_all['label'] < 10)

    iso = (iso_all['MH'] == mh) & iso_msrgb
    return iso_all[iso]


def lf_powerlaw(mags, lf_inp, alpha=2.35):
    """
    Calculate a power-law luminosity function.

    Parameters
    ----------
    mags : magnitudes at which to return the luminosity function
    lf_inp : binned luminosity function from the data
    alpha : power-law slope (Salpeter: 2.35)

    Returns
    -------
    lf_out : powerlaw luminosity function normalized to the number
             counts at 25th magnitude

    # dN = Phi*m^(-alpha), where "m" is the magnitude
    # Want dN = value at mag25bin, so Phi = dN*m^(alpha) = dN*25^(alpha)
    """

    mag25bin = np.argmin(np.abs(mags - 25.0))
    phi_salpeter = np.sum(lf_inp[:mag25bin])  # *(25.0**(-1*alpha))
    lf_out = phi_salpeter*(10.0**((6.0*alpha/5.0) * (mags-25.0)))
    return lf_out


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


def mstar_from_absmag(m_v):
    """
    From an input satellite M_V, calculate the luminosity in solar units.
    Assuming M*/LV = 1.6 for dSphs (Woo+2008), infer the stellar mass.

    L_V/L_Sun = 10^[(M_V,Sun-M_V)/2.5]

    Parameters
    ----------
    m_v : `float`
        Luminosity (V-band absolute magnitude) of the satellite

    Returns
    -------
    mstars : `float`
        Stellar mass of the satellite in solar masses

    """

    mv_sun = 4.83

    m_to_l = 1.6

    lv = 10.0**((mv_sun-m_v)/2.5)
    mstars=m_to_l * lv

    return mstars


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
    return rh_pc.value


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
