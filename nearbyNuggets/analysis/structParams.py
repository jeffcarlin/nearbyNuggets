import numpy as np
import emcee
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
from scipy.optimize import minimize
from nearbyNuggets.toolbox.utils import median_interval


def ln_likelihood(params, ra, dec, rad):

    ra0real = params[0]
    dec0real = params[1]
    ellipreal = params[2]
    thetareal = params[3]
    rhreal = params[4]
    sigmabreal = params[5]

    # kwargs are starting guesses
    nobj = len(ra)
    # sigmabreal=2.53 # from plummer or exponential
    # term1 = (1.0*nobj-144.0*np.pi*sigmabreal)/(2.0*np.pi*rhreal*rhreal*nreal*gamma(2.0*nreal)*(1.0D - ellipreal))

    # rad = 10.0  # radius of field in arcminb
    term1 = (2.8224 * (1.0 * nobj - (rad**2) * np.pi * sigmabreal)) / \
            (2.0 * np.pi * rhreal * rhreal * (1.0 - ellipreal))

    xi = (ra - ra0real) * np.cos(np.median(dec) * np.pi / 180.0)
    yi = (dec - dec0real)

    term2a = (1.0 / (1.0 - ellipreal)) * (xi * np.cos(thetareal * np.pi / 180.0) -
                                          yi * np.sin(thetareal * np.pi / 180.0))

    term2b = (xi * np.sin(thetareal * np.pi / 180.0) + yi * np.cos(thetareal * np.pi / 180.0))

    ri = np.sqrt(term2a**2.0 + term2b**2.0) * 60.0

    # term2c = np.exp(-1.0*((ri/rhreal)**(1./nreal)))
    term2c = np.exp(-1.68 * (ri / rhreal))

    lnP = np.sum(np.log(sigmabreal + term1 * term2c))
    min_lnp = -1.0 * lnP

    # if not np.isfinite(min_lnp):
    #    return -np.inf

    return min_lnp


######################################################


def ln_like(params, ra, dec):

    ra0real = params[0]
    dec0real = params[1]
    ellipreal = params[2]
    thetareal = params[3]
    rhreal = params[4]
    sigmabreal = params[5]

    # kwargs are starting guesses
    nobj = len(ra)
    # sigmabreal=2.53 # from plummer or exponential
    # term1 = (1.0*nobj-144.0*np.pi*sigmabreal)/(2.0*np.pi*rhreal*rhreal*nreal*gamma(2.0*nreal)*(1.0D - ellipreal))

    rad = 5.0  # radius of field in arcminb
    term1 = (2.8224*(1.0*nobj-(rad**2)*np.pi*sigmabreal))/(2.0*np.pi*rhreal*rhreal*(1.0 - ellipreal))

    if term1 < 0:
        return -np.inf

    # import pdb; pdb.set_trace()
    xi = (ra - ra0real)*np.cos(np.median(dec)*np.pi/180.0)
    yi = (dec - dec0real)

    term2a = (1.0/(1.0 - ellipreal))*(xi*np.cos(thetareal*np.pi/180.0) - yi*np.sin(thetareal*np.pi/180.0))

    term2b = (xi*np.sin(thetareal*np.pi/180.0) + yi*np.cos(thetareal*np.pi/180.0))

    ri = np.sqrt(term2a**2.0 + term2b**2.0)*60.0

    # term2c = np.exp(-1.0*((ri/rhreal)**(1./nreal)))
    term2c = np.exp(-1.68*(ri/rhreal))

    lnP = np.sum(np.log(sigmabreal + term1*term2c))

    # if not np.isfinite(lnP):
    #    return -np.inf

    return lnP


def lnprior(params):
    """ The log-prior. Add whatever you want here...

    Parameters
    ----------
    theta : model parameters

    Returns
    -------
    lnprior : log-prior
    """
    # PRIORS FOR EACH FITTED PARAMETER
    # TO REMOVE ANY PARAMETER FROM THE FIT, REMOVE IT FROM THE LINE BELOW AND ELIMINATE THE CONSTRAINT FOR IT
    # rich1,lon1,lat1,ext1,ell1,pa1,rich2,lon2,lat2,ext2,ell2,pa2,theta[0],theta[1],theta[2],theta[3],theta[4],theta[5],theta[6],theta[7],theta[8],theta[9],theta[10],theta[11]
    ra0, dec0, ellip0, theta0, rh0, sigma0 = params
    if not (0 < ra0 < 360):
        return np.inf
    if not (-90 < dec0 < 90):
        return np.inf
    if not (0 < ellip0 < 1):
        return np.inf
    if not (0 < theta0 < 360):
        return np.inf
    if not (0.05 < rh0 < 6):
        return np.inf
    if not (0.1 < sigma0 < 30):
        return np.inf
    return 0


def lnprob(theta, x, y):
    """ The log-probability = lnlike + lnprob

    Parameters
    ----------
    theta : the model parameter vector
    x     : x-coord of the data
    y     : y-coord of the data

    Returns
    -------
    lnprob : log-probability
    """
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_like(theta, x, y)


def calc_sb(M_V, e, dmod, rh):
    """ Calculate average surface brightness given absolute magnitude,
    ellipticity, distance modulus, and half-light radius.

    Parameters
    ----------
    M_V   : total V-band luminosity of the dwarf
    e     : ellipticity
    dmod  : distance modulus
    rh    : half-light radius (arcsec)

    Returns
    -------
    sb    : average surface brightness within the half-light radius
    """
    app_mag = dmod + M_V + 0.75  # the 0.75 = 2.5log(2.0) is because half the flux is within r_h
    a = rh
    b = a*(1-e)
    area = np.pi * a * b
    sb = app_mag + 2.5*np.log10(area)
    return sb


def mlStructParams(sc_inp, cat, rad,
                   cmdsel_flag, star_flag,
                   ellip0=0.5, theta0=90.0,
                   rh0=1.0, sigmab0=0.1,
                   racol='ra', deccol='dec', radec_units=u.radian):
    """ Calculate the structural parameters of a candidate dwarf using
        a maximum likelihood method

    Parameters
    ----------
    sc_inp : `SkyCoord`
        central coordinate of field to select
    cat : ``
        input catalog of data (prefiltered to only the region of interest)
    rad : `float`
        radius of field in arcminutes
    cmdsel_flag : `` "rgbbox"
        array of boolean flag values to select the sample of interest
        (e.g., RGB stars)
    star_flag : `` "isstar"
        array of boolean flag values to select point sources (i.e., stars)
    ellip0 : `float`
        starting guess at ellipticity (default: 0.5)
    theta0 : `float`
        starting guess at position angle (in degrees E of N; default: 90.0)
    rh0 : `float`
        starting guess at half-light radius (arcmin; default: 1.0)
    sigmab0 : `float`
        starting guess at background surface density (stars/arcmin^2)
        default: 0.1
    """

    sc_all = SkyCoord(ra=cat[racol]*radec_units, dec=cat[deccol]*radec_units, frame='icrs')
    ra = sc_all.ra.value
    dec = sc_all.dec.value
    spatial_msk_tmp = sc_all.separation(sc_inp) < (rad*u.arcmin)

    ra_cand = ra[spatial_msk_tmp & star_flag & cmdsel_flag]
    dec_cand = dec[spatial_msk_tmp & star_flag & cmdsel_flag]

    # if cen0 is None:
    #     cen0 = SkyCoord(ra=np.median(cat.dat['ra'])*u.radian,
    #                 dec=np.median(cat.dat['dec'])*u.radian, frame='icrs')

    # sc_all = SkyCoord(ra=cat.dat['ra']*u.radian, dec=cat.dat['dec']*u.radian, frame='icrs')
    # ra_cand = sc_all.ra.value
    # dec_cand = sc_all.dec.value

    par0 = [sc_inp.ra.value, sc_inp.dec.value, ellip0, theta0, rh0, sigmab0]
    nll = lambda *args: ln_likelihood(*args)
    initial = np.array(par0 + 1e-4 * np.random.randn(len(par0)))
    soln = minimize(nll, initial, args=(ra_cand, dec_cand, rad), method='Nelder-Mead')
    ra_ml, dec_ml, ellip_ml, theta_ml, rh_ml, sigmab_ml = soln.x

    # print('len(ra_cand): ', len(ra_cand))
    # print('solution: ', soln.x)

    # Estimate the number of stars in the dwarf:
    # N_* = N_tot - pi*r^2*sigma_b
    nstars_ml = len(ra_cand) - np.pi * (rad**2) * sigmab_ml
    # print('nstars_ml: ', nstars_ml)

    ml_params = Table([[ra_ml], [dec_ml], [ellip_ml], [theta_ml], [rh_ml], [sigmab_ml], [nstars_ml]],
                      names=['ra', 'dec', 'ellip', 'theta', 'rhalf', 'sigmaB', 'nstars'])

    return ml_params


def mcmcStructParams(sc_inp, cat, rad,
                     cmdsel_flag, star_flag,
                     ellip0=0.5, theta0=90.0,
                     rh0=1.0, sigmab0=0.1, nwalkers=100, nthreads=16,
                     nsamples=10000, nburn=1000,
                     racol='ra', deccol='dec', radec_units=u.radian):
    """ Calculate the structural parameters of a candidate dwarf using
        a maximum likelihood method + MCMC
    Parameters
    ----------
    sc_inp : `SkyCoord`
        central coordinate of field to select
    cat : ``
        input catalog of data (prefiltered to only the region of interest)
    rad : `float`
        radius of field in arcminutes
    cmdsel_flag : `` "rgbbox"
        array of boolean flag values to select the sample of interest
        (e.g., RGB stars)
    star_flag : `` "isstar"
        array of boolean flag values to select point sources (i.e., stars)
    ellip0 : `float`
        starting guess at ellipticity (default: 0.5)
    theta0 : `float`
        starting guess at position angle (in degrees E of N; default: 90.0)
    rh0 : `float`
        starting guess at half-light radius (arcmin; default: 1.0)
    sigmab0 : `float`
        starting guess at background surface density (stars/arcmin^2)
        default: 0.1
    nwalkers : `int`
        number of mcmc nwalkers (default: 100)
    nthreads : `int`
        number of mcmc threads (default: 16)
    nsamples : `int`
        number of mcmc samples (default: 10000)
    nburn : `int`
        number of mcmc burn-in iterations (default: 1000)
    """

    sc_all = SkyCoord(ra=cat[racol]*radec_units, dec=cat[deccol]*radec_units, frame='icrs')
    ra = sc_all.ra.value
    dec = sc_all.dec.value
    spatial_msk_tmp = sc_all.separation(sc_inp) < (rad*u.arcmin)

    ra_cand = ra[spatial_msk_tmp & star_flag & cmdsel_flag]
    dec_cand = dec[spatial_msk_tmp & star_flag & cmdsel_flag]

# Initialize and run the mcmc
    print("Running mcmc...")
    # if cen0 is None:
    #     cen0 = SkyCoord(ra=np.median(cat.dat['ra'])*u.radian,
    #                     dec=np.median(cat.dat['dec'])*u.radian, frame='icrs')

    # sc_all = SkyCoord(ra=cat.dat['ra']*u.radian, dec=cat.dat['dec']*u.radian, frame='icrs')
    # ra_cand = sc_all.ra.value
    # dec_cand = sc_all.dec.value

    par0 = [sc_inp.ra.value, sc_inp.dec.value, ellip0, theta0, rh0, sigmab0]

    ndim = len(par0)
    pos = [par0 + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=(ra_cand, dec_cand),
                                    threads=nthreads)
#                                kwargs=(par0), threads=nthreads)
    sampler.run_mcmc(pos, nsamples)

    samples = sampler.chain[:, nburn:, :].reshape((-1, ndim))

    # MEDIAN VALUES AND +/-1 SIGMA FOR EACH FIT PARAMETER
    ra_fit, [ra_fit_min, ra_fit_max] = median_interval(samples[:, 0])
    dec_fit, [dec_fit_min, dec_fit_max] = median_interval(samples[:, 1])
    ellip_fit, [ellip_fit_min, ellip_fit_max] = median_interval(samples[:, 2])
    theta_fit, [theta_fit_min, theta_fit_max] = median_interval(samples[:, 3])
    rh_fit, [rh_fit_min, rh_fit_max] = median_interval(samples[:, 4])
    sigmab_fit, [sigmab_fit_min, sigmab_fit_max] = median_interval(samples[:, 5])

    ra_plus = ra_fit_max-ra_fit
    ra_minus = ra_fit-ra_fit_min
    dec_plus = dec_fit_max-dec_fit
    dec_minus = dec_fit-dec_fit_min
    ra_err = np.mean(np.abs([ra_plus, ra_minus]))*u.deg*np.cos(dec_fit*u.deg)
    dec_err = np.mean(np.abs([dec_plus, dec_minus]))*u.deg

    ellip_err = np.mean(np.abs([ellip_fit_max-ellip_fit, ellip_fit-ellip_fit_min]))
    theta_err = np.mean(np.abs([theta_fit_max-theta_fit, theta_fit-theta_fit_min]))
    rh_err = np.mean(np.abs([rh_fit_max-rh_fit, rh_fit-rh_fit_min]))
    sigmab_err = np.mean(np.abs([sigmab_fit_max-sigmab_fit, sigmab_fit-sigmab_fit_min]))
    nstars_fit = len(ra_cand) - np.pi*(rad**2)*sigmab_fit
    nstars_fit_min = len(ra_cand) - np.pi*(rad**2)*sigmab_fit_max
    nstars_fit_max = len(ra_cand) - np.pi*(rad**2)*sigmab_fit_min

    # print('n_stars: ',nstars_fit, '+',nstars_fit_max-nstars_fit,' -',nstars_fit-nstars_fit_min)

    nstars_err = np.mean((nstars_fit_max-nstars_fit, nstars_fit-nstars_fit_min))

    mcmc_params = Table([[ra_fit], [ra_err], [dec_fit], [dec_err],
                         [ellip_fit], [ellip_err], [theta_fit], [theta_err],
                         [rh_fit], [rh_err], [sigmab_fit], [sigmab_err],
                         [nstars_fit], [nstars_err]],
                        names=['ra', 'ra_err', 'dec', 'dec_err',
                               'ellip', 'ellip_err', 'theta', 'theta_err',
                               'rhalf', 'rhalf_err', 'sigmaB', 'sigmaB_err',
                               'nstars', 'nstars_err'])

    return mcmc_params
