import numpy as np
import emcee
import corner
from astropy.io import fits, ascii
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize


#def ln_likelihood(ra, dec, ra0real=None, dec0real=None, ellipreal=None, thetareal=None, rhreal=None, nreal=None, sigmabreal=None):
#    if ra0real is None:
#        ra0real = np.median(ra)
#    if dec0real is None:
#        dec0real = np.median(dec)
#    if ellipreal is None:
#        ellipreal = 0.4
#    if thetareal is None:
#        thetareal = 0.0
#    if rhreal is None:
#        rhreal = 0.6 # in kpc
#    if nreal is None:
#        nreal = 1.0
#    if sigmabreal is None:
#        sigmabreal = 1.51

######################################################


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

    #if not np.isfinite(min_lnp):
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

    #if not np.isfinite(lnP):
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
    app_mag = dmod + M_V + 0.75  # the 0.75 = 2.5log(2.0) is because half the flux is within r_h
    a = rh
    b = a*(1-e)
    area = np.pi * a * b
    sb = app_mag + 2.5*np.log10(area)
    return sb

#%%
# Read in some data:
#hdulist = fits.open('/Users/jcarlin/Dropbox/local_volume_dwarfs/ngc2403/catalogs_dec2020/fake_dwarfs/cat_jan2021_fakedwarfs_faint_NGC2403_withEBV.fits.gz')
#hdulist = fits.open('/Users/jcarlin/Dropbox/local_volume_dwarfs/ngc2403/catalogs_dec2020/cat_N2403_dec2020_with_ebv.fits.gz')
hdulist = fits.open('cat_m33_HSC_feb2021_extras.fits')
dat = hdulist[1].data


#%%

g0_bgfix = dat['g0']
i0_bgfix = dat['i0']
gi0_bgfix = g0_bgfix - i0_bgfix
gerr = dat['gerr']
ierr = dat['ierr']

sc_all = SkyCoord(dat['ra']*u.rad, dat['dec']*u.rad, distance=(dat['ra']*0.0+853.0)*u.kpc, frame='icrs')
andxxii_cen = SkyCoord(ra=21.9167*u.deg,dec=28.0903*u.deg,frame='icrs') # from Simbad
sep = sc_all.separation(andxxii_cen)
rhalf = 1.5 # 1.8 # rough guess -- r_h = 0.9' from Martin+2016
cen = (sep.arcmin < rhalf)
tab = Table(dat)

dist = 861.0 # Kristy's distance in kpc
dmod = 5.0*np.log10(861.0*1e3)-5.0
disterr = 52.0
dmod_plus = 5.0*np.log10((861.0+disterr)*1e3)-5.0
dmod_minus = 5.0*np.log10((861.0-disterr)*1e3)-5.0
dmod_err = np.mean(np.abs([dmod_plus-dmod, dmod-dmod_minus]))

# Annulus of equal area (r_outer = sqrt(r_h^2 + r_inner^2)):
rinner = rhalf*2.0
router = np.sqrt(rhalf**2 + rinner**2)
ann = (sep.arcmin < router) & (sep.arcmin > rinner)
rinner2 = rhalf*4.0
router2 = np.sqrt(rhalf**2 + rinner2**2)
ann2 = (sep.arcmin < router2) & (sep.arcmin > rinner2)

isstar = dat['isstar_flag']
rgbbox = dat['rgb_flag']

#%%

# Create an RGB selection box:
corners = [[0.9,21.3],[0.4,25.0],[1.0,25.0],[1.8,21.3]]

slope_left = (corners[1][1]-corners[0][1])/(corners[1][0]-corners[0][0])
int_left = corners[1][1] - slope_left*corners[1][0]
slope_right = (corners[3][1]-corners[2][1])/(corners[3][0]-corners[2][0])
int_right = corners[3][1] - slope_right*corners[3][0]

eval_left = slope_left*(gi0_bgfix)+int_left
eval_right = slope_right*(gi0_bgfix)+int_right

faintlim_rgb = 25.0 # faintest stars to keep
brightlim_rgb = 21.3
# rgbbox = (i0_bgfix > eval_left) & (i0_bgfix < eval_right) & (i0_bgfix > brightlim_rgb) & (i0_bgfix < faintlim_rgb)


#%%
# Calculate the median mag errors as a function of magnitude:
magbinsize = 0.2
magbins = np.arange(17, 29, magbinsize)

mederr_g = []
mederr_i = []
magbin_centers = []
for i in range(np.size(magbins)):
    inbin_g = (g0_bgfix < magbins[i]+magbinsize) & (g0_bgfix > magbins[i])
    inbin_i = (i0_bgfix < magbins[i]+magbinsize) & (i0_bgfix > magbins[i])
    mederr_g.append(np.median(gerr[inbin_g]))
    mederr_i.append(np.median(ierr[inbin_i]))
    magbin_centers.append(magbins[i]+magbinsize/2)

gerr_interp = interp1d(magbin_centers, mederr_g, kind='cubic', bounds_error=False, fill_value='extrapolate')
ierr_interp = interp1d(magbin_centers, mederr_i, kind='cubic', bounds_error=False, fill_value='extrapolate')

fakemags = np.linspace(17.0, 29.0, num=121, endpoint=True)

plt.plot(i0_bgfix, ierr, 'k.', alpha=0.005)
plt.plot(fakemags, ierr_interp(fakemags), color='Red')
#plt.plot(g0_bgfix, gerr, 'k.', alpha=0.005)
#plt.plot(fakemags, gerr_interp(fakemags), color='Red')
plt.xlim(16.5,29.5)
plt.ylim(-0.1, 0.5)
plt.show()

#print('Mag bin center, median g error, median i error')
#for i in range(np.size(magbin_centers)):
#    print(magbin_centers[i], mederr_g[i], mederr_i[i])

#%%

#sc_cand = SkyCoord(ra=117.44866*u.deg,dec=66.41991*u.deg,frame='icrs') # faint fake gx 1765
#sc_cand = SkyCoord(ra=113.656*u.deg,dec=66.701*u.deg,frame='icrs') # dwarf cand 7
# sc_cand = SkyCoord(ra=114.12865*u.deg,dec=64.71958*u.deg,frame='icrs') # faint fake gx 2360
sc_cand = andxxii_cen

r_cand = 6.0
near_cand = (sc_all.separation(sc_cand).arcmin < r_cand)

ra_cand = sc_all[near_cand & rgbbox & isstar].ra.value
dec_cand = sc_all[near_cand & rgbbox & isstar].dec.value

#%%
# Maximum likelihood:

######################################################


def mlStructParams(cat, rad):
    """ Calculate the structural parameters of a candidate dwarf using
        a maximum likelihood method

    Parameters
    ----------
    cat : ``
        input catalog of data (prefiltered to only the region of interest)
    rad : `float`
        radius of field in arcminutes
    """

    sc_all = SkyCoord(ra=cat.dat['ra']*u.radian, dec=cat.dat['dec']*u.radian, frame='icrs')
    ra_cand = sc_all.ra.value
    dec_cand = sc_all.dec.value

    #np.random.seed(42)
    # par0: [ra0, dec0, ellip0, theta0, rh0, sigmab0]
    #par0=[np.median(ra_cand),np.median(dec_cand),0.1,0.0,1.6,5.0]
    par0 = [np.median(ra_cand), np.median(dec_cand), 0.5, 90.0, 1.0, 0.1]
    nll = lambda *args: ln_likelihood(*args)
    initial = np.array(par0 + 1e-4 * np.random.randn(len(par0)))
    soln = minimize(nll, initial, args=(ra_cand, dec_cand, rad), method='Nelder-Mead')
    # , tol=1e-7, options={'maxiter':1000})
    # soln = minimize(nll, initial, args=(ra_cand,dec_cand), method='Nelder-Mead', bounds=((0,360),(90,90),(0,1),(0,360),(0.05,6.0),(0.1,15.0)), options={'maxiter':1000})
    ra_ml, dec_ml, ellip_ml, theta_ml, rh_ml, sigmab_ml = soln.x
    # ra_err_ml, dec_err_ml, ellip_err_ml, theta_err_ml, rh_err_ml, sigmab_err_ml = np.sqrt(np.diag(soln.hess_inv))

    # Estimate the number of stars in the dwarf:
    # N_* = N_tot - pi*r^2*sigma_b
    nstars_ml = len(ra_cand) - np.pi * (r_cand**2) * sigmab_ml

    ml_params = Table([ra_ml, dec_ml, ellip_ml, theta_ml, rh_ml, sigmab_ml, nstars_ml],
                      names=['ra', 'dec', 'ellip', 'theta', 'rhalf', 'sigmaB', 'nstars'])

    return ml_params


######################################################

#%%
# Initialize and run the mcmc
print("Running mcmc...")
par0=[np.median(ra_cand), np.median(dec_cand), 0.4, 90.0, 1.0, 0.1]

ndim, nwalkers = len(par0), 100
nthreads,nsamples = 16, 10000
nburn = 1000
pos = [par0 + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers,ndim,lnprob,args=(ra_cand,dec_cand),
                                threads=nthreads)
#                                kwargs=(par0), threads=nthreads)
sampler.run_mcmc(pos,nsamples)

samples = sampler.chain[:, nburn:, :].reshape((-1, ndim))

#MEDIAN VALUES AND +/-1 SIGMA FOR EACH FIT PARAMETER
ra_fit,[ra_fit_min,ra_fit_max] = median_interval(samples[:,0])
dec_fit,[dec_fit_min,dec_fit_max] = median_interval(samples[:,1])
ellip_fit,[ellip_fit_min,ellip_fit_max] = median_interval(samples[:,2])
theta_fit,[theta_fit_min,theta_fit_max] = median_interval(samples[:,3])
rh_fit,[rh_fit_min,rh_fit_max] = median_interval(samples[:,4])
sigmab_fit,[sigmab_fit_min, sigmab_fit_max] = median_interval(samples[:,5])

#%%
scmean = SkyCoord(ra=ra_fit*u.deg, dec=dec_fit*u.deg)
ra_plus = ra_fit_max-ra_fit
ra_minus = ra_fit-ra_fit_min
dec_plus = dec_fit_max-dec_fit
dec_minus = dec_fit-dec_fit_min
ra_err = np.mean(np.abs([ra_plus, ra_minus]))*u.deg*np.cos(dec_fit*u.deg)
dec_err = np.mean(np.abs([dec_plus, dec_minus]))*u.deg
rh_plus = rh_fit_max-rh_fit
rh_minus = rh_fit-rh_fit_min
rh_err = np.mean(np.abs([rh_plus, rh_minus]))

rh_pc = (dist*1e3)*((rh_fit*u.arcmin).to(u.rad))
rh_plus_pc = ((dist+52.0)*1e3)*((rh_fit_max*u.arcmin).to(u.rad))
rh_minus_pc = ((dist-52.0)*1e3)*((rh_fit_min*u.arcmin).to(u.rad))
rh_err_pc = np.mean(np.abs([rh_plus_pc.value-rh_pc.value, rh_pc.value-rh_minus_pc.value]))

ellip_err = np.mean(np.abs([ellip_fit_max-ellip_fit, ellip_fit-ellip_fit_min]))

print('RA: ', scmean.ra.hms, ra_err.to(u.arcsec))
print('Dec: ', scmean.dec.dms, dec_err.to(u.arcsec))
print('RA: ',ra_fit,' +',ra_fit_max-ra_fit,' -', ra_fit-ra_fit_min)
print('Dec: ',dec_fit,' +',dec_fit_max-dec_fit,' -', dec_fit-dec_fit_min)
print('ellip: ',ellip_fit,' +',ellip_fit_max-ellip_fit,' -', ellip_fit-ellip_fit_min)
print('theta: ',theta_fit,' +',theta_fit_max-theta_fit,' -', theta_fit-theta_fit_min)
print('rh: ',rh_fit,' +',rh_fit_max-rh_fit,' -', rh_fit-rh_fit_min,' arcmin')
print('rh: ',rh_pc.value,' +',rh_plus_pc.value-rh_pc.value,' -', rh_pc.value-rh_minus_pc.value,' pc')

print('sigma_b: ',sigmab_fit,' +',sigmab_fit_max-sigmab_fit,' -', sigmab_fit-sigmab_fit_min)

nstars_fit = len(ra_cand) - np.pi*(r_cand**2)*sigmab_fit
nstars_fit_min = len(ra_cand) - np.pi*(r_cand**2)*sigmab_fit_max
nstars_fit_max = len(ra_cand) - np.pi*(r_cand**2)*sigmab_fit_min

print('n_stars: ',nstars_fit, '+',nstars_fit_max-nstars_fit,' -',nstars_fit-nstars_fit_min)

nstars_err = np.mean((nstars_fit_max-nstars_fit, nstars_fit-nstars_fit_min))
#%%
fig = corner.corner(samples, labels=["ra", "dec", "ellip", "theta", "r$_{\mathrm{h}}$ (arcmin)", "$\Sigma_b$"])

#%%
# Now estimate the total luminosity:

# Select stars within measured r_h:
rh_cand = rh_fit
inrh_cand = (sc_all.separation(sc_cand).arcmin < rh_cand)

ra_cand_rh = sc_all[inrh_cand & rgbbox & isstar].ra.value
dec_cand_rh = sc_all[inrh_cand & rgbbox & isstar].dec.value

# calculate luminosity of all these stars:
# From http://mips.as.arizona.edu/~cnaw/sun.html, M_g,Sun = 5.07 (in AB mags, HSC g filter)
# sum the g-band fluxes
mg_sun = 5.07
absg = g0_bgfix - dmod
lumg = 10.0**((absg-mg_sun)/(-2.5))
tot_lumg = np.sum(lumg[inrh_cand & rgbbox & isstar])
mvtot = mg_sun-2.5*np.log10(tot_lumg)

#%%
# Now use a Dartmouth lumin. function to figure out how much light we're missing:

#c     Output file includes a header similar to that of the input         c
#c     isochrone, followed by sequential lines of LF data. Including:     c
#c     1. bin number                                                      c
#c     2. absolute magnitude of the bin in the chosen filter              c
#c     3. log_10 cumulative number of stars, total number is nstars       c
#c        (default is nstars=10^5, to change, edit nstars below)          c
#c     4. log_10 number of stars in each bin                              c

# Theoretical (Dartmouth) luminosity function:
#lf_n,lf_gmag,lf_logn,lf_logdn = np.loadtxt('a13000fehm2p0afe0p4_slope_m0p5.lf',unpack=True)
lf_n,lf_gmag,lf_logn,lf_logdn = np.loadtxt('/Users/jcarlin/Dropbox/local_volume_dwarfs/ngc2403/dwarfsearch/a10gyr_feh_m2p0_Chabrier2001LF.lf',unpack=True)
#lf_n,lf_gmag,lf_logn,lf_logdn = np.loadtxt('a13000fehm2p0afe0p4_salpeter.lf',unpack=True)

n_per_bin = 10.0**lf_logdn
g_lf = lf_gmag+dmod
lumg_lf=10.0**((lf_gmag-mg_sun)/(-2.5))

#%%
# count up the luminosity in the RGB box, correcting for missing stars using completeness:

# Completeness fits from ASTs:
# popt_g_andxxii
# Out[83]: array([26.06899642,  0.83617971,  1.00940663])

# popt_i_andxxii
# Out[84]: array([26.32611259,  0.58590768,  0.98660214])

completeness_coeffs_g = [26.06899642, 0.83617971,  1.00940663]

lf_bins = np.arange(22,29,0.1)

# "Raw" observed luminosity function of RGB candidates:
g_lf_obs = np.histogram(g0_bgfix[inrh_cand & rgbbox & isstar], bins=lf_bins)

# Get the completeness values in each bin:
g_lf_completeness = completenessFunction(g_lf_obs[1][:-1], *completeness_coeffs_g)

g_lf_completeness_corr = g_lf_obs[0]/g_lf_completeness

# Number of candidates is the total of the completeness-corrected bin counts:
ncand = np.int(np.round(np.sum(g_lf_completeness_corr)))

# Factor to apply to RGB counts to correct for incompleteness:
completeness_factor = np.sum(g_lf_completeness_corr)/np.sum(g_lf_obs[:-1])

#%%
# Read in the fake dwarf data:
hhh = fits.open('/Users/jcarlin/Dropbox/local_volume_dwarfs/dwarfsample_code/dwarfsample/n2403/n2403_dwarf_forLuminosity_age10gyr_feh_m2p0.fits')
dat_fake = hhh[1].data

fake_g0 = dat_fake['g']
fake_i0 = dat_fake['i']

### Convolve the fake dwarf with errors from data
rng = np.random.default_rng()
rand_norm_g = rng.normal(0.0, 1.0, len(fake_g0))
fake_err_g = rand_norm_g*gerr_interp(fake_g0)
rand_norm_i = rng.normal(0.0, 1.0, len(fake_i0))
fake_err_i = rand_norm_i*ierr_interp(fake_i0)

fake_g = fake_g0+fake_err_g
fake_i = fake_i0+fake_err_i
fake_gi = fake_g-fake_i

### Select stars from the fake using the RGB box
eval_left_fake = slope_left*(fake_gi)+int_left
eval_right_fake = slope_right*(fake_gi)+int_right
rgbbox_fake = (fake_i > eval_left_fake) & (fake_i < eval_right_fake) & (fake_i > brightlim_rgb) & (fake_i < faintlim_rgb)

fake_g_rgb = fake_g[rgbbox_fake]
fake_i_rgb = fake_i[rgbbox_fake]

'''
plt.plot(fake_gi, fake_i, 'k.', alpha=0.01)
plt.plot(fake_gi[rgbbox_fake], fake_i[rgbbox_fake], 'b.', alpha=0.1)
plt.plot(fake_g0-fake_i0, fake_i0, 'r.', alpha=0.1)
plt.ylim(30, 22)
plt.show()
'''
#%%
nsamples = 10000

lum_samp_g = []
magref_samp_g = []
lum_samp_i = []
magref_samp_i = []
nsamp = []

for samp in range(nsamples):
    # Apply a scatter in the number of stars of +/-nstars_err:
    #    The factor of 2 accounts for the stars being within the half-light radius
#    samp_ind = rng.choice(fake_g_rgb.shape[0], np.int(2*nstars_fit*completeness_factor+(nstars_err*rng.normal(0, 1))))
    samp_ind = rng.choice(fake_g_rgb.shape[0], np.int(nstars_fit*completeness_factor+(nstars_err*rng.normal(0, 1))))
    # samp_ind = rng.choice(fake_g_rgb.shape[0], np.int(nstars_fit+(nstars_err*rng.normal(0, 1))))
    nsamp.append(len(samp_ind))

    magref_g = fake_g_rgb[samp_ind[0]]
    magref_samp_g.append(magref_g)
    totlum_g = np.sum(10.0**((magref_g - fake_g_rgb[samp_ind[1:]])/2.5))
    lum_samp_g.append(totlum_g)

    magref_i = fake_i_rgb[samp_ind[0]]
    magref_samp_i.append(magref_i)
    totlum_i = np.sum(10.0**((magref_i - fake_i_rgb[samp_ind[1:]])/2.5))
    lum_samp_i.append(totlum_i)


# l = 10^((m-m0)/-2.5)
# magtot = m0-2.5*np.log10(l)

mtot_g = magref_samp_g-2.5*np.log10(lum_samp_g)-dmod
mtot_i = magref_samp_i-2.5*np.log10(lum_samp_i)-dmod

#%%
'''
# number of stars between min/max g mag of RGB box in LF:
lfmsk = (g_lf > np.min(fake_g[rgbbox_fake])) & (g_lf < np.max(fake_g[rgbbox_fake]))

# Correct for the fraction of luminosity that is below the limiting magnitude:
totlum_lf_rgb = np.sum(n_per_bin[lfmsk]*lumg_lf[lfmsk])
totlum_lf_all = np.sum(n_per_bin*lumg_lf)

lum_corr_fact = totlum_lf_all/totlum_lf_rgb

mtot_g_corr = magref_samp_g-2.5*np.log10(np.array(lum_samp_g)*lum_corr_fact)-dmod
mtot_i_corr = magref_samp_i-2.5*np.log10(np.array(lum_samp_i)*lum_corr_fact)-dmod

'''
# Next:
# 3. Calculate N realizations of the observed dwarf by sampling the fake dwarf N times, with the
#      number of stars in the sample equal to ncand+/-e_ncand.
# 4. Correct for the luminosity below the mag cutoff to get the "total" luminosity.
# QQQ: Do we need to shift up/down for distance uncertainty?

#%%
# plot a CMD of the candidate:

g0_cand_rgb = g0_bgfix[inrh_cand & rgbbox & isstar]
i0_cand_rgb = i0_bgfix[inrh_cand & rgbbox & isstar]
gi0_cand_rgb = g0_cand_rgb - i0_cand_rgb
g0_cand = g0_bgfix[inrh_cand & isstar]
i0_cand = i0_bgfix[inrh_cand & isstar]
gi0_cand = g0_cand - i0_cand

g0_cand0 = g0_bgfix[near_cand & isstar]
i0_cand0 = i0_bgfix[near_cand & isstar]
gi0_cand0 = g0_cand0-i0_cand0

plt.plot(gi0_cand, i0_cand, 'k.')
plt.plot(gi0_cand_rgb, i0_cand_rgb, 'r.')
plt.ylim(28, 22)
plt.show()

#%%
lf_marigo = ascii.read('/Users/jcarlin/Dropbox/local_volume_dwarfs/dwarfsample_code/dwarfsample/n2403/lf_marigo_age10gyr_feh_m2p0_HSC.dat', header_start=12)

#n_per_bin = 10.0**lf_logdn
#g_lf = lf_gmag+dmod
#lumg_lf=10.0**((lf_gmag-mg_sun)/(-2.5))

g_lf = lf_marigo['magbinc']+dmod
n_per_bin = lf_marigo['gmag']
lumg_lf = 10.0**((g_lf-mg_sun)/(-2.5))

# number of stars between min/max g mag of RGB box in LF:
lfmsk = (g_lf > np.min(fake_g[rgbbox_fake])) & (g_lf < np.max(fake_g[rgbbox_fake]))

# Correct for the fraction of luminosity that is below the limiting magnitude:
totlum_lf_rgb = np.sum(n_per_bin[lfmsk]*lumg_lf[lfmsk])
totlum_lf_all = np.sum(n_per_bin*lumg_lf)

lum_corr_fact = totlum_lf_all/totlum_lf_rgb

mtot_g_corr = magref_samp_g-2.5*np.log10(np.array(lum_samp_g)*lum_corr_fact)-dmod
mtot_i_corr = magref_samp_i-2.5*np.log10(np.array(lum_samp_i)*lum_corr_fact)-dmod

print('M_g: ',np.median(mtot_g_corr),'+/-',np.std(mtot_g_corr), '; M_i: ',np.median(mtot_i_corr),'+/-',np.std(mtot_i_corr))

# Use the conversion from Appendix A of Komiyama+2018, ApJ, 853, 29:
# V = g_hsc - 0.371*(gi_hsc)-0.068

mv = np.median(mtot_g_corr)-0.371*(np.median(mtot_g_corr)-np.median(mtot_i_corr))-0.068
print('M_V: ',mv)

#%%

# Mean surface brightness within r_half:
rh_fit_arcsec = (rh_fit*u.arcmin).to(u.arcsec).value
rh_err_arcsec = (rh_err*u.arcmin).to(u.arcsec).value

sb_andxxii = calc_sb(mv, ellip_fit, dmod, rh_fit_arcsec)
sb_andxxii_min = calc_sb(mv-np.std(mtot_g_corr), ellip_fit-ellip_err, dmod-dmod_err, rh_fit_arcsec-rh_err_arcsec)
sb_andxxii_max = calc_sb(mv+np.std(mtot_g_corr), ellip_fit+ellip_err, dmod+dmod_err, rh_fit_arcsec+rh_err_arcsec)

print(sb_andxxii, sb_andxxii_max, sb_andxxii_min)
print('SB(And XXII): ',sb_andxxii,'+',sb_andxxii_max-sb_andxxii,', -',sb_andxxii-sb_andxxii_min)

#%%
params = {
   'axes.labelsize': 20,
   'font.size': 20,
   'legend.fontsize': 14,
#   'xtick.labelsize': 16,
   'xtick.major.width': 3,
   'xtick.minor.width': 2,
   'xtick.major.size': 8,
   'xtick.minor.size': 5,
   'xtick.direction': 'in',
   'xtick.top': True,
   'lines.linewidth':3,
   'axes.linewidth':3,
   'axes.labelweight':3,
   'axes.titleweight':3,
   'ytick.major.width':3,
   'ytick.minor.width':2,
   'ytick.major.size': 8,
   'ytick.minor.size': 5,
   'ytick.direction': 'in',
   'ytick.right': True,
#   'ytick.labelsize': 20,
   'text.usetex': True,
   'text.latex.preamble': r'\boldmath',
   'figure.figsize': [9, 7],
   'figure.facecolor': 'White'
   }

plt.rcParams.update(params)
fig=plt.figure()

mv_values = mtot_g_corr-0.371*(mtot_g_corr-mtot_i_corr)-0.068

plt.hist(mtot_g_corr, bins=np.arange(-8.8, -4.7, 0.05), histtype='step', linewidth=4, color='Blue', linestyle='--', label='g')
plt.hist(mv_values, bins=np.arange(-8.8, -4.7, 0.05), histtype='step', linewidth=4, color='Black', label='V')
plt.hist(mtot_i_corr, bins=np.arange(-8.8, -4.7, 0.05), histtype='step', linewidth=4, color='Red', linestyle=':', label='i')
plt.xlabel(r'$M$')
plt.legend()
plt.ylabel('N')
plt.minorticks_on()

plt.savefig('m_abs_hist_22jun2021.png')
plt.show()
