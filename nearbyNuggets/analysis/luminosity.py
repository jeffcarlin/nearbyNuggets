import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table
from astropy.io import fits
from nearbyNuggets.toolbox.utils import completenessFunction
from scipy.interpolate import interp1d


def totLum(sc_inp, cat, rh, gmagbins, gerr_medians, imagbins, ierr_medians,
           cmdsel_flag, star_flag, dmod):
    # Estimate the total luminosity:
    """ Calculate the structural parameters of a candidate dwarf using
        a maximum likelihood method + MCMC
    Parameters
    ----------
    sc_inp : `SkyCoord`
        central coordinate of field to select
    cat : ``
        input catalog of data (prefiltered to only the region of interest)
    rh : `float`
        measured half-light radius of (candidate) dwarf, in arcminutes
    gmagbins :
        centers of g magnitude bins in which median errors were measured
    gerr_medians :
        median gmag error in each magnitude bin
    imagbins :
        centers of i magnitude bins in which median errors were measured
    ierr_medians :
        median imag error in each magnitude bin
    cmdsel_flag : `` "rgbbox"
        array of boolean flag values to select the sample of interest
        (e.g., RGB stars)
    star_flag : `` "isstar"
        array of boolean flag values to select point sources (i.e., stars)
    dmod  : `float`
        distance modulus

    TO ADD:
    nsamples ???
    completeness coeffs
    """

    sc_all = SkyCoord(ra=cat.dat['ra']*u.radian, dec=cat.dat['dec']*u.radian, frame='icrs')
    ra = sc_all.ra.value
    dec = sc_all.dec.value
    # Select stars within measured r_h:
    spatial_msk_tmp = sc_all.separation(sc_inp) < (rh*u.arcmin)

    ra_cand_rh = ra[spatial_msk_tmp & star_flag & cmdsel_flag]
    dec_cand_rh = dec[spatial_msk_tmp & star_flag & cmdsel_flag]

    # calculate luminosity of all these stars:
    # From http://mips.as.arizona.edu/~cnaw/sun.html, M_g,Sun = 5.07
    #    (in AB mags, HSC g filter)
    # sum the g-band fluxes
    mg_sun = 5.07
    g0_bgfix = cat.dat['g0_bgfix']
    absg = g0_bgfix - dmod
    # absg = g0_bgfix - dmod
    lumg = 10.0**((absg-mg_sun)/(-2.5))
    tot_lumg = np.sum(lumg[spatial_msk_tmp & star_flag & cmdsel_flag])
    mvtot = mg_sun-2.5*np.log10(tot_lumg)

    # Now use a Dartmouth lumin. function to figure out how much light we're missing:

#     Output file includes a header similar to that of the input
#     isochrone, followed by sequential lines of LF data. Including:
#     1. bin number
#     2. absolute magnitude of the bin in the chosen filter
#     3. log_10 cumulative number of stars, total number is nstars
#        (default is nstars=10^5, to change, edit nstars below)
#     4. log_10 number of stars in each bin

# Theoretical (Dartmouth) luminosity function:
# lf_n,lf_gmag,lf_logn,lf_logdn = np.loadtxt('a13000fehm2p0afe0p4_slope_m0p5.lf',unpack=True)
    lf_path = '/Users/jcarlin/Dropbox/local_volume_dwarfs/ngc2403/dwarfsearch/a10gyr_feh_m2p0_Chabrier2001LF.lf'
    lf_n, lf_gmag, lf_logn, lf_logdn = np.loadtxt(lf_path, unpack=True)
# lf_n,lf_gmag,lf_logn,lf_logdn = np.loadtxt('a13000fehm2p0afe0p4_salpeter.lf',unpack=True)

    n_per_bin = 10.0**lf_logdn
    g_lf = lf_gmag+dmod
    lumg_lf = 10.0**((lf_gmag-mg_sun)/(-2.5))

# count up the luminosity in the RGB box, correcting for missing stars using completeness:

# Completeness fits from ASTs:
# popt_g_andxxii
# Out[83]: array([26.06899642,  0.83617971,  1.00940663])

# popt_i_andxxii
# Out[84]: array([26.32611259,  0.58590768,  0.98660214])

    completeness_coeffs_g = [26.06899642, 0.83617971, 1.00940663]

    lf_bins = np.arange(22, 29, 0.1)

# "Raw" observed luminosity function of RGB candidates:
    g_lf_obs = np.histogram(g0_bgfix[spatial_msk_tmp & star_flag & cmdsel_flag],
                            bins=lf_bins)

# Get the completeness values in each bin:
    g_lf_completeness = completenessFunction(g_lf_obs[1][:-1], *completeness_coeffs_g)
    g_lf_completeness_corr = g_lf_obs[0]/g_lf_completeness

# Number of candidates is the total of the completeness-corrected bin counts:
    ncand = np.int(np.round(np.sum(g_lf_completeness_corr)))

# Factor to apply to RGB counts to correct for incompleteness:
    completeness_factor = np.sum(g_lf_completeness_corr)/np.sum(g_lf_obs[:-1])

# Read in the fake dwarf data:
    # hhh = fits.open('/Users/jcarlin/Dropbox/local_volume_dwarfs/dwarfsample_code/dwarfsample/n2403/n2403_dwarf_forLuminosity_age10gyr_feh_m2p0.fits')
    # dat_fake = hhh[1].data

    # fake_g0 = dat_fake['g']
    # fake_i0 = dat_fake['i']

    fakePath = '/Users/jcarlin/Dropbox/local_volume_dwarfs/dwarfsample_code/dwarfsample/n2403/n2403_dwarf_forLuminosity_age10gyr_feh_m2p0.fits'
    fakeCat = pcat.photomCat(fakePath)

    gerr_interp = interp1d(gmagbins, gerr_medians, kind='cubic', bounds_error=False, fill_value='extrapolate')
    ierr_interp = interp1d(imagbins, ierr_medians, kind='cubic', bounds_error=False, fill_value='extrapolate')

    # Convolve the fake dwarf with errors from data
    rng = np.random.default_rng()
    rand_norm_g = rng.normal(0.0, 1.0, len(fakeCat['g']))
    # fake_err_g = rand_norm_g*gerr_interp(fake_g0)
    rand_norm_i = rng.normal(0.0, 1.0, len(fakeCat['i']))
    # fake_err_i = rand_norm_i*ierr_interp(fake_i0)

    # fake_g = fake_g0+fake_err_g
    # fake_i = fake_i0+fake_err_i
    # fake_gi = fake_g-fake_i

    fakeCat.dat.add_column(fakeCat.dat['g']*gerr_interp[fakeCat.dat['g']], name='fake_err_g')
    fakeCat.dat.add_column(fakeCat.dat['i']*gerr_interp[fakeCat.dat['i']], name='fake_err_i')
    fakeCat.dat.add_column(fakeCat.dat['g']+fakeCat.dat['fake_err_g']], name='fake_g')
    fakeCat.dat.add_column(fakeCat.dat['i']+fakeCat.dat['fake_err_i']], name='fake_i')

    # fakeCat.calcColumn(q1='g', q2='i', colname='fake_err_g', )
    # fakeCat.calcColumn(q1='gmag_bgfix', q2='imag_bgfix', colname='gi_bgfix', op='-')

    fakeCat.calcColumn(q1='fake_g', q2='fake_i', colname='fake_gi', op='-')
    fakeCat.rgbBoxFilter(colorColumn='fake_gi', magColumn='fake_i')

    # AT THIS POINT I NEED TO FIGURE OUT HOW TO GENERALIZE THE RGBFILTER ROUTINE TO
    #   ALLOW ME TO APPLY IT TO AN ARBITRARY CATALOG. ALTERNATIVELY, I COULD GENERALIZE
    #   THE CATALOG INITIALIZATION...

    # Select stars from the fake using the RGB box
# eval_left_fake = slope_left*(fake_gi)+int_left
# eval_right_fake = slope_right*(fake_gi)+int_right
# rgbbox_fake = (fake_i > eval_left_fake) & (fake_i < eval_right_fake) & (fake_i > brightlim_rgb) & (fake_i < faintlim_rgb)

# fake_g_rgb = fake_g[rgbbox_fake]
# fake_i_rgb = fake_i[rgbbox_fake]

    fake_g_rgb = fakeCat.dat[fakeCat.rgbFlag]['fake_g']
    fake_i_rgb = fakeCat.dat[fakeCat.rgbFlag]['fake_i']

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

    lf_marigo = ascii.read('/Users/jcarlin/Dropbox/local_volume_dwarfs/dwarfsample_code/dwarfsample/n2403/lf_marigo_age10gyr_feh_m2p0_HSC.dat', header_start=12)

#n_per_bin = 10.0**lf_logdn
#g_lf = lf_gmag+dmod
#lumg_lf=10.0**((lf_gmag-mg_sun)/(-2.5))

    g_lf = lf_marigo['magbinc']+dmod
    n_per_bin = lf_marigo['gmag']
    lumg_lf = 10.0**((g_lf-mg_sun)/(-2.5))

    # number of stars between min/max g mag of RGB box in LF:
    lfmsk = (g_lf > np.min(fake_g_rgb)) & (g_lf < np.max(fake_g_rgb))

    # Correct for the fraction of luminosity that is below the limiting magnitude:
    totlum_lf_rgb = np.sum(n_per_bin[lfmsk]*lumg_lf[lfmsk])
    totlum_lf_all = np.sum(n_per_bin*lumg_lf)

    lum_corr_fact = totlum_lf_all/totlum_lf_rgb

    mtot_g_corr = magref_samp_g-2.5*np.log10(np.array(lum_samp_g)*lum_corr_fact)-dmod
    mtot_i_corr = magref_samp_i-2.5*np.log10(np.array(lum_samp_i)*lum_corr_fact)-dmod

    print('M_g: ', np.median(mtot_g_corr), '+/-', np.std(mtot_g_corr), '; M_i: ',
          np.median(mtot_i_corr), '+/-', np.std(mtot_i_corr))

# Use the conversion from Appendix A of Komiyama+2018, ApJ, 853, 29:
# V = g_hsc - 0.371*(gi_hsc)-0.068

    mv = np.median(mtot_g_corr)-0.371*(np.median(mtot_g_corr)-np.median(mtot_i_corr))-0.068
    print('M_V: ',mv)

#%%

# Mean surface brightness within r_half:
# rh_fit_arcsec = (rh_fit*u.arcmin).to(u.arcsec).value
# rh_err_arcsec = (rh_err*u.arcmin).to(u.arcsec).value

# sb_andxxii = calc_sb(mv, ellip_fit, dmod, rh_fit_arcsec)
# sb_andxxii_min = calc_sb(mv-np.std(mtot_g_corr), ellip_fit-ellip_err, dmod-dmod_err, rh_fit_arcsec-rh_err_arcsec)
# sb_andxxii_max = calc_sb(mv+np.std(mtot_g_corr), ellip_fit+ellip_err, dmod+dmod_err, rh_fit_arcsec+rh_err_arcsec)

# print(sb_andxxii, sb_andxxii_max, sb_andxxii_min)
# print('SB(And XXII): ',sb_andxxii,'+',sb_andxxii_max-sb_andxxii,', -',sb_andxxii-sb_andxxii_min)
