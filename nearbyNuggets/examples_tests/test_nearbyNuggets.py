import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import sys
#from astropy.io import fits
from astropy.coordinates import SkyCoord
#from astropy.table import Table
#from dustmaps.sfd import SFDQuery
#import operator

# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
sys.path.append('/Users/jcarlin/repos/nearbyNuggets/')

import nearbyNuggets.inputCatalogs.photomCat as pcat

#rcat = rw.photomCat('/Users/jcarlin/Dropbox/local_volume_dwarfs/ngc2403//catalogs_dec2020/fake_dwarfs/cat_jan2021_fakedwarfs_NGC2403.fits.gz')
#tmpcat = rcat.dat[330:430]
#tmpcat.write('testcat.fits')

rcat = pcat.photomCat('testcat_fake573.fits', crapFilt=True)
rcat.calcColumn(q1='gmag_bgfix', q2='imag_bgfix', colname='gi_bgfix', op='-')

# rcat.rgbBoxFilter(colorColumn='gi_bgfix', magColumn='imag_bgfix')

rcat.setExtinctionCorr(units=u.radian)

rcat.calcColumn(q1='gmag_bgfix', q2='a_g', colname='g0_bgfix', op='-')
rcat.calcColumn(q1='imag_bgfix', q2='a_i', colname='i0_bgfix', op='-')
rcat.calcColumn(q1='g0_bgfix', q2='i0_bgfix', colname='gi0_bgfix', op='-')

rcat.rgbBoxFilter(colorColumn='gi0_bgfix', magColumn='i0_bgfix')
rcat.setStarGalFlag()

rcat.dat[rcat.isstarFlag & rcat.rgbFlag]

#rcat.medianMagErrors(magColumn='g0_bgfix', errColumn='gerr')
#magbins = rcat.magErrBins
#gerr_medians = rcat.magErrMedians

'''
#%%
from nearbyNuggets.mining.findNuggets import findNuggets

#dfind = findNuggets.testCat(rcat)

dens = findNuggets(rcat.dat[rcat.isstarFlag & rcat.rgbFlag])
meanSC = SkyCoord(ra=np.mean(rcat.dat['ra'])*u.radian, dec=np.mean(rcat.dat['dec']*u.radian), frame='icrs')
dens.densityMap(binsize=0.1*u.arcmin, meanSC = meanSC)

# 2D histogram of all detected objects in the catalog (to be used for masking places with no data)
dens_all = findNuggets(rcat.dat)
dens_all.densityMap(binsize=0.1*u.arcmin, meanSC = meanSC)
'''

#%%
from nearbyNuggets.mining.findNuggets import densityMap, makeBins, densityBinStats, binsToImage, binsToSkyCoord

sc_dat = SkyCoord(ra=rcat.dat['ra']*u.radian, dec=rcat.dat['dec']*u.radian, frame='icrs')

meanSC = SkyCoord(ra=np.mean(rcat.dat['ra'])*u.radian, dec=np.mean(rcat.dat['dec']*u.radian), frame='icrs')
ra_bins, dec_bins, ra_bin_centers, dec_bin_centers = makeBins(sc_dat, binsize=1.0*u.arcmin)
binCounts = densityMap(sc_dat[rcat.isstarFlag & rcat.rgbFlag], ra_bins, dec_bins)

# 2D histogram of all detected objects in the catalog (to be used for masking places with no data)
binCounts_all = densityMap(sc_dat, ra_bins, dec_bins)

#%%
nsig_bins, std_bins, bg_bins, bgareafrac_bins, counts_bins = densityBinStats(binCounts, ra_bins, dec_bins, 
                                                                             binned_counts_all=binCounts_all)

nsig_bins_img = binsToImage(nsig_bins, binCounts)
bg_bins_img = binsToImage(bg_bins, binCounts)
std_bins_img = binsToImage(std_bins, binCounts)
bincounts_img = binsToImage(counts_bins, binCounts)
bgareafrac_bins_img = binsToImage(bgareafrac_bins, binCounts)

#%%
# Replace NaN and Infinite values with -9.99:
nsig_bins[np.isnan(nsig_bins)] = -9.99
nsig_bins[np.isinf(nsig_bins)] = -9.99

# Get the indices sorted from highest significance to lowest:
sorted_bins = np.flip(np.argsort(nsig_bins))

sc_bins = binsToSkyCoord(ra_bins, dec_bins)
n2403cen = SkyCoord(ra='07h36m51.4s',dec='65d36m09s',frame='icrs') # from NED
ddo44cen = SkyCoord(ra='07h34m11.50s',dec='66d52m47.0s',frame='icrs') # from NED

# Pick out the ones greater than some threshold "sigma":
sigthresh = 5.0 
sep_bins_n2403 = sc_bins.separation(n2403cen)
sep_bins_ddo44 = sc_bins.separation(ddo44cen)
#dwarfcand_select = np.where((nsig_bins[sorted_bins] > sigthresh) & (bgareafrac_bins[sorted_bins] > 0.4) &\
dwarfcand_select = np.where((nsig_bins[sorted_bins] > sigthresh))# & (bgareafrac_bins[sorted_bins] > 0.2))# \
#                            & (sep_bins_n2403[sorted_bins].arcmin > 1.0) & (sep_bins_ddo44[sorted_bins].arcmin > 2.0))
dwarfcands = sorted_bins[dwarfcand_select]

#%%
from nearbyNuggets.mining.findNuggets import pickNuggets, getPeaks

dwarfcands2 = pickNuggets(counts_bins, nsig_bins, bgareafrac_bins, sigma_thresh=sigthresh)

peaks2 = getPeaks(counts_bins, binCounts, bg_bins, std_bins,
                  bgareafrac_bins, ra_bin_centers, dec_bin_centers, sigma_thresh=sigthresh)
#peaks2['ra'] = ra_bin_centers[peaks2['x_peak']]
#peaks2['dec'] = dec_bin_centers[peaks2['y_peak']]
#peaks2['sig'] = (peaks2['peak_value']-bg_bins_img[peaks2['y_peak'], peaks2['x_peak']])/std_bins_img[peaks2['y_peak'], peaks2['x_peak']]
#peaks2['n_in_bin'] = bincounts_img[peaks2['y_peak'], peaks2['x_peak']]
#peaks2['bg'] = bg_bins_img[peaks2['y_peak'], peaks2['x_peak']]
#peaks2['bgareafrac'] = bgareafrac_bins_img[peaks2['y_peak'], peaks2['x_peak']]
#peaks2['std'] = std_bins_img[peaks2['y_peak'], peaks2['x_peak']]

#%%
from nearbyNuggets.analysis.diagnosticPlots import plotcand6
sep_cen = sc_dat.separation(n2403cen)
cen = (sep_cen.arcmin < 15.0)

plotcand6(sc_bins[dwarfcands][2], rcat, 0, nsig_bins[dwarfcands][2],
          cen, rcat.rgbFlag,
          rcat.isstarFlag, binsize=1.5, recalc_cen=True, savefig=False,
          name_append='', overlay_pts=False)

#%%
import nearbyNuggets.analysis.structParams as structParams
# from nearbyNuggets.toolbox.utils import median_pos
import nearbyNuggets.toolbox.utils as nn_utils

rcat.setRadiusFlag(sc_bins[dwarfcands][2], 2.0)

sc_test = nn_utils.median_pos(sc_dat[rcat.radiusFlag & rcat.isstarFlag & rcat.rgbFlag])

struct_params = structParams.mlStructParams(sc_test, rcat, 2.0, rcat.rgbFlag,
                                            rcat.isstarFlag)

struct_params_mcmc = structParams.mcmcStructParams(sc_test, rcat, 2.0, rcat.rgbFlag,
                                                   rcat.isstarFlag, nsamples=1000, nburn=100)

#%%
import nearbyNuggets.analysis.luminosity as lumin
import nearbyNuggets.inputCatalogs.characterize as charData

gmagbins, gerr_medians = charData.getMedianMagErrors(rcat, magColumn='g0_bgfix', errColumn='gerr')
imagbins, ierr_medians = charData.getMedianMagErrors(rcat, magColumn='i0_bgfix', errColumn='ierr')

sc_cen_mcmc = SkyCoord(ra=struct_params_mcmc['ra']*u.deg, dec=struct_params_mcmc['dec']*u.deg, frame='icrs')

dmod = nn_utils.distToDmod(3.2e6)

# lumin.totLum(sc_cen_mcmc, rcat, 2*struct_params_mcmc['rhalf'], gmagbins, gerr_medians,
lumin.totLum(sc_cen_mcmc, rcat, 3.0, gmagbins, gerr_medians,
             imagbins, ierr_medians, rcat.rgbFlag, rcat.isstarFlag, dmod,
             struct_params_mcmc['nstars'], struct_params_mcmc['nstars_err'])



