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
sys.path.append('/Users/jcarlin/nearbyNuggets/')

import nearbyNuggets.inputCatalogs.photomCat as pcat

#rcat = rw.photomCat('/Users/jcarlin/Dropbox/local_volume_dwarfs/ngc2403//catalogs_dec2020/fake_dwarfs/cat_jan2021_fakedwarfs_NGC2403.fits.gz')
#tmpcat = rcat.dat[330:430]
#tmpcat.write('testcat.fits')

rcat = pcat.photomCat('testcat_fake573.fits')
rcat.calcColumn(q1='gmag_bgfix', q2='imag_bgfix', colname='gi_bgfix', op='-')

# rcat.rgbBoxFilter(colorColumn='gi_bgfix', magColumn='imag_bgfix')

rcat.extinctionCorr(units=u.radian)

rcat.calcColumn(q1='gmag_bgfix', q2='a_g', colname='g0_bgfix', op='-')
rcat.calcColumn(q1='imag_bgfix', q2='a_i', colname='i0_bgfix', op='-')
rcat.calcColumn(q1='g0_bgfix', q2='i0_bgfix', colname='gi0_bgfix', op='-')

rcat.rgbBoxFilter(colorColumn='gi0_bgfix', magColumn='i0_bgfix')
rcat.starGalFlag()

rcat.dat[rcat.isstarFlag & rcat.rgbFlag]

rcat.medianMagErrors(magColumn='g0_bgfix', errColumn='gerr')
magbins = rcat.magErrBins
gerr_medians = rcat.magErrMedians

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
ra_bins, dec_bins = makeBins(sc_dat, binsize=0.1*u.arcmin)
binCounts = densityMap(sc_dat[rcat.isstarFlag & rcat.rgbFlag], ra_bins, dec_bins)

# 2D histogram of all detected objects in the catalog (to be used for masking places with no data)
binCounts_all = densityMap(sc_dat, ra_bins, dec_bins)

#%%
nsig_bins, std_bins, bg_bins, bgareafrac_bins = densityBinStats(binCounts, ra_bins, dec_bins, binned_counts_all=binCounts_all)

nsig_bins_img = binsToImage(nsig_bins, binCounts)
bg_bins_img = binsToImage(bg_bins, binCounts)

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
from nearbyNuggets.analysis.diagnosticPlots import plotcand6

sep_cen = sc_dat.separation(n2403cen)
cen = (sep_cen.arcmin < 15.0)

plotcand6(sc_bins[dwarfcands][2], rcat, 0, nsig_bins[dwarfcands][2],
          cen, rcat.rgbFlag,
          rcat.isstarFlag, binsize=1.5, recalc_cen=True, savefig=False,
          name_append='', overlay_pts=False)

