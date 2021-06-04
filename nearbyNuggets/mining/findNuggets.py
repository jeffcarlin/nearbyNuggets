from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.stats import sigma_clipped_stats
import numpy as np

'''
class findNuggets:
    def __init__(self, catalog):
        # pass
        self.cat = catalog
        # print(self.photomCat.dat[self.photomCat.rgbFlag])

    # def testCat(self):
        # print(self.dat[self.rgbFlag])

    def densityMap(self, binsize=1.5*u.arcmin, meanSC=None):
        # meanSC: mean position relative to which you'd like to calculate
        #         separations -- SkyCoord object
        #
        # Input catalog should already be filtered (i.e., this function assumes
        #   you have already done quality cuts, RGB selection, star/galaxy separation, etc.)
        if not meanSC:
            # Default: NGC 2403 position
            meanSC = SkyCoord(ra='07h36m51.4s', dec='65d36m09s', frame='icrs')

        self.meanSC = meanSC

        sc_dat = SkyCoord(ra=self.cat['ra']*u.radian, dec=self.cat['dec']*u.radian, frame='icrs')
        # Make things in terms of separation from meanSC, instead of RA, Dec
        meandec = meanSC.dec.value
        self.dRa = (sc_dat.ra.value-meanSC.ra.value)*np.cos(meandec*u.deg)
        self.dDec = sc_dat.dec.value-meanSC.dec.value

        # define the bin edges
        binsize_deg = binsize.to(u.deg)  # arcmin converted to degrees
        binx = binsize_deg/np.cos(meandec*u.deg)  # correct for cos(dec) term
        biny = binsize_deg
        xedges0 = np.arange(1.05*np.nanmin(self.dRa), 1.05*np.nanmax(self.dRa), binx.value)
        yedges0 = np.arange(1.05*np.nanmin(self.dDec), 1.05*np.nanmax(self.dDec), biny.value)

        hh, xedges, yedges = np.histogram2d(self.dRa, self.dDec, bins=(xedges0, yedges0))
        # H needs to be rotated and flipped
        hh = np.rot90(hh)

        # Note that the binsize in RA must be corrected for the cos(Dec) term
        binsize_ra = binsize_deg.value/np.cos(meandec*u.deg)
        decrange = np.max(sc_dat.dec.value)-np.min(sc_dat.dec.value)
        nbins = int(np.ceil(decrange/binsize.to(u.degree).value))
        ra_bins = np.linspace(np.min(sc_dat.ra.value), np.min(sc_dat.ra.value)+nbins*binsize_ra, nbins)
        dec_bins = np.linspace(np.min(sc_dat.dec.value),
                               np.min(sc_dat.dec.value)+nbins*binsize.to(u.degree).value, nbins)

        # 2D histogram of stars within the RGB box:
        binned_counts, xedges, yedges = np.histogram2d(sc_dat.ra.value, sc_dat.dec.value,
                                                       bins=[ra_bins, dec_bins])
        # For display, the histogram array needs to be rotated and flipped (i.e., transposed)
        binned_counts_displ = binned_counts.T

        self.binCounts = binned_counts_displ
        self.xbinEdges = xedges
        self.ybinEdges = yedges
'''


def makeBins(sc_dat, binsize=1.5*u.arcmin):
    # sc_dat: catalog of coordinates for which you'd like to create bins
    #         (as a SkyCoord object)

    # Calculate the mean Dec for correcting the RA offsets:
    meandec = np.mean(sc_dat.dec.value)

    # define the bin size and bin edges
    binsize_deg = binsize.to(u.deg)  # arcmin converted to degrees

    # Note that the binsize in RA must be corrected for the cos(Dec) term
    binsize_ra = binsize_deg.value/np.cos(meandec*u.deg)
    decrange = np.max(sc_dat.dec.value)-np.min(sc_dat.dec.value)
    rarange = (np.max(sc_dat.ra.value)-np.min(sc_dat.ra.value))/np.cos(meandec*u.deg)
    nbins_dec = int(np.ceil(decrange/binsize.to(u.degree).value))
    nbins_ra = int(np.ceil(rarange/binsize_ra.value))
    if nbins_ra > nbins_dec:
        nbins = nbins_ra
    else:
        nbins = nbins_dec
    # Pad the number of bins a bit just to make sure it covers the area:
    nbins += 5

    minra = np.mean(sc_dat.ra.value)-(nbins/2)*binsize_ra.value
    maxra = np.mean(sc_dat.ra.value)+(nbins/2)*binsize_ra.value
    mindec = np.mean(sc_dat.dec.value)-(nbins/2)*binsize_deg.value
    maxdec = np.mean(sc_dat.dec.value)+(nbins/2)*binsize_deg.value
    ra_bins = np.linspace(minra, maxra, nbins)
    dec_bins = np.linspace(mindec, maxdec, nbins)

    return (ra_bins, dec_bins)


def densityMap(sc_dat, ra_bins, dec_bins):
    # ra_bins, dec_bins: bin coordinates as returned by makeBins
    #
    # If desired, input catalog should already be filtered (i.e., this function assumes
    #   you have already done quality cuts, RGB selection, star/galaxy separation, etc.)

    # 2D histogram of stars within the RGB box:
    binned_counts, xedges, yedges = np.histogram2d(sc_dat.ra.value, sc_dat.dec.value,
                                                   bins=[ra_bins, dec_bins])

    return binned_counts


def binsToSkyCoord(ra_bins, dec_bins):
    binsize = (dec_bins[2]*u.deg-dec_bins[1]*u.deg).to(u.arcmin)
    binsize_ra = binsize.value/np.cos(np.mean(dec_bins * u.deg))

    # Create a SkyCoord array for all of the bin centers:
    ra_allbins = []
    dec_allbins = []

    for ii in range(len(ra_bins)-1):
        for jj in range(len(dec_bins)-1):
            ra_allbins = np.append(ra_allbins, (ra_bins[ii]+binsize_ra/2.0))
            dec_allbins = np.append(dec_allbins, (dec_bins[jj]+binsize.to(u.degree).value/2.0))

    sc_allbins = SkyCoord(ra=ra_allbins*u.deg, dec=dec_allbins*u.deg)
    return sc_allbins


def densityBinStats(binned_counts, ra_bins, dec_bins, binned_counts_all=None,
                    binsize=None, inner_annulus=2.0, outer_annulus=5.0):
    # binned_counts: density of filtered sources in 2D bins as returned by densityMap
    # ra_bins, dec_bins: bin coordinates as returned by makeBins
    # binned_counts_all: density of _all_ sources in 2D bins as returned by densityMap

    sc_bins = SkyCoord(ra=ra_bins*u.deg, dec=dec_bins*u.deg)

    # If the "all objects" density map is not provided, set it equal to the filtered one.
    if binned_counts_all is None:
        binned_counts_all = binned_counts.copy()

    if binsize is None:
        binsize = (sc_bins[2].dec-sc_bins[1].dec).to(u.arcmin)

    # binsize_ra = binsize.value/np.cos(np.mean(dec_bins * u.deg))

    # # Create a SkyCoord array for all of the bin centers:
    # ra_allbins = []
    # dec_allbins = []

    # for ii in range(len(ra_bins)-1):
    #     for jj in range(len(dec_bins)-1):
    #        ra_allbins = np.append(ra_allbins, (ra_bins[ii]+binsize_ra/2.0))
    #        dec_allbins = np.append(dec_allbins, (dec_bins[jj]+binsize.to(u.degree).value/2.0))

    # sc_allbins = SkyCoord(ra=ra_allbins*u.deg, dec=dec_allbins*u.deg)

    sc_allbins = binsToSkyCoord(ra_bins, dec_bins)

    # Limits of the background annulus:
    outer_radius = 5.0*binsize
    inner_radius = 2.0*binsize

    bincounts = binned_counts.flatten()
    bincounts_all = binned_counts_all.flatten()
    std_bins = np.copy(bincounts)*0.0
    bg_bins = np.copy(bincounts)*0.0
    bgareafrac_bins = np.copy(bincounts)*0.0
    nsig_bins = np.copy(bincounts)*0.0

    for i_bin in range(len(sc_allbins)):
        # for i_bin in [7035,7036,7037,7038,7039]:
        sep_tmp = sc_allbins[i_bin].separation(sc_allbins)
        # Account for incomplete annuli when calculating the background
        keep_bins_all = (sep_tmp.arcmin >= inner_radius.value) & (sep_tmp.arcmin <= outer_radius.value)
        # print(len(keep_bins_all), len(bincounts_all), len(bincounts))
        # Keep only bins within the footprint (bincounts_all>0) and those that don't overlap the main body of
        #   N2403 (bincounts<30; arbitrarily chosen to remove high starcount regions)
        keep_bins_okbg = keep_bins_all & (bincounts_all > 0) & (bincounts < 30)
        stats_tmp = sigma_clipped_stats(bincounts[keep_bins_okbg], sigma=3, maxiters=10)
        bgareafrac_bins[i_bin] = np.size(bincounts[keep_bins_okbg])/np.size(bincounts[keep_bins_all])
        bg_bins[i_bin] = stats_tmp[0]  # mean background
        std_bins[i_bin] = stats_tmp[2]  # std. deviation of background
        nsig_bins[i_bin] = (bincounts[i_bin]-bg_bins[i_bin])/std_bins[i_bin]

    return (nsig_bins, std_bins, bg_bins, bgareafrac_bins)


def binsToImage(binned_array, binned_counts):
    return binned_array.reshape(np.shape(binned_counts)).T
