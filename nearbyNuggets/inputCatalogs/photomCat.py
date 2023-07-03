from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
from dustmaps.sfd import SFDQuery
import numpy as np
import operator


class photomCat:
    def __init__(self, catalogPath, crapFilt=False,
                 gcol='gmag_bgfix', icol='imag_bgfix'):
        self.catalog = catalogPath

        # Assumes data are outputs from the LSST stack.
        #   - Add more flexibility later.

        assert(self.catalog.endswith(('.fits', '.fits.gz', '.fits.fz'))),\
            "Catalog does not have a .fits extension."

        hdulist = fits.open(self.catalog)
        dat = hdulist[1].data

        self.gColumnName = gcol
        self.iColumnName = icol

        if crapFilt:
            # filter out crap detections:
            datfilt = (np.abs(dat[self.gColumnName]) < 29) & (np.abs(dat[self.iColumnName]) < 29) &\
                      (dat[self.gColumnName]-dat[self.iColumnName] > -1.2) &\
                      (dat[self.gColumnName]-dat[self.iColumnName] < 4.0)
#           datfilt = (np.abs(dat['gmag_bgfix']) < 29) & (np.abs(dat['imag_bgfix']) < 29) &\
#                      (dat['gmag_bgfix']-dat['imag_bgfix'] > -1.2) &\
#                      (dat['gmag_bgfix']-dat['imag_bgfix'] < 4.0)
            self.dat = Table(dat[datfilt])
        else:
            self.dat = Table(dat)

        hdulist.close()
        # In [2]: rcat = readwrite.photomCat('/Users/jcarlin/Dropbox/local_volume_dw
        # ...: arfs/ngc2403//catalogs_dec2020/fake_dwarfs/cat_jan2021_fakedwarfs_NGC240
        # ...: 3.fits.gz')

    def setExtinctionCorr(self, units=u.radian):
        coords = SkyCoord(self.dat['ra']*units, self.dat['dec']*units, frame='icrs')

        sfd = SFDQuery()
        ebv = sfd(coords)

        self.dat['ebv'] = ebv
        self.dat['a_g'] = 3.172*ebv
        self.dat['a_i'] = 1.682*ebv

    def setRadiusFlag(self, sc_cen, rad_arcmin, radec_units=u.radian):
        coords = SkyCoord(self.dat['ra']*radec_units, self.dat['dec']*radec_units, frame='icrs')
        spatial_msk_tmp = coords.separation(sc_cen) < (rad_arcmin*u.arcmin)
        self.radiusFlag = spatial_msk_tmp

    def setStarGalFlag(self, iflux_col='iflux', i_modelflux_col='iflux_cmodel',
                       ifluxerr_col='ifluxerr', i_modelfluxerr_col='ifluxerr_cmodel'):
        # Star-galaxy separation:

        # fractional error in flux:
        # iflux_ratio = self.dat['iflux']/self.dat['iflux_cmodel']
        # gflux_ratio = self.dat['gflux']/self.dat['gflux_cmodel']
        # ifluxerr_frac = self.dat['ifluxerr']/self.dat['iflux']
        # ifluxerr_frac_cmodel = self.dat['ifluxerr_cmodel']/self.dat['iflux_cmodel']

        iflux_ratio = self.dat[iflux_col]/self.dat[i_modelflux_col]
        # gflux_ratio = self.dat['gflux']/self.dat['gflux_cmodel']
        ifluxerr_frac = self.dat[ifluxerr_col]/self.dat[iflux_col]
        ifluxerr_frac_cmodel = self.dat[i_modelfluxerr_col]/self.dat[i_modelflux_col]
        ierr_frac_combine = iflux_ratio*np.sqrt((ifluxerr_frac)**2 + (ifluxerr_frac_cmodel)**2)
        # gerr_frac_combine = gflux_ratio*np.sqrt((gfluxerr/gflux)**2 + (gfluxerr_cmodel/gflux_cmodel)**2)
        nsig_stargx = 0.5  # 0.5
        # Give the flux_ratio=1 locus some intrinsic width:
        locus_width = 0.03
        # Both i and g-band:
        # isstar = (np.abs(iflux_ratio-1.0) < (nsig_stargx*ierr_frac_combine+locus_width)) &\
        #          (np.abs(gflux_ratio-1.0) < (nsig_stargx*gerr_frac_combine+locus_width))
        # i-band only:
        isstar = (np.abs(iflux_ratio-1.0) < (nsig_stargx*ierr_frac_combine+locus_width))
        self.isstarFlag = isstar

    def setMagColumnNames(self, gcol='gmag_bgfix', icol='imag_bgfix'):
        self.gColumnName = gcol
        self.iColumnName = icol

    def calcColumn(self, q1=None, q2=None, op='-', colname=''):
        # rcat.calcColumn(q1='gmag_bgfix', q2='imag_bgfix', colname='gi_bgfix', op='-')
        ops = {'+': operator.add,
               '-': operator.sub,
               '*': operator.mul,
               '/': operator.truediv}
        try:
            self.dat[colname] = ops[op](self.dat[q1], self.dat[q2])
        except ValueError as e:
            print(e)

    def rgbBoxFilter(self, corners=[], colorColumn='gi0_bgfix', magColumn='i0_bgfix'):
        # NOTE: assumes corners = ['upper left', 'lower left', 'lower right', 'upper right']
        if len(corners) != 4:
            # If the corners haven't been input, or if the list is too
            # small, revert to a default RGB box.
            corners = [[1.1, 23.7], [0.6, 26.2], [1.5, 26.2], [2.4, 23.7]]
            self.corners = corners
            print('Corners of RGB box set to default: ', self.corners)
        else:
            self.corners = corners

        if (colorColumn in self.dat.colnames) and (magColumn in self.dat.colnames):
            self.slope_left = (self.corners[1][1]-self.corners[0][1])/(self.corners[1][0]-self.corners[0][0])
            self.int_left = self.corners[1][1] - self.slope_left*self.corners[1][0]
            self.slope_right = (self.corners[3][1]-self.corners[2][1])/(self.corners[3][0]-corners[2][0])
            self.int_right = self.corners[3][1] - self.slope_right*corners[3][0]

            self.eval_left = self.slope_left*(self.dat[colorColumn])+self.int_left
            self.eval_right = self.slope_right*(self.dat[colorColumn])+self.int_right

            cmags = [m[1] for m in self.corners]
            minMagBox = np.min(cmags)
            maxMagBox = np.max(cmags)

            rgbbox = (self.dat[magColumn] > self.eval_left) & (self.dat[magColumn] < self.eval_right) &\
                     (self.dat[magColumn] > minMagBox) & (self.dat[magColumn] < maxMagBox)

            self.rgbFlag = rgbbox
        else:
            print('Input magnitude/color columns do not exist in the catalog.')

    def resetFlag(self, flagList=[]):
        allFlagsList = ['rgbFlag', 'radiusFlag', 'isstarFlag']
        if len(flagList) > 0:
            for flag in flagList:
                if flag == 'rgbFlag':
                    self.rgbFlag = np.full(len(self.dat), False, dtype=bool)
                elif flag == 'radiusFlag':
                    self.radiusFlag = np.full(len(self.dat), False, dtype=bool)
                elif flag == 'isstarFlag':
                    self.isstarFlag = np.full(len(self.dat), False, dtype=bool)
                else:
                    print(flag, ' not in list of options.')
        else:
            print('Provide a list of flags to reset as flagList. Options: ', allFlagsList)
