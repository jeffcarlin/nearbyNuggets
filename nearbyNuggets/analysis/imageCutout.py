import astropy.units as u
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
from astropy.nddata import Cutout2D
from astropy.visualization import ZScaleInterval
import img_scale


def rgbCutout(rimg, gimg, bimg, cenCoord, cutout_size, savefig=False,
              outputFileName='', emptyG=False,
              rmax=0.6, gmax=0.9, bmax=0.75,
              rmin=0.1, gmin=0.1, bmin=0.1):
    """ Cutout RGB image

    Parameters
    ----------
    rimg : `string`
        path to the red channel FITS image
    gimg : `string`
        path to the green channel FITS image
    bimg : `string`
        path to the blue channel FITS image
    cenCoord : `SkyCoord`
        Astropy skycoord object with central position for cutout
    cutout_size : `float`
        size of cutout image in arcminutes
    savefig : `bool`
        whether to save the figure or not
    outputFileName : `string`
        name for the output image if savefig = True
    """

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Color image:

    params = {
        'axes.labelsize': 24,
        'font.size': 24,
        'legend.fontsize': 14,
        #   'xtick.labelsize': 16,
        'xtick.major.width': 3,
        'xtick.minor.width': 2,
        'xtick.major.size': 8,
        'xtick.minor.size': 5,
        'xtick.direction': 'in',
        'xtick.top': True,
        'lines.linewidth': 3,
        'axes.linewidth': 3,
        'axes.labelweight': 3,
        'axes.titleweight': 3,
        'ytick.major.width': 3,
        'ytick.minor.width': 2,
        'ytick.major.size': 8,
        'ytick.minor.size': 5,
        'ytick.direction': 'in',
        'ytick.right': True,
        #   'ytick.labelsize': 20,
        'text.usetex': True,
        'text.latex.preamble': r'\boldmath',
        'figure.figsize': [9, 9]
    }

    plt.rcParams.update(params)
    fig = plt.figure()

    cutout_size = cutout_size*u.arcmin

    # plt.subplot(324)
    # censtar = np.where(sc_all.separation(sc_bin) == np.min(sc_all.separation(sc_bin)))
    # patch = np.char.strip(cat.dat[censtar]['patch'].data[0])
    # patch = tab[censtar]['patch'].data[0]
    # print('patch: ',patch)
    # img_path = '/Users/jcarlin/Dropbox/local_volume_dwarfs/ngc2403/coadds_jan2021/'
    # gimg_path = img_path+'fakedwarfs_g/calexp-HSC-G-0-'+str(patch)+'_fakes.fits'
    # iimg_path = img_path+'fakedwarfs_i/calexp-HSC-I2-0-'+str(patch)+'_fakes.fits'

    #match_g='/Users/jcarlin/Dropbox/local_volume_dwarfs/ngc2403/coadds_jan2021/fakedwarfs_g/calexp-HSC-G-0-'+str(patch)+'_fakes.fits'
    #match_i='/Users/jcarlin/Dropbox/local_volume_dwarfs/ngc2403/coadds_jan2021/fakedwarfs_i/calexp-HSC-I2-0-'+str(patch)+'_fakes.fits'

    #patchstr = str(patch)+str('.fits')
    #match_g = [gim for gim in gimg_list if patchstr in gim]
    #match_i = [iim for iim in iimg_list if patchstr in iim]
    # print(match_g, match_i)

    hdulist = fits.open(rimg)
    w = WCS(hdulist[1].header, hdulist)
    img_r = hdulist[1].data
    pixscale = np.abs(hdulist[1].header['CD1_1']*3600.0)
    hdulist.close()

    hdulist = fits.open(bimg)
    w = WCS(hdulist[1].header, hdulist)
    img_b = hdulist[1].data
    hdulist.close()

    # imx, imy = w.all_world2pix([sc_all[censtar].ra.value], [sc_all[censtar].dec.value], 0)
    # print(imx, imy)
    # img_r = img_i

    imx, imy = w.all_world2pix([cenCoord.ra.value], [cenCoord.dec.value], 0)

    position = (imx, imy)
    xy_pix_size = np.int(np.floor(cutout_size.to(u.arcsec).value/pixscale))
    size = (xy_pix_size, xy_pix_size)     # pixels

    cutout_r = Cutout2D(img_r, position, size)
    cutout_b = Cutout2D(img_b, position, size)

    if emptyG:
        cutout_g = (cutout_r.data+cutout_b.data)/2
    else:
        hdulist = fits.open(gimg)
        w = WCS(hdulist[1].header, hdulist)
        img_g = hdulist[1].data
        hdulist.close()

        cutout_g = Cutout2D(img_g, position, size)

    zscale = ZScaleInterval()
    vmin_r, vmax_r = zscale.get_limits(cutout_r.data)
    vmin_g, vmax_g = zscale.get_limits(cutout_g.data)
    vmin_b, vmax_b = zscale.get_limits(cutout_b.data)

    img = np.zeros((cutout_r.shape[0], cutout_r.shape[1], 3), dtype=float)
    img[:, :, 0] = img_scale.linear(cutout_r.data, scale_min=rmin*vmin_r, scale_max=rmax*vmax_r)
    img[:, :, 1] = img_scale.linear(cutout_g.data, scale_min=gmin*vmin_g, scale_max=gmax*vmax_g)
    img[:, :, 2] = img_scale.linear(cutout_b.data, scale_min=bmin*vmin_b, scale_max=bmax*vmax_b)

    #    x_rgb, y_rgb = w.all_world2pix(ra[dwarf_msk & isstar & isofilt], dec[dwarf_msk & isstar & isofilt], 0)
    #    x_rgb, y_rgb = w.all_world2pix(ra[dwarf_msk & isstar & rgbbox], dec[dwarf_msk & isstar & rgbbox], 0)
    # x_rgb, y_rgb = w.all_world2pix(ra[cand_flag & star_flag & cmdsel_flag],
    #                                dec[cand_flag & star_flag & cmdsel_flag], 0)

    fig = plt.gca()
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.xaxis.set_ticklabels([])
    fig.axes.get_yaxis().set_visible(False)
    fig.axes.yaxis.set_ticklabels([])

    # linelength_arcsec = 10.0
    # linelength_pix = linelength_arcsec/pixscale
    # plt.hlines(60, 70, 70+linelength_pix, color='White')
    # plt.text(30, 85, '10 arcsec', color='White')

    # print('x: ',x_rgb-imx+(size[0]/2))
    # print('x: ',x_rgb-cutout_i.origin_original[0])
    # print('y: ',y_rgb)
    # if overlay_pts:
    #     plt.plot(x_rgb-cutout_i.xmin_original, y_rgb-cutout_i.ymin_original, 'ws', ms=10,
    #              markeredgewidth=1, fillstyle='none')

    # plt.imshow(img, origin='lower')
    plt.imshow(img, aspect='equal', origin='lower', extent=[0, xy_pix_size, 0, xy_pix_size])

    if savefig:
        if len(outputFileName) > 4:
            savefile = outputFileName
        else:
            savefile = 'rgb_cutout.png'
        plt.savefig(savefile, dpi=180)
        print('Saved figure to ', savefile, '. Rename to avoid overwriting.')

    # plt.tight_layout()
