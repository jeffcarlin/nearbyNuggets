import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.visualization import ZScaleInterval
from nearbyNuggets.toolbox import img_scale
from astropy.wcs import WCS
# from astropy.convolution import convolve, Gaussian2DKernel
from nearbyNuggets.toolbox.utils import lf_powerlaw


def plotcand6(sc_inp, cat, i, nsig,
              comp_flag, cmdsel_flag,
              star_flag, binsize=1.5, recalc_cen=False, savefig=False,
              name_append='', overlay_pts=False):
    """ 6 panel diagnostic figure -- CMD, spatial plot, density profile,
        color image, smoothed images in g and i

    Parameters
    ----------
    sc_inp : `astropy.skycoord`
        input central coordinates of the candidate's position
    ###sc_all : `astropy.skycoord`
        ###input coordinates of all objects
    cat : ``
        input catalog of all data
    ### gi_mags : ``
        ### arrays of g and i-band magnitudes ([g_array, i_array])
    comp_flag : `` "cen"
        array of boolean flag values for the comparison population
        (e.g., "cen" for the central regions of the host)
    cmdsel_flag : `` "rgbbox_blue"
        array of boolean flag values to select the sample of interest
        (e.g., RGB stars)
    ### cand_flag_cmd : `` "dwarf_msk_cmd"
        ### array of boolean flag values to select the candidate of interest
        ### (e.g., a candidate dwarf galaxy) for CMD plotting
    ### cand_flag : `` "dwarf_msk"
        ### array of boolean flag values to select the candidate of interest
        ### (e.g., a candidate dwarf galaxy)
    star_flag : `` "isstar"
        array of boolean flag values to select point sources (i.e., stars)
    binsize : `astropy.quantity`
        size of bins in arcminutes
    i : ``

    nsig : `float`
        significance (in # of sigma excess above background) of overdensity
    recalc_cen : `bool`

    savefig : `bool`

    name_append : `str`

    overlay_pts : `bool`
        whether to overlay detections on the images
    """

# To do:
# remove/replace the input "i"
# add RGB box corners as input params
# make img_path an input
# save path for figure

    sc_all = SkyCoord(ra=cat.dat['ra']*u.radian, dec=cat.dat['dec']*u.radian, frame='icrs')
    ra = sc_all.ra.value
    dec = sc_all.dec.value
    gmag = cat.dat['g0_bgfix']
    imag = cat.dat['i0_bgfix']
    gi = gmag-imag

    params = {
        'axes.labelsize': 14,
        'font.size': 14,
        'legend.fontsize': 12,
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
        'figure.figsize': [14, 18],
        'figure.facecolor': 'White'
    }

    plt.rcParams.update(params)

    binsize = binsize*u.arcmin

    # Recalculate the center based on RGB stars, if flag is set:
    if recalc_cen:
        # sc_orig = sc_inp
        dwarf_msk_tmp = sc_all.separation(sc_inp) < (1.5*binsize)
#        med_ra = np.median(ra[dwarf_msk_tmp & isstar & isofilt])
#        med_dec = np.median(dec[dwarf_msk_tmp & isstar & isofilt])
#        med_ra = np.median(ra[dwarf_msk_tmp & isstar & rgbbox])
#        med_dec = np.median(dec[dwarf_msk_tmp & isstar & rgbbox])
        med_ra = np.median(ra[dwarf_msk_tmp & star_flag & cmdsel_flag])
        med_dec = np.median(dec[dwarf_msk_tmp & star_flag & cmdsel_flag])
        sc_bin = SkyCoord(med_ra*u.deg, med_dec*u.deg)
    else:
        sc_bin = sc_inp.copy()

    # For testing:
    print('sc_bin: ', sc_bin)
    print('sc_inp: ', sc_inp)

    cand_flag = sc_all.separation(sc_bin) < (1.5*binsize)
    cand_flag_cmd = sc_all.separation(sc_bin) < (1.0*binsize)
    # "background annulus of same area as cand_flag_cmd"
    cand_flag_bg_cmd = (sc_all.separation(sc_bin) < (2.0*binsize)) &\
                       (sc_all.separation(sc_bin) > np.sqrt(3)*binsize)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Color-magnitude diagram:
    plt.subplot(321)
    plt.scatter(gi[comp_flag & star_flag], imag[comp_flag & star_flag], s=1,
                c='Gray', alpha=.5, label='comparison')
    plt.scatter(gi[cand_flag_cmd & star_flag], imag[cand_flag_cmd & star_flag], s=40,
                color='DodgerBlue', label='candidate overdensity')
    plt.scatter(gi[cand_flag_cmd & star_flag & cmdsel_flag],
                imag[cand_flag_cmd & star_flag & cmdsel_flag], s=40,
                color='Red', label='RGB')

    # To do:
    # plt.plot(xcorners_blue, ycorners_blue, color='Gray')

    plt.legend(loc='upper left')
    plt.xlim(-1, 3)
    plt.ylim(26.8, 19.8)
    plt.xlabel(r'$(g-i)_0$')
    plt.ylabel(r'$i_0$')
    plt.minorticks_on()
    # if recalc_cen:
        # # Print the recalculated bin center on the CMD:
        # ra_str = str('{:7.3f}'.format(sc_bin.ra.value))
        # dec_str = str('{:7.3f}'.format(sc_bin.dec.value))
        # nsig_str = str('{:5.1f}'.format(nsig))
        # plt.title(r', (RA, Dec)$_0$=('+ra_str+','+dec_str+'), Nsig='+nsig_str)
        # # plt.title(str(i)+r', (RA, Dec)$_0$=('+str('{:7.3f}'.format(sc_orig.ra.value))+','+str('{:7.3f}'.format(sc_orig.dec.value))+'), Nsig='+str('{:5.1f}'.format(info)))
    #        plt.title(str(i)+r', (RA, Dec)$_0$=('+str('{:7.3f}'.format(sc_orig.ra.value))+','+str('{:7.3f}'.format(sc_orig.dec.value))+'), Nsig='+str('{:5.1f}'.format(info['SIG'])))
    # else:

    # Print the original bin center on the CMD:
    ra_str = str('{:7.3f}'.format(sc_inp.ra.value))
    dec_str = str('{:7.3f}'.format(sc_inp.dec.value))
    nsig_str = str('{:5.1f}'.format(nsig))
    plt.title(r', (RA, Dec)$_0$=('+ra_str+','+dec_str+'), Nsig='+nsig_str)

            # plt.title(str(i)+', (RA, Dec)=('+str('{:7.3f}'.format(sc_inp.ra.value))+','+str('{:7.3f}'.format(sc_inp.dec.value))+'), Nsig='+str('{:5.1f}'.format(info)))
    #        plt.title(str(i)+', (RA, Dec)=('+str('{:7.3f}'.format(sc_inp.ra.value))+','+str('{:7.3f}'.format(sc_inp.dec.value))+'), Nsig='+str('{:5.1f}'.format(info['SIG'])))

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Spatial plot:
    plt.subplot(322)
    plt.plot(ra[cand_flag & ~star_flag], dec[cand_flag & ~star_flag], '.',
             ms=4, color='Gray', alpha=0.3, label='not stars')
    plt.plot(ra[cand_flag & star_flag], dec[cand_flag & star_flag], '.',
             ms=10, color='DodgerBlue', label='stars')
    #    plt.plot(ra[dwarf_msk & isstar & isofilt], dec[dwarf_msk & isstar & isofilt], '.', ms=10, color='Red', label='RGB stars')
    #    plt.plot(ra[dwarf_msk & isstar & rgbbox], dec[dwarf_msk & isstar & rgbbox], '.', ms=10, color='Red', label='RGB stars')
    plt.plot(ra[cand_flag & star_flag & cmdsel_flag], dec[cand_flag & star_flag & cmdsel_flag],
             '.', ms=10, color='Red', label='RGB stars')
    plt.xlabel('RA (deg)')
    plt.ylabel('DEC (deg)')
    plt.legend(loc='upper left')
    plt.gca().invert_xaxis()
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=5)
    plt.minorticks_on()

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 1D radial density plot
    nradbins = 10
    radbinsize = 2*binsize/nradbins
    radbins = np.linspace(0, binsize, nradbins+1)
    radbin_cens = (radbins[1:] + radbins[:-1]) / 2.0
    rad_density = np.zeros(nradbins)
    for irad in range(nradbins):
        area = 2*np.pi*(radbins[irad+1].value**2 - radbins[irad].value**2)
        annulus_mask = (sc_all.separation(sc_bin) >= radbins[irad]) &\
                       (sc_all.separation(sc_bin) < radbins[irad+1])
    #        rad_density[irad] = len(ra[annulus_mask & isstar & isofilt]) / area
    #        rad_density[irad] = len(ra[annulus_mask & isstar & rgbbox]) / area
        rad_density[irad] = len(ra[annulus_mask & star_flag & cmdsel_flag]) / area

    plt.subplot(323)
    plt.plot(radbin_cens, rad_density, '-.')
    plt.plot(radbin_cens, rad_density, 'ko')
    plt.minorticks_on()

    # if recalc_cen:
    ra_str = str('{:7.3f}'.format(sc_bin.ra.value))
    dec_str = str('{:7.3f}'.format(sc_bin.dec.value))
    nsig_str = str('{:5.1f}'.format(nsig))
    plt.title(r', (RA, Dec)$_0$=('+ra_str+','+dec_str+'), Nsig='+nsig_str)

        # plt.title(str(i)+r', (RA, Dec)$_{new}$=('+str('{:7.3f}'.format(sc_inp.ra.value))+','+str('{:7.3f}'.format(sc_inp.dec.value))+')')

    plt.xlabel('Distance from center [arcmin]')
    plt.ylabel(r'Number of stars per arcsec$^2$')

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Color image:
    plt.subplot(324)
    censtar = np.where(sc_all.separation(sc_bin) == np.min(sc_all.separation(sc_bin)))
    patch = np.char.strip(cat.dat[censtar]['patch'].data[0])
    # patch = tab[censtar]['patch'].data[0]
# print('patch: ',patch)
    img_path = '/Users/jcarlin/Dropbox/local_volume_dwarfs/ngc2403/coadds_jan2021/'
    gimg_path = img_path+'fakedwarfs_g/calexp-HSC-G-0-'+str(patch)+'_fakes.fits'
    iimg_path = img_path+'fakedwarfs_i/calexp-HSC-I2-0-'+str(patch)+'_fakes.fits'

#match_g='/Users/jcarlin/Dropbox/local_volume_dwarfs/ngc2403/coadds_jan2021/fakedwarfs_g/calexp-HSC-G-0-'+str(patch)+'_fakes.fits'
#match_i='/Users/jcarlin/Dropbox/local_volume_dwarfs/ngc2403/coadds_jan2021/fakedwarfs_i/calexp-HSC-I2-0-'+str(patch)+'_fakes.fits'

#patchstr = str(patch)+str('.fits')
#match_g = [gim for gim in gimg_list if patchstr in gim]
#match_i = [iim for iim in iimg_list if patchstr in iim]
# print(match_g, match_i)
    hdulist = fits.open(iimg_path)
    w = WCS(hdulist[1].header, hdulist)
    img_i = hdulist[1].data
    hdulist.close()

    hdulist = fits.open(gimg_path)
    w = WCS(hdulist[1].header, hdulist)
    img_g = hdulist[1].data
    pixscale = np.abs(hdulist[1].header['CD1_1']*3600.0)
    hdulist.close()

    imx, imy = w.all_world2pix([sc_all[censtar].ra.value], [sc_all[censtar].dec.value], 0)
    img_r = img_i

    position = (imx, imy)
    xy_pix_size = int(np.floor(1.0*binsize.to(u.arcsec).value/pixscale))
    size = (xy_pix_size, xy_pix_size)     # pixels

    cutout_g = Cutout2D(img_g, position, size)
    cutout_r_tmp = Cutout2D(img_r, position, size)
    cutout_i = Cutout2D(img_i, position, size)

    cutout_r = (cutout_r_tmp.data+cutout_g.data)/2

    zscale = ZScaleInterval()
    vmin_g, vmax_g = zscale.get_limits(cutout_g.data)
    vmin_r, vmax_r = zscale.get_limits(cutout_r.data)
    vmin_i, vmax_i = zscale.get_limits(cutout_i.data)

    img = np.zeros((cutout_i.shape[0], cutout_i.shape[1], 3), dtype=float)
    img[:, :, 0] = img_scale.linear(cutout_i.data, scale_min=0.1*vmin_i, scale_max=0.6*vmax_i)
    img[:, :, 1] = img_scale.linear(cutout_r.data, scale_min=0.1*vmin_r, scale_max=0.9*vmax_r)
    img[:, :, 2] = img_scale.linear(cutout_g.data, scale_min=0.1*vmin_g, scale_max=0.75*vmax_g)

#    x_rgb, y_rgb = w.all_world2pix(ra[dwarf_msk & isstar & isofilt], dec[dwarf_msk & isstar & isofilt], 0)
#    x_rgb, y_rgb = w.all_world2pix(ra[dwarf_msk & isstar & rgbbox], dec[dwarf_msk & isstar & rgbbox], 0)
#    x_rgb, y_rgb = w.all_world2pix(ra[cand_flag & star_flag & cmdsel_flag],
#                                   dec[cand_flag & star_flag & cmdsel_flag], 0)

    # Create RA, Dec ticks, labels:
    nticks = 4
    xtick_locs = np.arange(0, xy_pix_size, xy_pix_size/nticks) + xy_pix_size/(2*nticks)
    ytick_locs = np.arange(0, xy_pix_size, xy_pix_size/nticks) + xy_pix_size/(2*nticks)

    tick_locs_tmp = []
    for loc in xtick_locs:
        tick_locs_tmp.append(cutout_g.to_original_position((loc, 0)))

    xtick_locs_orig = []
    for loc in tick_locs_tmp:
        xtick_locs_orig.append(loc[0])

    tick_locs_tmp = []
    for loc in ytick_locs:
        tick_locs_tmp.append(cutout_g.to_original_position((0, loc)))

    ytick_locs_orig = []
    for loc in tick_locs_tmp:
        ytick_locs_orig.append(loc[1])

    xtick_lbls = []
    ytick_lbls = []
    for xt, yt in zip(xtick_locs_orig, ytick_locs_orig):
        temp = w.all_pix2world([[xt, yt]], 0)
        xtick_lbls.append(str(np.round(temp[0][0], 2)))
        ytick_lbls.append(str(np.round(temp[0][1], 2)))

    fig = plt.gca()
    fig.axes.xaxis.set_ticks(xtick_locs, labels=xtick_lbls)
    fig.axes.yaxis.set_ticks(ytick_locs, labels=ytick_lbls)
    fig.axes.tick_params(direction='out')
    fig.axes.get_xaxis().set_visible(True)
    fig.axes.get_yaxis().set_visible(True)

    linelength_arcsec = 10.0
    linelength_pix = linelength_arcsec/pixscale
    fig.hlines(60, 70, 70+linelength_pix, color='White')
    fig.text(30, 85, '10 arcsec', color='White')

    # print('x: ',x_rgb-imx+(size[0]/2))
    # print('x: ',x_rgb-cutout_i.origin_original[0])
    # print('y: ',y_rgb)
    # if overlay_pts:
        # fig.plot(x_rgb-cutout_i.xmin_original, y_rgb-cutout_i.ymin_original, 'ws', ms=10,
                 # markeredgewidth=1, fillstyle='none')

    # plt.imshow(img, origin='lower')
    fig.imshow(img, aspect='equal', origin='lower', extent=[0, xy_pix_size, 0, xy_pix_size])

    # plt.tight_layout()

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Color image:
    plt.subplot(325)

    xy_pix_size = int(np.floor(2.0*binsize.to(u.arcsec).value/pixscale))
    size = (xy_pix_size, xy_pix_size)     # pixels

    cutout_g = Cutout2D(img_g, position, size)
    cutout_r_tmp = Cutout2D(img_r, position, size)
    cutout_i = Cutout2D(img_i, position, size)

    cutout_r = (cutout_r_tmp.data+cutout_g.data)/2

    zscale = ZScaleInterval()
    vmin_g, vmax_g = zscale.get_limits(cutout_g.data)
    vmin_r, vmax_r = zscale.get_limits(cutout_r.data)
    vmin_i, vmax_i = zscale.get_limits(cutout_i.data)

    img = np.zeros((cutout_i.shape[0], cutout_i.shape[1], 3), dtype=float)
    img[:, :, 0] = img_scale.linear(cutout_i.data, scale_min=0.1*vmin_i, scale_max=0.6*vmax_i)
    img[:, :, 1] = img_scale.linear(cutout_r.data, scale_min=0.1*vmin_r, scale_max=0.9*vmax_r)
    img[:, :, 2] = img_scale.linear(cutout_g.data, scale_min=0.1*vmin_g, scale_max=0.75*vmax_g)

    # Create RA, Dec ticks, labels:
    nticks = 4
    xtick_locs = np.arange(0, xy_pix_size, xy_pix_size/nticks) + xy_pix_size/(2*nticks)
    ytick_locs = np.arange(0, xy_pix_size, xy_pix_size/nticks) + xy_pix_size/(2*nticks)

    tick_locs_tmp = []
    for loc in xtick_locs:
        tick_locs_tmp.append(cutout_g.to_original_position((loc, 0)))

    xtick_locs_orig = []
    for loc in tick_locs_tmp:
        xtick_locs_orig.append(loc[0])

    tick_locs_tmp = []
    for loc in ytick_locs:
        tick_locs_tmp.append(cutout_g.to_original_position((0, loc)))

    ytick_locs_orig = []
    for loc in tick_locs_tmp:
        ytick_locs_orig.append(loc[1])

    xtick_lbls = []
    ytick_lbls = []
    for xt, yt in zip(xtick_locs_orig, ytick_locs_orig):
        temp = w.all_pix2world([[xt, yt]], 0)
        xtick_lbls.append(str(np.round(temp[0][0], 2)))
        ytick_lbls.append(str(np.round(temp[0][1], 2)))

    fig = plt.gca()
    fig.axes.xaxis.set_ticks(xtick_locs, labels=xtick_lbls)
    fig.axes.yaxis.set_ticks(ytick_locs, labels=ytick_lbls)
    fig.axes.tick_params(direction='out')
    fig.axes.get_xaxis().set_visible(True)
    fig.axes.get_yaxis().set_visible(True)

    linelength_arcsec = 10.0
    linelength_pix = linelength_arcsec/pixscale
    fig.hlines(60, 70, 70+linelength_pix, color='White')
    fig.text(30, 85, '10 arcsec', color='White')

    fig.imshow(img, aspect='equal', origin='lower', extent=[0, xy_pix_size, 0, xy_pix_size])

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Color image:
    plt.subplot(326)

    x_rgb, y_rgb = w.all_world2pix(ra[cand_flag & star_flag & cmdsel_flag],
                                   dec[cand_flag & star_flag & cmdsel_flag], 0)

    fig = plt.gca()
    fig.axes.xaxis.set_ticks(xtick_locs, labels=xtick_lbls)
    fig.axes.yaxis.set_ticks(ytick_locs, labels=ytick_lbls)
    fig.axes.tick_params(direction='out')
    fig.axes.get_xaxis().set_visible(True)
    fig.axes.get_yaxis().set_visible(True)

    linelength_arcsec = 10.0
    linelength_pix = linelength_arcsec/pixscale
    fig.hlines(60, 70, 70+linelength_pix, color='White')
    fig.text(30, 85, '10 arcsec', color='White')

    if overlay_pts:
        fig.plot(x_rgb-cutout_i.xmin_original, y_rgb-cutout_i.ymin_original, 'ws', ms=10,
                 markeredgewidth=1, fillstyle='none')

    fig.imshow(img, aspect='equal', origin='lower', extent=[0, xy_pix_size, 0, xy_pix_size])

    if savefig:
        savefile = 'cand_diagnostic_plots_6panel_'+name_append+'.png'
        plt.savefig(savefile, dpi=180)
        print('Saved figure to ', savefile, '. Rename to avoid overwriting.')

    plt.show()
    plt.close()


def plotcand6b(sc_inp, cat, i, nsig,
               comp_flag, cmdsel_flag,
               star_flag, binsize=1.5, recalc_cen=False, savefig=False,
               name_append='', overlay_pts=False):
    """ 6 panel diagnostic figure -- CMD, spatial plot, density profile,
        color image, smoothed images in g and i

    Parameters
    ----------
    sc_inp : `astropy.skycoord`
        input central coordinates of the candidate's position
    ###sc_all : `astropy.skycoord`
        ###input coordinates of all objects
    cat : ``
        input catalog of all data
    ### gi_mags : ``
        ### arrays of g and i-band magnitudes ([g_array, i_array])
    comp_flag : `` "cen"
        array of boolean flag values for the comparison population
        (e.g., "cen" for the central regions of the host)
    cmdsel_flag : `` "rgbbox_blue"
        array of boolean flag values to select the sample of interest
        (e.g., RGB stars)
    ### cand_flag_cmd : `` "dwarf_msk_cmd"
        ### array of boolean flag values to select the candidate of interest
        ### (e.g., a candidate dwarf galaxy) for CMD plotting
    ### cand_flag : `` "dwarf_msk"
        ### array of boolean flag values to select the candidate of interest
        ### (e.g., a candidate dwarf galaxy)
    star_flag : `` "isstar"
        array of boolean flag values to select point sources (i.e., stars)
    binsize : `astropy.quantity`
        size of bins in arcminutes
    i : ``

    nsig : `float`
        significance (in # of sigma excess above background) of overdensity
    recalc_cen : `bool`

    savefig : `bool`

    name_append : `str`

    overlay_pts : `bool`
        whether to overlay detections on the images
    """

# To do:
# remove/replace the input "i"
# add RGB box corners as input params
# make img_path an input
# save path for figure

    sc_all = SkyCoord(ra=cat.dat['ra']*u.radian, dec=cat.dat['dec']*u.radian, frame='icrs')
    ra = sc_all.ra.value
    dec = sc_all.dec.value
    gmag = cat.dat['g0_bgfix']
    imag = cat.dat['i0_bgfix']
    gi = gmag-imag

    params = {
        'axes.labelsize': 14,
        'font.size': 14,
        'legend.fontsize': 12,
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
        'figure.figsize': [14, 18]
    }

    plt.rcParams.update(params)

    binsize = binsize*u.arcmin

    # Recalculate the center based on RGB stars, if flag is set:
    if recalc_cen:
        # sc_orig = sc_inp
        dwarf_msk_tmp = sc_all.separation(sc_inp) < (1.5*binsize)
        med_ra = np.median(ra[dwarf_msk_tmp & star_flag & cmdsel_flag])
        med_dec = np.median(dec[dwarf_msk_tmp & star_flag & cmdsel_flag])
        sc_bin = SkyCoord(med_ra*u.deg, med_dec*u.deg)
    else:
        sc_bin = sc_inp.copy()

    # For testing:
    print('sc_bin: ', sc_bin)
    print('sc_inp: ', sc_inp)

    cand_flag = sc_all.separation(sc_bin) < (1.5*binsize)
    cand_flag_cmd = sc_all.separation(sc_bin) < (1.0*binsize)
    cand_flag_bg_cmd = (sc_all.separation(sc_bin) < (2.0*binsize)) &\
                       (sc_all.separation(sc_bin) > np.sqrt(3)*binsize)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Color-magnitude diagram:
    plt.subplot(321)
    plt.scatter(gi[comp_flag & star_flag], imag[comp_flag & star_flag], s=1,
                c='Gray', alpha=.5, label='comparison')
    plt.scatter(gi[cand_flag_cmd & star_flag], imag[cand_flag_cmd & star_flag], s=40,
                color='DodgerBlue', label='candidate overdensity')
    plt.scatter(gi[cand_flag_cmd & star_flag & cmdsel_flag],
                imag[cand_flag_cmd & star_flag & cmdsel_flag], s=40,
                color='Red', label='RGB')

    # To do:
    # plt.plot(xcorners_blue, ycorners_blue, color='Gray')

    plt.legend(loc='upper left')
    plt.xlim(-1, 3)
    plt.ylim(26.8, 19.8)
    plt.xlabel(r'$(g-i)_0$')
    plt.ylabel(r'$i_0$')
    plt.minorticks_on()

    # Print the original bin center on the CMD:
    ra_str = str('{:7.3f}'.format(sc_inp.ra.value))
    dec_str = str('{:7.3f}'.format(sc_inp.dec.value))
    nsig_str = str('{:5.1f}'.format(nsig))
    plt.title(r', (RA, Dec)$_0$=('+ra_str+','+dec_str+'), Nsig='+nsig_str)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Spatial plot:
    plt.subplot(322)
    plt.plot(ra[cand_flag & ~star_flag], dec[cand_flag & ~star_flag], '.',
             ms=4, color='Gray', alpha=0.3, label='not stars')
    plt.plot(ra[cand_flag & star_flag], dec[cand_flag & star_flag], '.',
             ms=10, color='DodgerBlue', label='stars')
    plt.plot(ra[cand_flag & star_flag & cmdsel_flag], dec[cand_flag & star_flag & cmdsel_flag],
             '.', ms=10, color='Red', label='RGB stars')
    plt.xlabel('RA (deg)')
    plt.ylabel('DEC (deg)')
    plt.legend(loc='upper left')
    plt.gca().invert_xaxis()
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=5)
    plt.minorticks_on()

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 1D radial density plot
    nradbins = 10
    radbinsize = 2*binsize/nradbins
    radbins = np.linspace(0, binsize, nradbins+1)
    radbin_cens = (radbins[1:] + radbins[:-1]) / 2.0
    rad_density = np.zeros(nradbins)
    for irad in range(nradbins):
        area = 2*np.pi*(radbins[irad+1].value**2 - radbins[irad].value**2)
        annulus_mask = (sc_all.separation(sc_bin) >= radbins[irad]) &\
                       (sc_all.separation(sc_bin) < radbins[irad+1])
        rad_density[irad] = len(ra[annulus_mask & star_flag & cmdsel_flag]) / area

    plt.subplot(323)
    plt.plot(radbin_cens, rad_density, '-.')
    plt.plot(radbin_cens, rad_density, 'ko')
    plt.minorticks_on()

    # if recalc_cen:
    ra_str = str('{:7.3f}'.format(sc_bin.ra.value))
    dec_str = str('{:7.3f}'.format(sc_bin.dec.value))
    nsig_str = str('{:5.1f}'.format(nsig))
    plt.title(r', (RA, Dec)$_0$=('+ra_str+','+dec_str+'), Nsig='+nsig_str)

    plt.xlabel('Distance from center [arcmin]')
    plt.ylabel(r'Number of stars per arcsec$^2$')

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Color image:
    plt.subplot(324)
    censtar = np.where(sc_all.separation(sc_bin) == np.min(sc_all.separation(sc_bin)))
    patch = np.char.strip(cat.dat[censtar]['patch'].data[0])

    img_path = '/Users/jcarlin/Dropbox/local_volume_dwarfs/ngc2403/coadds_jan2021/'
    gimg_path = img_path+'fakedwarfs_g/calexp-HSC-G-0-'+str(patch)+'_fakes.fits'
    iimg_path = img_path+'fakedwarfs_i/calexp-HSC-I2-0-'+str(patch)+'_fakes.fits'

    hdulist = fits.open(iimg_path)
    w = WCS(hdulist[1].header, hdulist)
    img_i = hdulist[1].data
    hdulist.close()

    hdulist = fits.open(gimg_path)
    w = WCS(hdulist[1].header, hdulist)
    img_g = hdulist[1].data
    pixscale = np.abs(hdulist[1].header['CD1_1']*3600.0)
    hdulist.close()

    imx, imy = w.all_world2pix([sc_all[censtar].ra.value], [sc_all[censtar].dec.value], 0)
    img_r = img_i

    position = (imx, imy)
    xy_pix_size = int(np.floor(2.0*binsize.to(u.arcsec).value/pixscale))
    size = (xy_pix_size, xy_pix_size)     # pixels

    cutout_g = Cutout2D(img_g, position, size)
    cutout_r_tmp = Cutout2D(img_r, position, size)
    cutout_i = Cutout2D(img_i, position, size)

    cutout_r = (cutout_r_tmp.data+cutout_g.data)/2

    zscale = ZScaleInterval()
    vmin_g, vmax_g = zscale.get_limits(cutout_g.data)
    vmin_r, vmax_r = zscale.get_limits(cutout_r.data)
    vmin_i, vmax_i = zscale.get_limits(cutout_i.data)

    img = np.zeros((cutout_i.shape[0], cutout_i.shape[1], 3), dtype=float)
    img[:, :, 0] = img_scale.linear(cutout_i.data, scale_min=0.1*vmin_i, scale_max=0.6*vmax_i)
    img[:, :, 1] = img_scale.linear(cutout_r.data, scale_min=0.1*vmin_r, scale_max=0.9*vmax_r)
    img[:, :, 2] = img_scale.linear(cutout_g.data, scale_min=0.1*vmin_g, scale_max=0.75*vmax_g)

    # Create RA, Dec ticks, labels:
    nticks = 4
    xtick_locs = np.arange(0, xy_pix_size, xy_pix_size/nticks) + xy_pix_size/(2*nticks)
    ytick_locs = np.arange(0, xy_pix_size, xy_pix_size/nticks) + xy_pix_size/(2*nticks)

    tick_locs_tmp = []
    for loc in xtick_locs:
        tick_locs_tmp.append(cutout_g.to_original_position((loc, 0)))

    xtick_locs_orig = []
    for loc in tick_locs_tmp:
        xtick_locs_orig.append(loc[0])

    tick_locs_tmp = []
    for loc in ytick_locs:
        tick_locs_tmp.append(cutout_g.to_original_position((0, loc)))

    ytick_locs_orig = []
    for loc in tick_locs_tmp:
        ytick_locs_orig.append(loc[1])

    xtick_lbls = []
    ytick_lbls = []
    for xt, yt in zip(xtick_locs_orig, ytick_locs_orig):
        temp = w.all_pix2world([[xt, yt]], 0)
        xtick_lbls.append(str(np.round(temp[0][0], 2)))
        ytick_lbls.append(str(np.round(temp[0][1], 2)))

    fig = plt.gca()
    fig.axes.xaxis.set_ticks(xtick_locs, labels=xtick_lbls)
    fig.axes.yaxis.set_ticks(ytick_locs, labels=ytick_lbls)
    fig.axes.tick_params(direction='out')
    fig.axes.get_xaxis().set_visible(True)
    fig.axes.get_yaxis().set_visible(True)

    linelength_arcsec = 10.0
    linelength_pix = linelength_arcsec/pixscale
    fig.hlines(60, 70, 70+linelength_pix, color='White')
    fig.text(30, 85, '10 arcsec', color='White')

    fig.imshow(img, aspect='equal', origin='lower', extent=[0, xy_pix_size, 0, xy_pix_size])

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Luminosity function:
    plt.subplot(325)

    lf_binsize = 0.25
    lf_minmag = 21.25
    lf_maxmag = 26.5
    lf_bins = np.arange(lf_minmag, lf_maxmag, lf_binsize)
    imag_cand_rgb = imag[cand_flag_cmd & star_flag & cmdsel_flag]
    imag_cand_bg_rgb = imag[cand_flag_bg_cmd & star_flag & cmdsel_flag]
    imag_cand_all = imag[cand_flag & ~cand_flag_cmd & star_flag]

    lf_mag_all, lf_hist_bins = np.histogram(imag_cand_all, bins=lf_bins)
    lf_bin_cens = lf_hist_bins + lf_binsize/2.0
    lf_mag_rgb, lf_hist_bins = np.histogram(imag_cand_rgb, bins=lf_bins)
    lf_mag_bg_rgb, lf_hist_bins = np.histogram(imag_cand_bg_rgb, bins=lf_bins)

    #plt.plot(lf_bin_cens[:-1], np.cumsum(lf_mag_all), '-.', color='DodgerBlue', label='all stars')
    #plt.plot(lf_bin_cens[:-1], np.cumsum(lf_mag_all), 'ko')
    plt.plot(lf_bin_cens[:-1], np.cumsum(lf_mag_rgb), '-.', color='Red', label='cmd selected')
    plt.plot(lf_bin_cens[:-1], np.cumsum(lf_mag_rgb), 'ko')
    plt.plot(lf_bin_cens[:-1], np.cumsum(lf_mag_bg_rgb), '-.', color='Gray', label='background w/ cmd selection')
    plt.plot(lf_bin_cens[:-1], np.cumsum(lf_mag_bg_rgb), 'ko')
    #plt.plot(lf_bin_cens[:-1], np.cumsum(lf_mag_rgb)/np.cumsum(lf_mag_all), '-.', color='DodgerBlue')
    plt.legend()
    plt.minorticks_on()
    plt.semilogy()

    #def lf_powerlaw(mags, lf_inp, alpha=2.35):
    #    # dN = Phi*m^(-alpha), where "m" is the magnitude
    #    # Want dN = value at mag25bin, so Phi = dN*m^(alpha) = dN*25^(alpha)
    #    mag25bin = np.argmin(np.abs(mags - 25.0))
    #    phi_salpeter = np.sum(lf_inp[:mag25bin])  # *(25.0**(-1*alpha))
    #    return phi_salpeter*(10.0**((6.0*alpha/5.0) * (mags-25.0)))

    # print(phi_salpeter)
    # lf_salpeter = phi_salpeter * (lf_bin_cens**(-2.35))

    lf_salpeter = lf_powerlaw(lf_bin_cens, lf_mag_rgb, alpha=2.35)
    #plt.plot(lf_bin_cens[:-1], lf_salpeter[:-1], ':', color='Gray')

    lf_geha = lf_powerlaw(lf_bin_cens, lf_mag_rgb, alpha=1.30)
    #plt.plot(lf_bin_cens[:-1], lf_geha[:-1], '--', color='Gray')

    # plt.title('luminosity function')
    plt.ylim(8*10**(-1), 1.15*np.sum(lf_mag_rgb))

    plt.xlabel('imag')
    plt.ylabel(r'$N_{*, cand} <$ imag')

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Color image:
    plt.subplot(326)

    #xy_pix_size = np.int(np.floor(2.0*binsize.to(u.arcsec).value/pixscale))
    #size = (xy_pix_size, xy_pix_size)     # pixels

    #cutout_g = Cutout2D(img_g, position, size)
    #cutout_r_tmp = Cutout2D(img_r, position, size)
    #cutout_i = Cutout2D(img_i, position, size)

    #cutout_r = (cutout_r_tmp.data+cutout_g.data)/2

    #zscale = ZScaleInterval()
    #vmin_g, vmax_g = zscale.get_limits(cutout_g.data)
    #vmin_r, vmax_r = zscale.get_limits(cutout_r.data)
    #vmin_i, vmax_i = zscale.get_limits(cutout_i.data)

    #img = np.zeros((cutout_i.shape[0], cutout_i.shape[1], 3), dtype=float)
    #img[:, :, 0] = img_scale.linear(cutout_i.data, scale_min=0.1*vmin_i, scale_max=0.6*vmax_i)
    #img[:, :, 1] = img_scale.linear(cutout_r.data, scale_min=0.1*vmin_r, scale_max=0.9*vmax_r)
    #img[:, :, 2] = img_scale.linear(cutout_g.data, scale_min=0.1*vmin_g, scale_max=0.75*vmax_g)

    ## Create RA, Dec ticks, labels:
    #nticks = 4
    #xtick_locs = np.arange(0, xy_pix_size, xy_pix_size/nticks) + xy_pix_size/(2*nticks)
    #ytick_locs = np.arange(0, xy_pix_size, xy_pix_size/nticks) + xy_pix_size/(2*nticks)

    #tick_locs_tmp = []
    #for loc in xtick_locs:
    #    tick_locs_tmp.append(cutout_g.to_original_position((loc, 0)))

    #xtick_locs_orig = []
    #for loc in tick_locs_tmp:
    #    xtick_locs_orig.append(loc[0])

    #tick_locs_tmp = []
    #for loc in ytick_locs:
    #    tick_locs_tmp.append(cutout_g.to_original_position((0, loc)))

    #ytick_locs_orig = []
    #for loc in tick_locs_tmp:
    #    ytick_locs_orig.append(loc[1])

    #xtick_lbls = []
    #ytick_lbls = []
    #for xt, yt in zip(xtick_locs_orig, ytick_locs_orig):
    #    temp = w.all_pix2world([[xt, yt]], 0)
    #    xtick_lbls.append(str(np.round(temp[0][0], 2)))
    #    ytick_lbls.append(str(np.round(temp[0][1], 2)))

    fig = plt.gca()
    fig.axes.xaxis.set_ticks(xtick_locs, labels=xtick_lbls)
    fig.axes.yaxis.set_ticks(ytick_locs, labels=ytick_lbls)
    fig.axes.tick_params(direction='out')
    fig.axes.get_xaxis().set_visible(True)
    fig.axes.get_yaxis().set_visible(True)

    #linelength_arcsec = 10.0
    #linelength_pix = linelength_arcsec/pixscale
    #fig.hlines(60, 70, 70+linelength_pix, color='White')
    #fig.text(30, 85, '10 arcsec', color='White')

    x_rgb, y_rgb = w.all_world2pix(ra[cand_flag & star_flag & cmdsel_flag],
                                   dec[cand_flag & star_flag & cmdsel_flag], 0)

    if overlay_pts:
        fig.plot(x_rgb-cutout_i.xmin_original, y_rgb-cutout_i.ymin_original, 'ws', ms=10,
                 markeredgewidth=1, fillstyle='none')

    fig.imshow(img, aspect='equal', origin='lower', extent=[0, xy_pix_size, 0, xy_pix_size])

    if savefig:
        savefile = 'cand_diagnostic_plots_6panel_'+name_append+'.png'
        plt.savefig(savefile, dpi=180)
        print('Saved figure to ', savefile, '. Rename to avoid overwriting.')
    else:
        plt.show()

    plt.close()


def plotcand6_gimli(sc_inp, cat, i, nsig,
                    comp_flag, cmdsel_flag,
                    star_flag, tract, patch,
                    binsize=1.5, recalc_cen=False, savefig=False,
                    name_append='', overlay_pts=False):
    """ 6 panel diagnostic figure -- CMD, spatial plot, density profile,
        color image, smoothed images in g and i

    Parameters
    ----------
    sc_inp : `astropy.skycoord`
        input central coordinates of the candidate's position
    ###sc_all : `astropy.skycoord`
        ###input coordinates of all objects
    cat : ``
        input catalog of all data
    ### gi_mags : ``
        ### arrays of g and i-band magnitudes ([g_array, i_array])
    comp_flag : `` "cen"
        array of boolean flag values for the comparison population
        (e.g., "cen" for the central regions of the host)
    cmdsel_flag : `` "rgbbox_blue"
        array of boolean flag values to select the sample of interest
        (e.g., RGB stars)
    ### cand_flag_cmd : `` "dwarf_msk_cmd"
        ### array of boolean flag values to select the candidate of interest
        ### (e.g., a candidate dwarf galaxy) for CMD plotting
    ### cand_flag : `` "dwarf_msk"
        ### array of boolean flag values to select the candidate of interest
        ### (e.g., a candidate dwarf galaxy)
    star_flag : `` "isstar"
        array of boolean flag values to select point sources (i.e., stars)
    binsize : `astropy.quantity`
        size of bins in arcminutes
    i : ``

    nsig : `float`
        significance (in # of sigma excess above background) of overdensity
    recalc_cen : `bool`

    savefig : `bool`

    name_append : `str`

    overlay_pts : `bool`
        whether to overlay detections on the images
    """

# To do:
# remove/replace the input "i"
# add RGB box corners as input params
# make img_path an input
# save path for figure

    sc_all = SkyCoord(ra=cat.dat['coord_ra']*u.degree, dec=cat.dat['coord_dec']*u.degree, frame='icrs')
    ra = sc_all.ra.value
    dec = sc_all.dec.value
    gmag = cat.dat['g0']
    imag = cat.dat['i0']
    gi = gmag-imag

    params = {
        'axes.labelsize': 14,
        'font.size': 14,
        'legend.fontsize': 12,
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
        'figure.facecolor': 'White',
        'figure.figsize': [14, 18]
    }

    plt.rcParams.update(params)

    binsize = binsize*u.arcmin

    # Recalculate the center based on RGB stars, if flag is set:
    if recalc_cen:
        dwarf_msk_tmp = sc_all.separation(sc_inp) < (1.5*binsize)
        med_ra = np.median(ra[dwarf_msk_tmp & star_flag & cmdsel_flag])
        med_dec = np.median(dec[dwarf_msk_tmp & star_flag & cmdsel_flag])
        sc_bin = SkyCoord(med_ra*u.deg, med_dec*u.deg)
    else:
        sc_bin = sc_inp.copy()

    # For testing:
    print('sc_bin: ', sc_bin)
    print('sc_inp: ', sc_inp)

    cand_flag = sc_all.separation(sc_bin) < (1.5*binsize)
    cand_flag_cmd = sc_all.separation(sc_bin) < (1.0*binsize)
    cand_flag_bg_cmd = (sc_all.separation(sc_bin) < (2.0*binsize)) &\
                       (sc_all.separation(sc_bin) > np.sqrt(3)*binsize)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Color-magnitude diagram:
    plt.subplot(321)
    plt.scatter(gi[comp_flag & star_flag], imag[comp_flag & star_flag], s=1,
                c='Gray', alpha=.5, label='comparison')
    plt.scatter(gi[cand_flag_cmd & star_flag], imag[cand_flag_cmd & star_flag], s=40,
                color='DodgerBlue', label='candidate overdensity')
    plt.scatter(gi[cand_flag_cmd & star_flag & cmdsel_flag],
                imag[cand_flag_cmd & star_flag & cmdsel_flag], s=40,
                color='Red', label='RGB')

    # To do:
    # plt.plot(xcorners_blue, ycorners_blue, color='Gray')

    plt.legend(loc='upper left')
    plt.xlim(-1, 3)
    plt.ylim(26.8, 19.8)
    plt.xlabel(r'$(g-i)_0$')
    plt.ylabel(r'$i_0$')
    plt.minorticks_on()

    # Print the original bin center on the CMD:
    ra_str = str('{:7.3f}'.format(sc_inp.ra.value))
    dec_str = str('{:7.3f}'.format(sc_inp.dec.value))
    nsig_str = str('{:5.1f}'.format(nsig))
    plt.title(r', (RA, Dec)$_0$=('+ra_str+','+dec_str+'), Nsig='+nsig_str)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Spatial plot:
    plt.subplot(322)
    plt.plot(ra[cand_flag & ~star_flag], dec[cand_flag & ~star_flag], '.',
             ms=4, color='Gray', alpha=0.3, label='not stars')
    plt.plot(ra[cand_flag & star_flag], dec[cand_flag & star_flag], '.',
             ms=10, color='DodgerBlue', label='stars')
    plt.plot(ra[cand_flag & star_flag & cmdsel_flag], dec[cand_flag & star_flag & cmdsel_flag],
             '.', ms=10, color='Red', label='RGB stars')
    plt.xlabel('RA (deg)')
    plt.ylabel('DEC (deg)')
    plt.legend(loc='upper left')
    plt.gca().invert_xaxis()
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=5)
    plt.minorticks_on()

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 1D radial density plot
    nradbins = 10
    radbinsize = 2*binsize/nradbins
    radbins = np.linspace(0, binsize, nradbins+1)
    radbin_cens = (radbins[1:] + radbins[:-1]) / 2.0
    rad_density = np.zeros(nradbins)
    for irad in range(nradbins):
        area = 2*np.pi*(radbins[irad+1].value**2 - radbins[irad].value**2)
        annulus_mask = (sc_all.separation(sc_bin) >= radbins[irad]) &\
                       (sc_all.separation(sc_bin) < radbins[irad+1])
        rad_density[irad] = len(ra[annulus_mask & star_flag & cmdsel_flag]) / area

    plt.subplot(323)
    plt.plot(radbin_cens, rad_density, '-.')
    plt.plot(radbin_cens, rad_density, 'ko')
    plt.minorticks_on()

    # if recalc_cen:
    ra_str = str('{:7.3f}'.format(sc_bin.ra.value))
    dec_str = str('{:7.3f}'.format(sc_bin.dec.value))
    nsig_str = str('{:5.1f}'.format(nsig))
    plt.title(r', (RA, Dec)$_0$=('+ra_str+','+dec_str+'), Nsig='+nsig_str)

    plt.xlabel('Distance from center [arcmin]')
    plt.ylabel(r'Number of stars per arcsec$^2$')

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Color image:
    plt.subplot(324)
    censtar = np.where(sc_all.separation(sc_bin) == np.min(sc_all.separation(sc_bin)))
    # patch = np.char.strip(cat.dat[censtar]['patch'].data[0])

    img_path = '/Volumes/gimli/hsc_data/repo/HSC/calib/madcash/ngc247/20240708T115843Z/deepCoadd_calexp/'
    # deepCoadd_calexp_6147_43_g_hsc_rings_v1_HSC_calib_madcash_ngc247_20240708T115843Z.fits
    img_base = 'hsc_rings_v1_HSC_calib_madcash_ngc247_20240708T115843Z.fits'
    gimg_path = img_path+str(tract)+'/'+str(patch)+'/g/deepCoadd_calexp_'+str(tract)+'_'+str(patch)+'_g_'+img_base
    iimg_path = img_path+str(tract)+'/'+str(patch)+'/i/deepCoadd_calexp_'+str(tract)+'_'+str(patch)+'_i_'+img_base
    # iimg_path = img_path+'fakedwarfs_i/calexp-HSC-I2-0-'+str(patch)+'_fakes.fits'

    hdulist = fits.open(iimg_path)
    w = WCS(hdulist[1].header, hdulist)
    img_i = hdulist[1].data
    hdulist.close()

    hdulist = fits.open(gimg_path)
    w = WCS(hdulist[1].header, hdulist)
    img_g = hdulist[1].data
    pixscale = np.abs(hdulist[1].header['CD1_1']*3600.0)
    hdulist.close()

    imx, imy = w.all_world2pix([sc_all[censtar].ra.value], [sc_all[censtar].dec.value], 0)
    img_r = img_i

    position = (imx, imy)
    xy_pix_size = int(np.floor(2.0*binsize.to(u.arcsec).value/pixscale))
    size = (xy_pix_size, xy_pix_size)     # pixels

    cutout_g = Cutout2D(img_g, position, size)
    cutout_r_tmp = Cutout2D(img_r, position, size)
    cutout_i = Cutout2D(img_i, position, size)

    cutout_r = (cutout_r_tmp.data+cutout_g.data)/2

    zscale = ZScaleInterval()
    vmin_g, vmax_g = zscale.get_limits(cutout_g.data)
    vmin_r, vmax_r = zscale.get_limits(cutout_r.data)
    vmin_i, vmax_i = zscale.get_limits(cutout_i.data)

    img = np.zeros((cutout_i.shape[0], cutout_i.shape[1], 3), dtype=float)
    img[:, :, 0] = img_scale.linear(cutout_i.data, scale_min=0.1*vmin_i, scale_max=0.6*vmax_i)
    img[:, :, 1] = img_scale.linear(cutout_r.data, scale_min=0.1*vmin_r, scale_max=0.9*vmax_r)
    img[:, :, 2] = img_scale.linear(cutout_g.data, scale_min=0.1*vmin_g, scale_max=0.75*vmax_g)

    # Create RA, Dec ticks, labels:
    nticks = 4
    xtick_locs = np.arange(0, xy_pix_size, xy_pix_size/nticks) + xy_pix_size/(2*nticks)
    ytick_locs = np.arange(0, xy_pix_size, xy_pix_size/nticks) + xy_pix_size/(2*nticks)

    tick_locs_tmp = []
    for loc in xtick_locs:
        tick_locs_tmp.append(cutout_g.to_original_position((loc, 0)))

    xtick_locs_orig = []
    for loc in tick_locs_tmp:
        xtick_locs_orig.append(loc[0])

    tick_locs_tmp = []
    for loc in ytick_locs:
        tick_locs_tmp.append(cutout_g.to_original_position((0, loc)))

    ytick_locs_orig = []
    for loc in tick_locs_tmp:
        ytick_locs_orig.append(loc[1])

    xtick_lbls = []
    ytick_lbls = []
    for xt, yt in zip(xtick_locs_orig, ytick_locs_orig):
        temp = w.all_pix2world([[xt, yt]], 0)
        xtick_lbls.append(str(np.round(temp[0][0], 2)))
        ytick_lbls.append(str(np.round(temp[0][1], 2)))

    fig = plt.gca()
    fig.axes.xaxis.set_ticks(xtick_locs, labels=xtick_lbls)
    fig.axes.yaxis.set_ticks(ytick_locs, labels=ytick_lbls)
    fig.axes.tick_params(direction='out')
    fig.axes.get_xaxis().set_visible(True)
    fig.axes.get_yaxis().set_visible(True)

    linelength_arcsec = 10.0
    linelength_pix = linelength_arcsec/pixscale
    fig.hlines(60, 70, 70+linelength_pix, color='White')
    fig.text(30, 85, '10 arcsec', color='White')

    fig.imshow(img, aspect='equal', origin='lower', extent=[0, xy_pix_size, 0, xy_pix_size])

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Luminosity function:
    plt.subplot(325)

    lf_binsize = 0.25
    lf_minmag = 21.25
    lf_maxmag = 26.5
    lf_bins = np.arange(lf_minmag, lf_maxmag, lf_binsize)
    imag_cand_rgb = imag[cand_flag_cmd & star_flag & cmdsel_flag]
    imag_cand_bg_rgb = imag[cand_flag_bg_cmd & star_flag & cmdsel_flag]
    imag_cand_all = imag[cand_flag & ~cand_flag_cmd & star_flag]

    lf_mag_all, lf_hist_bins = np.histogram(imag_cand_all, bins=lf_bins)
    lf_bin_cens = lf_hist_bins + lf_binsize/2.0
    lf_mag_rgb, lf_hist_bins = np.histogram(imag_cand_rgb, bins=lf_bins)
    lf_mag_bg_rgb, lf_hist_bins = np.histogram(imag_cand_bg_rgb, bins=lf_bins)

    #plt.plot(lf_bin_cens[:-1], np.cumsum(lf_mag_all), '-.', color='DodgerBlue', label='all stars')
    #plt.plot(lf_bin_cens[:-1], np.cumsum(lf_mag_all), 'ko')
    plt.plot(lf_bin_cens[:-1], np.cumsum(lf_mag_rgb), '-.', color='Red', label='cmd selected')
    plt.plot(lf_bin_cens[:-1], np.cumsum(lf_mag_rgb), 'ko')
    plt.plot(lf_bin_cens[:-1], np.cumsum(lf_mag_bg_rgb), '-.', color='Gray', label='background w/ cmd selection')
    plt.plot(lf_bin_cens[:-1], np.cumsum(lf_mag_bg_rgb), 'ko')
    #plt.plot(lf_bin_cens[:-1], np.cumsum(lf_mag_rgb)/np.cumsum(lf_mag_all), '-.', color='DodgerBlue')
    plt.legend()
    plt.minorticks_on()
    plt.semilogy()

    #def lf_powerlaw(mags, lf_inp, alpha=2.35):
    #    # dN = Phi*m^(-alpha), where "m" is the magnitude
    #    # Want dN = value at mag25bin, so Phi = dN*m^(alpha) = dN*25^(alpha)
    #    mag25bin = np.argmin(np.abs(mags - 25.0))
    #    phi_salpeter = np.sum(lf_inp[:mag25bin])  # *(25.0**(-1*alpha))
    #    return phi_salpeter*(10.0**((6.0*alpha/5.0) * (mags-25.0)))

    # print(phi_salpeter)
    # lf_salpeter = phi_salpeter * (lf_bin_cens**(-2.35))

    lf_salpeter = lf_powerlaw(lf_bin_cens, lf_mag_rgb, alpha=2.35)
    #plt.plot(lf_bin_cens[:-1], lf_salpeter[:-1], ':', color='Gray')

    lf_geha = lf_powerlaw(lf_bin_cens, lf_mag_rgb, alpha=1.30)
    #plt.plot(lf_bin_cens[:-1], lf_geha[:-1], '--', color='Gray')

    # plt.title('luminosity function')
    plt.ylim(8*10**(-1), 1.15*np.sum(lf_mag_rgb))

    plt.xlabel('imag')
    plt.ylabel(r'$N_{*, cand} <$ imag')

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Color image:
    plt.subplot(326)

    fig = plt.gca()
    fig.axes.xaxis.set_ticks(xtick_locs, labels=xtick_lbls)
    fig.axes.yaxis.set_ticks(ytick_locs, labels=ytick_lbls)
    fig.axes.tick_params(direction='out')
    fig.axes.get_xaxis().set_visible(True)
    fig.axes.get_yaxis().set_visible(True)

    x_rgb, y_rgb = w.all_world2pix(ra[cand_flag & star_flag & cmdsel_flag],
                                   dec[cand_flag & star_flag & cmdsel_flag], 0)

    if overlay_pts:
        fig.plot(x_rgb-cutout_i.xmin_original, y_rgb-cutout_i.ymin_original, 'ws', ms=10,
                 markeredgewidth=1, fillstyle='none')

    fig.imshow(img, aspect='equal', origin='lower', extent=[0, xy_pix_size, 0, xy_pix_size])

    if savefig:
        savefile = 'cand_diagnostic_plots_6panel_'+name_append+'.png'
        plt.savefig(savefile, dpi=180)
        print('Saved figure to ', savefile, '. Rename to avoid overwriting.')
    else:
        plt.show()

    plt.close()

'''
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Smoothed image, g-band:
    plt.subplot(325)

    fig = plt.gca()
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.xaxis.set_ticklabels([])
    fig.axes.get_yaxis().set_visible(False)
    fig.axes.yaxis.set_ticklabels([])

    # PLOT A SMOOTHED G-BAND IMAGE:
    # Create kernel
    gkernel = Gaussian2DKernel(x_stddev=3.0)
    # Convolve data
    z = convolve(img[:, :, 2], gkernel)
    # For blue RGB box:
    minval = 0
    maxval = 0.5  # 7
    plt.imshow(z, aspect='equal', origin='lower', vmin=minval, vmax=maxval,
               extent=[0, xy_pix_size, 0, xy_pix_size], cmap='gray')

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Smoothed image, i-band:
    plt.subplot(326)

    fig = plt.gca()
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.xaxis.set_ticklabels([])
    fig.axes.get_yaxis().set_visible(False)
    fig.axes.yaxis.set_ticklabels([])

    # PLOT A SMOOTHED I-BAND IMAGE:
    # Create kernel
    gkernel = Gaussian2DKernel(x_stddev=3.0)
    # Convolve data
    z = convolve(img[:, :, 0], gkernel)
    # For blue RGB box:
    minval = 0
    maxval = 0.5  # 7
    plt.imshow(z, aspect='equal', origin='lower', vmin=minval, vmax=maxval,
               extent=[0, xy_pix_size, 0, xy_pix_size], cmap='gray')

    if savefig:
        savefile = 'cand_diagnostic_plots_6panel_'+name_append+'.png'
        plt.savefig(savefile, dpi=180)
        print('Saved figure to ', savefile, '. Rename to avoid overwriting.')

    plt.show()
    plt.close()
'''
