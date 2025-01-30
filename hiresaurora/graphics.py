import warnings
from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from astropy.io import fits

from hiresaurora.general import rcparams, FuzzyQuantity

plt.style.use(rcparams)


def contour_rect_slow(im) -> list[tuple[float, float, float, float]]:
    """
    Code to calculate pixel outlines.

    Code copied from https://stackoverflow.com/questions/40892203/
    can-matplotlib-contours-match-pixel-edges.
    """
    pad = np.pad(im, [(1, 1), (1, 1)])  # noqa
    im0 = np.abs(np.diff(pad, n=1, axis=0))[:, 1:]
    im1 = np.abs(np.diff(pad, n=1, axis=1))[1:, :]
    lines = []
    for ii, jj in np.ndindex(im0.shape):
        if im0[ii, jj] == 1:
            lines += [([ii-0.5, ii-0.5], [jj-0.5, jj+0.5])]
        if im1[ii, jj] == 1:
            lines += [([ii-0.5, ii+0.5], [jj-0.5, jj-0.5])]
    return lines


def _place_colorbar(img: plt.cm.ScalarMappable,
                    cax: plt.Axes) -> None:
    """
    Place a colorbar and label it with units.
    """
    plt.colorbar(img, cax=cax, label=r'R nm$^{-1}$')
    cax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 3))
    cax.yaxis.set_major_locator(ticker.AutoLocator())
    cax.yaxis.set_minor_locator(ticker.AutoMinorLocator())


def _place_label(label: str,
                 axis: plt.Axes) -> None:
    """
    Place a label in the corner of an axis.
    """
    bbox = dict(facecolor=(1, 1, 1, 0.825), edgecolor='black')
    axis.annotate(label, xy=(0, 1), xytext=(7, -7), xycoords='axes fraction',
                  textcoords='offset points', ha='left', va='top',
                  color='black', bbox=bbox)


# noinspection DuplicatedCode
def make_quicklook(file_path: Path) -> None:
    """
    Make a calibrated data quicklook for a given data file.
    """

    savename = Path(str(file_path).replace('.fits.gz', '.pdf'))

    with fits.open(file_path) as hdul:
        data = hdul['PRIMARY'].data
        background = hdul['BACKGROUND_FIT'].data + hdul['SKYLINE_FIT'].data
        bgsub_data = data - background
        edges = hdul['APERTURE_EDGES'].data
        spascale = hdul['PRIMARY'].header['SPASCALE']
        spescale = hdul['PRIMARY'].header['SPESCALE']
        brightness = hdul['PRIMARY'].header['BRGHT']
        uncertainty = hdul['PRIMARY'].header['BRGHTUNC']
        fuzz = FuzzyQuantity(brightness, uncertainty)

        cmap = plt.get_cmap('viridis')
        bottom = 0.3948
        top = 0.201
        middle = 0.294
        left = 0.6
        right = 0.75
        n_spa, n_spe = data.shape
        width = 6
        subplot_width = 6 - left - right
        subplot_height = subplot_width * n_spa / n_spe * spascale / spescale
        lineplot_scale = 3
        height = (3 + lineplot_scale) * subplot_height + top + bottom + 3 * middle

        fig = plt.figure(figsize=(width, height))
        ax1 = plt.axes(
            (left / width, (bottom + 3 * middle + (lineplot_scale+2) * subplot_height) / height,
             subplot_width / width, subplot_height / height))
        ax2 = plt.axes(
            (left / width, (bottom + 2 * middle + (lineplot_scale+1) * subplot_height) / height,
             subplot_width / width, subplot_height / height),
            sharex=ax1)
        ax3 = plt.axes(
            (left / width, (bottom + middle + lineplot_scale * subplot_height) / height,
             subplot_width / width, subplot_height / height),
            sharex=ax1)
        ax4 = plt.axes((left/width, bottom/height, subplot_width/width,
                        lineplot_scale*subplot_height/height), sharex=ax1)

        axes = [ax1, ax2, ax3, ax4]

        cax1 = plt.axes(((left + subplot_width+0.1) / width,
                         (bottom+2*middle+(lineplot_scale+1)*subplot_height) / height,
                         0.1 / width, (2*subplot_height+middle) / height))
        cax2 = plt.axes(((left + subplot_width + 0.1) / width,
                         (bottom + middle + lineplot_scale*subplot_height) / height,
                         0.1 / width, subplot_height / height))

        [axis.xaxis.set_major_locator(ticker.NullLocator())
         for axis in axes[:-1]]
        [axis.xaxis.set_minor_locator(ticker.NullLocator())
         for axis in axes[:-1]]
        [axis.yaxis.set_major_locator(ticker.NullLocator())
         for axis in axes[:-1]]
        [axis.yaxis.set_minor_locator(ticker.NullLocator())
         for axis in axes[:-1]]

        x = np.arange(data.shape[1]) * spescale
        pixel2wavelength = np.poly1d(np.polyfit(
            x, hdul['WAVELENGTH_CENTERS_SHIFTED'].data, deg=3))
        wavelength2pixel = np.poly1d(np.polyfit(
            hdul['WAVELENGTH_CENTERS_SHIFTED'].data, x, deg=3))
        wavelength_axis = axes[-1].secondary_xaxis(
            'bottom', functions=(pixel2wavelength, wavelength2pixel))
        # wavelength_axis.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        wavelength_axis.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        wavelength_axis.set_xlabel('Doppler-Shifted Wavelength [nm]')

        emission_lines = hdul['TARGETED_LINES'].data
        if len(emission_lines) > 1:
            end = 's'
        else:
            end = ''

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            norm = colors.Normalize(vmin=np.nanpercentile(data, 1),
                                    vmax=np.nanpercentile(data, 99.9))

        x, y = np.meshgrid(np.arange(data.shape[1]) * spescale,
                           np.arange(data.shape[0]) * spascale)
        img0 = axes[0].pcolormesh(x, y, data, norm=norm, cmap=cmap,
                                  rasterized=True)
        _place_colorbar(img=img0, cax=cax1)
        axes[0].set_title('Calibrated Data')

        axes[1].pcolormesh(x, y, background, norm=norm, cmap=cmap,
                           rasterized=True)
        axes[1].set_title('Background Fit')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            norm = colors.Normalize(vmin=np.nanpercentile(bgsub_data, 1),
                                    vmax=np.nanpercentile(bgsub_data, 99.9))
        img2 = axes[2].pcolormesh(x, y, bgsub_data, norm=norm, cmap=cmap,
                                  rasterized=True)
        _place_colorbar(img=img2, cax=cax2)
        axes[2].set_title(f'Background-Subtracted Data w/ Aperture{end}')
        lines = contour_rect_slow(edges)
        for line in lines:
            axes[2].plot(np.array(line[1]) * spescale,
                         np.array(line[0]) * spascale,
                         color='#D62728')

        axes[3].plot(np.arange(data.shape[1]) * spescale,
                     hdul['SPECTRUM_1D'].data, color='#C9C9C9', zorder=2)
        axes[3].plot(np.arange(data.shape[1]) * spescale,
                     hdul['BEST_FIT_1D'].data, color='#D62728', zorder=4)
        axes[3].fill_between(
            np.arange(data.shape[1]) * spescale,
            hdul['BEST_FIT_1D'].data - hdul['BEST_FIT_1D_unc'].data,
            hdul['BEST_FIT_1D'].data + hdul['BEST_FIT_1D_unc'].data,
            color='#D62728', alpha=0.5, linewidth=0, zorder=3)
        label = fr'Best-fit brightness: ${fuzz.latex}$ R'
        axes[3].annotate(label, xy=(0, 1), xytext=(3, -3),
                         xycoords='axes fraction', textcoords='offset points',
                         ha='left', va='top')
        axes[3].set_ylabel(r'R nm$^{-1}$')
        axes[3].ticklabel_format(axis='y', style='sci', scilimits=(-2, 3))

        [axis.set_aspect('equal') for axis in axes[:3]]

        plt.savefig(savename)
        plt.close(fig)
