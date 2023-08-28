import warnings
from pathlib import Path

import astropy.units as u
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from astropy.io import fits
from lmfit.model import ModelResult
from lmfit.models import PolynomialModel, GaussianModel, ConstantModel
from matplotlib.collections import QuadMesh
from astropy.time import Time
from datetime import datetime
from matplotlib.patches import Circle

from hiresaurora.ephemeris import _get_ephemeris
from hiresaurora.general import rcparams, AuroraLines, FuzzyQuantity

plt.style.use(rcparams)


def contour_rect_slow(im):
    """
    Code copied from https://stackoverflow.com/questions/40892203/
    can-matplotlib-contours-match-pixel-edges
    """

    pad = np.pad(im, [(1, 1), (1, 1)])
    im0 = np.abs(np.diff(pad, n=1, axis=0))[:, 1:]
    im1 = np.abs(np.diff(pad, n=1, axis=1))[1:, :]
    lines = []
    for ii, jj in np.ndindex(im0.shape):
        if im0[ii, jj] == 1:
            lines += [([ii - .5, ii - .5], [jj - .5, jj + .5])]
        if im1[ii, jj] == 1:
            lines += [([ii - .5, ii + .5], [jj - .5, jj - .5])]
    return lines


def _place_colorbar(img: plt.cm.ScalarMappable, unit: str, axis: plt.Axes):
    """
    Place a colorbar and label it with units.
    """
    plt.colorbar(img, ax=axis, label=unit, pad=0.0125, aspect=15)


def _place_label(label: str, axis: plt.Axes):
    """
    Place a label in the corner of an axis.
    """
    bbox = dict(facecolor=(1, 1, 1, 0.825), edgecolor='black')
    axis.annotate(label, xy=(0, 1), xytext=(7, -7), xycoords='axes fraction',
                  textcoords='offset points', ha='left', va='top',
                  color='black', bbox=bbox)


def make_quicklook(file_path: Path):
    """
    Make a calibrated data quicklook for a given data file.
    """

    savename = Path(str(file_path).replace('.fits.gz', '.jpg'))

    with fits.open(file_path) as hdul:
        header = hdul['PRIMARY'].header
        raw_data = hdul['RAW'].data
        background = hdul['BACKGROUND_FIT'].data
        edges = hdul['APERTURE_EDGES'].data
        calibrated_data = hdul['CALIBRATED'].data
        calibrated_unc = hdul['CALIBRATED_UNC'].data
        snr = calibrated_data / calibrated_unc

        cmap = plt.get_cmap('viridis')
        dunit = 0.06
        aspect_ratio = header['SPASCALE'] / header['SPESCALE']
        n_spa, n_spe = raw_data.shape
        height = dunit * n_spa * 4
        width = dunit * n_spe / aspect_ratio
        height += 4 * dunit
        width += 2 * dunit
        fig, axes = plt.subplots(4, figsize=(width, height),
                                 layout='constrained', clear=True)
        [axis.xaxis.set_major_locator(ticker.NullLocator()) for axis in axes]
        [axis.xaxis.set_minor_locator(ticker.NullLocator()) for axis in axes]
        [axis.yaxis.set_major_locator(ticker.NullLocator()) for axis in axes]
        [axis.yaxis.set_minor_locator(ticker.NullLocator()) for axis in axes]

        emission_lines = hdul['TARGETED_LINES'].data
        if len(emission_lines) > 1:
            end = 's'
        else:
            end = ''

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            norm = colors.Normalize(vmin=np.nanpercentile(raw_data, 1),
                                    vmax=np.nanpercentile(raw_data, 99.9))
        spascale = hdul['PRIMARY'].header['SPASCALE']
        spescale = hdul['PRIMARY'].header['SPESCALE']
        x, y = np.meshgrid(np.arange(raw_data.shape[1]) * spescale,
                           np.arange(raw_data.shape[0]) * spascale)
        img0 = axes[0].pcolormesh(x, y, raw_data, norm=norm, cmap=cmap)
        _place_colorbar(img=img0, unit=hdul['RAW'].header['BUNIT'],
                        axis=axes[0])
        _place_label(label='Raw Data', axis=axes[0])
        img1 = axes[1].pcolormesh(x, y, background, norm=norm, cmap=cmap)
        _place_colorbar(img=img1, unit=hdul['BACKGROUND_FIT'].header['BUNIT'],
                        axis=axes[1])
        _place_label(label='Background Fit', axis=axes[1])

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            norm = colors.Normalize(vmin=np.nanpercentile(snr, 1),
                                    vmax=np.nanpercentile(snr, 99.9))
        img2 = axes[2].pcolormesh(x, y, snr, norm=norm, cmap=cmap)
        _place_colorbar(img=img2, unit='Ratio',
                        axis=axes[2])
        _place_label(label=f'Signal-to-Noise Ratio w/ Aperture{end}',
                     axis=axes[2])
        lines = contour_rect_slow(edges)
        for line in lines:
            axes[2].plot(np.array(line[1]) * spescale,
                         np.array(line[0]) * spascale,
                         color='red')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            norm = colors.Normalize(
                vmin=np.nanpercentile(calibrated_data, 1),
                vmax=np.nanpercentile(calibrated_data, 99.9))
        img3 = axes[3].pcolormesh(x, y, calibrated_data, norm=norm, cmap=cmap)
        _place_colorbar(img=img3, unit=hdul['CALIBRATED'].header['BUNIT'],
                        axis=axes[3])
        _place_label(label=f'Calibrated Data', axis=axes[3])

        plt.savefig(savename)
        plt.close(fig)


class PostageStamps:
    """
    Class to create a postage-stamp-style quicklook for an observing night.
    """
    def __init__(self, calibrated_data_path: str or Path, extended: bool,
                 excluded: [bool], boundary_scale: float = 2.5):
        """
        Parameters
        ----------
        calibrated_data_path : str or Path
            The file path to the directory containing the calibrated data in
            wavelength subdirectories.
        extended : bool
            Whether or not the aurora pipeline used the extended Io line list.
        excluded : [bool]
            List of booleans indicating which observations were excluded from
            averaging.
        boundary_scale : float
            Bounds of each image in target radii. Default is ±2.5.
        """
        self._calibrated_data_path = Path(calibrated_data_path)
        self._extended = extended
        self._excluded = np.array(excluded).astype(bool)
        self._boundary_scale = boundary_scale
        self._n_obs = self._get_n_obs()
        self._aurora_lines = AuroraLines(extended=extended)

    @staticmethod
    def _find_trace(trace_data: np.ndarray):
        """
        Calculate the vertical fractional pixel number of the trace.
        """
        model = GaussianModel()
        nspa, nspe = trace_data.shape
        x = np.arange(nspa)
        centers = []
        centers_unc = []
        for col in range(nspe):
            data = trace_data[:, col]
            params = model.guess(data, x=x)
            fit = model.fit(data, params, x=x)
            centers.append(fit.params['center'].value)
            centers_unc.append(fit.params['center'].stderr)
        centers = np.array(centers)
        centers_unc = np.array(centers_unc)
        x = np.arange(nspe)
        model = ConstantModel()
        params = model.guess(centers, x=x)
        fit = model.fit(centers, params, weights=1/centers_unc**2, x=x)
        return fit.params['c'].value

    def _get_n_obs(self) -> int:
        """
        Check to see how many 630.0 nm files there are.
        """
        subpath = Path(self._calibrated_data_path, '[O I] 630.0 nm')
        return len(sorted(subpath.glob('*.fits.gz')))

    def _make_figure_and_axes(self) -> (plt.Figure, [plt.Axes]):
        """
        Make figure and axes grid and set basic parameters.
        """
        subplot_sizes = self._calculate_subplot_sizes()
        width = np.sum(subplot_sizes)
        fig, axes = plt.subplots(self._n_obs, len(self._aurora_lines.names),
                                 figsize=(width, self._n_obs), sharex='col',
                                 layout='constrained',
                                 gridspec_kw={'width_ratios': subplot_sizes})
        [axis.set_xticks([]) for axis in axes.ravel()]
        [axis.set_yticks([]) for axis in axes.ravel()]
        [axis.set_facecolor((0.75, 0.75, 0.75)) for axis in axes.ravel()]
        return fig, axes

    @staticmethod
    def _get_angular_radius(data_header: fits.Header) -> float:
        """
        Retrieve apparent angular radius of target satellite in arcseconds.
        """
        ephemeris = _get_ephemeris(target=data_header['OBJECT'],
                                   time=data_header['DATE-OBS'])
        radius = ephemeris['ang_width'].value[0] / 2
        unit = ephemeris['ang_width'].unit
        return (radius * unit).to(u.arcsec).value

    @staticmethod
    def _get_central_wavelength(targeted_lines: np.ndarray) -> float:
        """
        Find the center of the line wavelength range.
        """
        return np.mean([targeted_lines[0], targeted_lines[-1]]).squeeze()

    def _get_y0(self, hdul: fits.HDUList) -> float:
        try:
            return self._find_trace(hdul['TRACE'].data)
        except KeyError:
            return hdul['CALIBRATED'].data.shape[0] / 2

    @staticmethod
    def _calculate_inverse_wavelength_solution(
            wavelengths: np.ndarray) -> ModelResult:
        """
        Calculate 3rd-degree polynomial inverse wavelength solution, mapping
        wavelengths to fractional pixels.
        """
        model = PolynomialModel(degree=3)
        pixels = np.arange(wavelengths.shape[0])
        params = model.guess(pixels, x=wavelengths)
        return model.fit(pixels, params, x=wavelengths)

    @staticmethod
    def _make_meshgrids(hdul: fits.HDUList,
                        x0: float, y0: float) -> (np.ndarray, np.ndarray):
        """
        Make angular meshgrids for proper data display.
        """
        spescale = hdul['PRIMARY'].header['SPESCALE']
        spascale = hdul['PRIMARY'].header['SPASCALE']
        nspa, nspe = hdul['CALIBRATED'].data.shape
        return np.meshgrid((np.arange(nspe) - x0) * spescale,
                           (np.arange(nspa) - y0) * spascale)

    def _calculate_horizontal_bounds(
            self, inverse_wavelength_solution: ModelResult,
            targeted_lines: np.ndarray, radius: float,
            spescale: float, x0: float) -> (float, float):
        """
        Calculate display bounds to a given radius factor.
        """
        xmins = np.ones(
            targeted_lines.shape[0]+1) * -self._boundary_scale * radius
        xmaxs = np.ones(
            targeted_lines.shape[0]+1) * self._boundary_scale * radius
        for i, line in enumerate(targeted_lines):
            x = (inverse_wavelength_solution.eval(x=line) - x0) * spescale
            xmins[i+1] = x - self._boundary_scale * radius
            xmaxs[i+1] = x + self._boundary_scale * radius
        return xmins.min(), xmaxs.max()

    @staticmethod
    def _add_target_outlines(
            axis: plt.Axes, inverse_wavelength_solution: ModelResult,
            targeted_lines: np.ndarray, radius: float, spescale: float,
            x0: float) -> None:
        """
        Add a circular white outline of the target satellite's angular size.
        """
        for line in targeted_lines:
            x = (inverse_wavelength_solution.eval(x=line) - x0) * spescale
            outline = Circle(xy=(x, 0), radius=radius, edgecolor='white',
                             facecolor='none', linestyle='--', linewidth=0.5)
            axis.add_patch(outline)

    @staticmethod
    def _place_stamp_labels(axis: plt.Axes, hdul: fits.HDUList) -> None:
        """
        Place the brightness and uncertainty in the upper left corner. For the
        reported uncertainty, us the smaller of the propagated uncertainty or
        the image standard deviation.
        """
        unit = hdul['CALIBRATED'].header['BUNIT']
        unc = np.min([hdul['PRIMARY'].header['BGHT_UNC'],
                      hdul['PRIMARY'].header['BGHT_STD']])
        fuzz = FuzzyQuantity(hdul['PRIMARY'].header['BGHTNESS'], unc)
        val, unc = fuzz.value, fuzz.uncertainty
        axis.annotate(f'{val} ± {unc} {unit}'.replace("-", "–"), xy=(0, 1),
                      xytext=(4, -4), xycoords='axes fraction',
                      textcoords='offset points', ha='left', va='top',
                      color='white', fontweight='bold')

    # noinspection DuplicatedCode
    def _calculate_subplot_sizes(self) -> np.ndarray:
        """
        Make a signal-to-noise plot for each line to get a quick summary of
        detections.
        """
        sizes = np.ones(len(self._aurora_lines.names))
        for col, directory in enumerate(self._aurora_lines.names):
            filepath = Path(self._calibrated_data_path, directory)
            try:
                file = sorted(filepath.glob('*.fits.gz'))[0]
                with fits.open(file) as hdul:
                    targeted_lines = hdul['TARGETED_LINES'].data
                    primary_header = hdul['PRIMARY'].header
                    data_header = hdul['CALIBRATED'].header
                    wavelengths = hdul['WAVELENGTH_CENTERS'].data

                    radius = self._get_angular_radius(
                        data_header=data_header)
                    fit = self._calculate_inverse_wavelength_solution(
                        wavelengths=wavelengths)

                    center_wavelength = self._get_central_wavelength(
                        targeted_lines=targeted_lines)
                    x0 = fit.eval(x=center_wavelength)

                    xmin, xmax = self._calculate_horizontal_bounds(
                        inverse_wavelength_solution=fit,
                        targeted_lines=targeted_lines, radius=radius,
                        spescale=primary_header['SPESCALE'], x0=x0)
                    sizes[col] = xmax - xmin
            except IndexError:
                sizes[col] = sizes[0]
        return sizes / sizes[0]

    @staticmethod
    def _rescale_images(images: [QuadMesh]) -> None:
        """
        Rescale all images for a given wavelength to the same scale after
        they are all displayed.
        """
        images = np.array(images)
        vmins = []
        vmaxes = []
        for img in images[:-1]:  # skip average in rescaling
            vmin, vmax = img.get_clim()
            vmins.append(vmin)
            vmaxes.append(vmax)
        vmin = np.min(vmins)
        vmax = np.max(vmaxes)
        for img in images[:-1]:
            img.set_clim(vmin, vmax)

    @staticmethod
    def _make_time_label(time: str, fmt: str = '%H:%M:%S'):
        """
        Convert date to datetime string using supplied format.
        """
        dt = Time(time, format='isot').to_datetime()
        return datetime.strftime(dt, fmt)

    @staticmethod
    def _place_wavelength_ticks(
            axis: plt.Axes, inverse_wavelength_solution: ModelResult,
            targeted_lines: np.ndarray, x0: float, spescale: float) -> None:
        """
        Place ticks at locations of line wavelength(s).
        """
        ticks = inverse_wavelength_solution.eval(x=targeted_lines) - x0
        ticks *= spescale
        tick_labels = np.array([f'{i:.4f}' for i in targeted_lines])
        axis.set_xticks(ticks)
        axis.xaxis.set_minor_locator(ticker.NullLocator())
        axis.set_xticklabels(tick_labels, rotation=90)

    @staticmethod
    def _indicate_missing(axes: [plt.Axes]):
        """
        Place a "No data" label if a particular wavelength in the set is
        missing.
        """
        for axis in axes:
            axis.annotate('No data', xy=(0.5, 0.5), xycoords='axes fraction',
                          ha='center', va='center', color='grey')

    def create(self):
        """
        Make a signal-to-noise plot for each line to get a quick summary of
        detections.
        """
        n_cols = len(self._aurora_lines.names)
        fig, axes = self._make_figure_and_axes()

        for col, directory in enumerate(self._aurora_lines.names):
            filepath = Path(self._calibrated_data_path, directory)
            files = sorted(filepath.glob('*.fits.gz'))
            imgs = []
            for row, file in enumerate(files):
                try:
                    axis = axes[row, col]
                    with fits.open(file) as hdul:
                        targeted_lines = hdul['TARGETED_LINES'].data
                        primary_header = hdul['PRIMARY'].header
                        data_header = hdul['CALIBRATED'].header
                        calibrated_data = hdul['CALIBRATED'].data
                        wavelengths = hdul['WAVELENGTH_CENTERS'].data

                        radius = self._get_angular_radius(
                            data_header=data_header)
                        fit = self._calculate_inverse_wavelength_solution(
                            wavelengths=wavelengths)

                        center_wavelength = self._get_central_wavelength(
                            targeted_lines=targeted_lines)
                        x0 = fit.eval(x=center_wavelength)
                        y0 = self._get_y0(hdul)

                        xmesh, ymesh = self._make_meshgrids(
                            hdul=hdul, x0=x0, y0=y0)

                        xmin, xmax = self._calculate_horizontal_bounds(
                            inverse_wavelength_solution=fit,
                            targeted_lines=targeted_lines, radius=radius,
                            spescale=primary_header['SPESCALE'], x0=x0)
                        ymin = -self._boundary_scale * radius
                        ymax = self._boundary_scale * radius

                        norm = colors.Normalize(
                            vmin=np.nanpercentile(calibrated_data, 1),
                            vmax=np.nanpercentile(calibrated_data, 99.9))

                        img = axes[row, col].pcolormesh(
                            xmesh, ymesh, calibrated_data, norm=norm,
                            rasterized=True)
                        imgs.append(img)

                        self._add_target_outlines(
                            axis=axis, inverse_wavelength_solution=fit,
                            targeted_lines=targeted_lines, radius=radius,
                            spescale=primary_header['SPESCALE'], x0=x0)

                        self._place_stamp_labels(axis=axis, hdul=hdul)

                        axis.set_xlim(xmin, xmax)
                        axis.set_ylim(ymin, ymax)
                        axis.invert_xaxis()
                        axis.invert_yaxis()
                        axis.set_aspect('equal')
                        if row == 0:
                            axis.set_title(directory)
                        if (col == 0) and (file.name != 'average.fits.gz'):
                            axis.set_ylabel(
                                self._make_time_label(data_header['DATE-OBS']))
                        if (col == n_cols-1) \
                                and (file.name != 'average.fits.gz') \
                                and (self._excluded[row]):
                            axis.yaxis.set_label_position('right')
                            axis.set_ylabel('(excluded)')
                        elif (col == 0) and (file.name == 'average.fits.gz'):
                            axis.set_ylabel('Average')
                        if (row == 0) & (col == 0):
                            date = self._make_time_label(
                                data_header['DATE-OBS'], fmt='%B %d, %Y [UTC]')
                            fig.supylabel(f'Start of Observation on {date}')
                        if row == self._n_obs-1:
                            self._place_wavelength_ticks(
                                axis=axis, inverse_wavelength_solution=fit,
                                targeted_lines=targeted_lines, x0=x0,
                                spescale=primary_header['SPESCALE'])
                except ValueError:
                    continue
            try:
                self._rescale_images(images=imgs)
            except ValueError:
                self._indicate_missing(axes=axes[:, col])

        plt.savefig(Path(self._calibrated_data_path.parent, 'results.jpg'))
