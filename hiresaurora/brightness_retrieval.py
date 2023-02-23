import warnings
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import numpy as np
from astropy.io import fits
from astropy.time import Time
from lmfit import Parameters
from lmfit.model import ModelResult
from lmfit.models import ConstantModel, GaussianModel
from astropy.convolution import convolve, Gaussian2DKernel

from hirespipeline.files import make_directory
from hiresaurora.background_subtraction import _Background
from hiresaurora.alignment import _TraceOffsets
from hiresaurora.calibration import _FluxCalibration
from hiresaurora.ephemeris import _get_ephemeris
from hiresaurora.general import _doppler_shift_wavelengths, rcparams, \
    color_dict, emission_line_strengths, aurora_line_wavelengths, \
    aurora_line_names, format_uncertainty

plt.style.use(rcparams)


class WavelengthNotFoundError(Exception):
    """
    Raised if a wavelength isn't found in available solutions.
    """
    pass


class _Retrieval:
    """
    Retrieve both Gaussian-fitted and observed aurora brightnesses.
    """
    def __init__(self, reduced_data_directory: str or Path):
        self._reduced_data_directory = Path(reduced_data_directory)
        self._retrieval_data_directory = make_directory(
            Path(self._reduced_data_directory.parent, 'brightness_retrievals'))

    def _find_order_with_wavelength(self, wavelengths: u.Quantity) -> int:
        """
        Find which order index contains a user-supplied wavelength
        """
        average_wavelength = wavelengths.to(u.nm).mean().value
        files = sorted(
            Path(self._reduced_data_directory, 'science').glob('*.fits.gz'))
        found_order = None
        with fits.open(files[0]) as hdul:
            wavelengths = hdul['BIN_CENTER_WAVELENGTHS'].data
            for order in range(wavelengths.shape[0]):
                ind = np.abs(wavelengths[order] - average_wavelength).argmin()
                if (ind > 0) & (ind < wavelengths.shape[1] - 1):
                    found_order = order
                    break
        return found_order

    @staticmethod
    def _get_relative_velocity(header: dict) -> u.Quantity:
        """
        Query the JPL Horizons ephemeris tool to get the velocity of the target
        relative to Earth.
        """
        target = header['OBJECT']
        time = Time(header['DATE-OBS'], format='isot', scale='utc')
        ephemeris = _get_ephemeris(target=target, time=time,
                                   skip_daylight=False, airmass_lessthan=None)
        return ephemeris['delta_rate'].value[0] * ephemeris['delta_rate'].unit

    @staticmethod
    def _get_slit_profile(data: np.ndarray):
        """
        Create a normalized slit profile by averaging across all columns which
        don't have mask NaNs.
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            profile = np.nanmean(data, axis=1)
        return profile

    def _get_data(self, wavelengths: u.Quantity,
                  trim_bottom: int, trim_top: int, horizontal_offset: int,
                  dwavelength: u.Quantity = 0.1 * u.nm) -> dict:
        """
        Get data, uncertainty, headers and wavelengths from the order which
        contains a user-supplied wavelength. Selects data from ± 0.1 nm around
        the target wavelength.
        """
        files = sorted(
            Path(self._reduced_data_directory, 'science').glob('*.fits.gz'))
        trace_files = sorted(
            Path(self._reduced_data_directory,
                 'guide_satellite').glob('*.fits.gz'))
        order = self._find_order_with_wavelength(wavelengths=wavelengths)
        wavelength = wavelengths.to(u.nm).value
        dwavelength = dwavelength.to(u.nm).mean().value
        if order is None:
            raise WavelengthNotFoundError('Wavelength not found!')
        headers = []
        data = []
        background_profiles = []
        traces = []
        uncertainty = []
        wavelength_centers = None
        wavelength_edges = None
        left = None
        right = None
        locs = None
        filenames = []
        for file in files:
            with fits.open(file) as hdul:
                headers.append(hdul['PRIMARY'].header)
                filenames.append(file.name)
                if left is None:
                    velocity = self._get_relative_velocity(headers[-1])
                    wavelengths = _doppler_shift_wavelengths(
                        hdul['BIN_CENTER_WAVELENGTHS'].data * u.nm,
                        velocity=velocity).value
                    left = np.abs(
                        wavelengths[order] - (wavelength.min() - dwavelength)
                    ).argmin()
                    right = np.abs(
                        wavelengths[order] - (wavelength.max() + dwavelength)
                    ).argmin()
                    locs = [np.abs(wavelengths[order] - w).argmin() - left
                            for w in wavelength]
                select = np.s_[order, trim_bottom:-trim_top,
                               left+horizontal_offset:right+horizontal_offset]
                data.append(hdul['PRIMARY'].data[select])
                uncertainty.append(hdul['PRIMARY_UNC'].data[select])
                bg_select = np.s_[order, trim_bottom:-trim_top, 256:-512]
                background_profiles.append(
                    self._get_slit_profile(hdul['PRIMARY'].data[bg_select]))
                if wavelength_centers is None:
                    select = np.s_[order, left:right]
                    wavelength_centers = _doppler_shift_wavelengths(
                        hdul['BIN_CENTER_WAVELENGTHS'].data[select] * u. nm,
                        velocity=velocity)
                if wavelength_edges is None:
                    select = np.s_[order, left:right+1]
                    wavelength_edges = _doppler_shift_wavelengths(
                        hdul['BIN_EDGE_WAVELENGTHS'].data[select] * u.nm,
                        velocity=velocity)
        for file in trace_files:
            with fits.open(file) as hdul:
                select = np.s_[order, trim_bottom:-trim_top,
                               left+horizontal_offset:right+horizontal_offset]
                traces.append(hdul['PRIMARY'].data[select])
        return {
            'order': order,
            'headers': headers,
            'data': np.array(data),
            'uncertainty': np.array(uncertainty),
            'traces': np.array(traces),
            'wavelength_centers': wavelength_centers,
            'wavelength_edges': wavelength_edges,
            'feature_locations': locs,
            'background_profiles': np.array(background_profiles),
            'filenames': np.array(filenames),
        }

    @staticmethod
    def _get_target_radius(header: dict) -> u.Quantity:
        """
        Query the JPL Horizons ephemeris service to get the target angular
        radius.
        """
        target = header['OBJECT']
        time = Time(header['DATE-OBS'], format='isot', scale='utc')
        ephemeris = _get_ephemeris(target=target, time=time,
                                   airmass_lessthan=None, skip_daylight=False)
        return ephemeris['ang_width'][0] * ephemeris['ang_width'].unit / 2

    @staticmethod
    def _fit_gaussian(wavelengths: np.ndarray, spectrum: np.ndarray,
                      spectrum_unc: np.ndarray, line_wavelengths: [u.Quantity],
                      line_strengths: [float], target_radius: float) \
            -> ModelResult:
        """
        Make a (composite) Gaussian equal to the number of lines in the set and
        fit it. I've added a constant to account for any residual left over
        after background subtraction. I've also fixed the width of each of the
        Gaussians to radius/sqrt(2*ln(2)), under the assumption that the target
        radius is the HWHM of the Gaussian.

        Update June 22, 2022: I've added specific line ratios for the component
        lines in neutral oxygen 777.4 nm and 844.6 nm.
        """
        line_wavelengths = np.array([i.value for i in line_wavelengths])
        dwave = np.gradient(wavelengths)
        center_indices = [np.abs(wavelengths - wavelength).argmin()
                          for wavelength in line_wavelengths]
        n_lines = len(center_indices)
        prefixes = [f'gaussian{i + 1}_' for i in range(n_lines)]
        model = ConstantModel(prefix='constant_')
        model += np.sum([GaussianModel(prefix=prefix) for prefix in prefixes],
                        dtype=object)
        params = Parameters()
        params.add('constant_c', value=0, min=-np.inf, max=np.inf)
        for i, prefix in enumerate(prefixes):
            if i != 0:
                params.add(f'{prefix}amplitude', value=np.nanmax(spectrum),
                           min=0, max=np.inf, vary=False,
                           expr=f'gaussian1_amplitude '
                                f'* {line_strengths[i]}')
            else:
                params.add(f'{prefix}amplitude', value=np.nanmax(spectrum),
                           min=0, max=np.inf)
            if i == 0:
                ind = center_indices[i]
                dw = np.gradient(wavelengths)[ind] * target_radius
                params.add(f'{prefix}center',
                           value=line_wavelengths[i],
                           min=line_wavelengths[i] - dw,
                           max=line_wavelengths[i] + dw)
            else:
                dx = (line_wavelengths[i]
                      - line_wavelengths[0])
                params.add(f'{prefix}center', vary=False,
                           expr=f'gaussian1_center + {dx}')
            sigma = (dwave[center_indices[i]] * target_radius /
                     np.sqrt(2 * np.log(2)))
            params.add(f'{prefix}sigma', value=sigma, min=0.75*sigma,
                       max=1.25*sigma)
        return model.fit(spectrum, params=params, x=wavelengths,
                         weights=1/spectrum_unc**2, method='least_squares')

    def _make_mask(self, shape: (int, int), header: dict, seeing: float,
                   locs: [int], vertical_position: int or None):
        """
        Make a mask for the target using the vertical position calculated from
        the guide satellite frames and target radius padded with the seeing
        factor.
        """
        radius = self._get_target_radius(header).to(u.arcsec).value
        x, y = np.meshgrid(np.arange(shape[1]) * header['spescale'],
                           np.arange(shape[0]) * header['spascale'])
        masks = []
        edges = []
        if vertical_position is None:
            vertical_position = shape[0] / 2
        for loc in locs:
            distance = np.sqrt(
                (x - header['spescale'] * loc) ** 2 +
                (y - header['spascale'] * vertical_position) ** 2)
            mask = np.ones_like(distance)
            mask[np.where(distance < radius + seeing)] = np.nan
            mask[np.where(distance >= radius + seeing)] = 1
            edge = np.zeros_like(mask)
            edge[np.where(distance < radius + seeing)] = 1
            masks.append(mask)
            edges.append(edge)
        edges = np.sum(edges, axis=0)
        edges[np.where(edges > 0)] = 1
        return x, y, np.mean(masks, axis=0), edges

    @staticmethod
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

    def _save_average_results(self, save_directory,
                              fitted_brightness, fitted_uncertainty,
                              observed_brightness, observed_uncertainty):
        """
        Save the average results to a text file.
        """
        header = 'date fitted_brightness_[R] fitted_uncertainty_[R] ' \
                 'measured_brightness_[R] measured_uncertainty_[R]'
        save_path = Path(self._retrieval_data_directory, save_directory,
                         'results.txt')
        with open(save_path, 'w') as file:
            file.write(header + '\n')
            file.write(f'average '
                       f'{fitted_brightness} '
                       f'{fitted_uncertainty} '
                       f'{observed_brightness} '
                       f'{observed_uncertainty}\n')

    def _save_individual_results(self, save_directory,
                                 fitted_brightness, fitted_uncertainty,
                                 observed_brightness, observed_uncertainty,
                                 date):
        """
        Save the individual results to a text file.
        """
        save_path = Path(self._retrieval_data_directory, save_directory,
                         'results.txt')
        with open(save_path, 'a') as file:
            data_str = f'{date} ' \
                       f'{fitted_brightness} ' \
                       f'{fitted_uncertainty} ' \
                       f'{observed_brightness} ' \
                       f'{observed_uncertainty}\n'
            file.write(data_str)

    def _qa_graphic_2d(self, x, y, calibrated_data, calibrated_unc, edge,
                       spescale, spascale, save_directory, file_name):
        """
        Make a quality-assurance graphic for the 2D spectra. Shows raw and
        smooth data and corresponding uncertainty.
        """
        cmap = plt.get_cmap('viridis')
        dunit = 0.0875
        n_spa, n_spe = calibrated_data.shape
        aspect_ratio = spascale/spescale
        height = dunit * n_spa * 2
        width = dunit * n_spe / aspect_ratio * 2
        height += 4 * dunit
        width += 6 * dunit
        fig, axes = plt.subplots(2, 2, figsize=(width, height),
                                 constrained_layout=True)
        [ax.set_xticks([]) for ax in axes.ravel()]
        [ax.set_yticks([]) for ax in axes.ravel()]
        norm = colors.Normalize(
            vmin=np.nanpercentile(calibrated_data.value, 1),
            vmax=np.nanpercentile(calibrated_data.value, 99))
        img = axes[0, 0].pcolormesh(x, y, calibrated_data, cmap=cmap,
                                    norm=norm)
        plt.colorbar(img, ax=axes[0, 1], label='Spectral Brightness [R/nm]')
        axes[0, 1].set_title('Data (Smoothed)')
        axes[0, 1].pcolormesh(x, y, convolve(calibrated_data,
                                             Gaussian2DKernel(x_stddev=1)),
                              cmap=cmap, norm=norm)
        norm = colors.Normalize(
            vmin=np.nanpercentile(calibrated_unc.value, 1),
            vmax=np.nanpercentile(calibrated_unc.value, 99))
        img = axes[1, 0].pcolormesh(x, y, calibrated_unc, cmap=cmap, norm=norm)
        plt.colorbar(img, ax=axes[1, 1], label='Spectral Brightness [R/nm]')
        convolved_unc = convolve(calibrated_unc, Gaussian2DKernel(x_stddev=1),
                                 boundary='extend')
        axes[1, 1].pcolormesh(x, y, convolved_unc, cmap=cmap, norm=norm)
        axes[1, 1].set_title('Uncertainty (Smoothed)')
        lines = self.contour_rect_slow(edge)
        for line in lines:
            axes[0, 0].plot(np.array(line[1]) * spescale,
                            np.array(line[0]) * spascale,
                            color=color_dict['red'])
            axes[0, 1].plot(np.array(line[1]) * spescale,
                            np.array(line[0]) * spascale,
                            color=color_dict['red'])

        axes[0, 0].set_title('Data')
        axes[1, 0].set_title('Uncertainty')
        savename = Path(self._retrieval_data_directory, save_directory,
                        file_name)
        make_directory(savename.parent)
        plt.savefig(savename)
        plt.close(fig)

    def _qa_graphic_1d(self, wavelengths, spectrum_1d, spectrum_1d_unc, fit,
                       fitted_brightness, fitted_uncertainty,
                       observed_brightness, observed_uncertainty,
                       save_directory, file_name):
        """
        Make a quality-assurance graphic for the 1D spectra. Shows the observed
        data and the Gaussian fit with uncertainty.
        """
        fig, axis = plt.subplots(1, figsize=(4.5, 2), constrained_layout=True)
        axis.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        axis.xaxis.set_minor_locator(ticker.MultipleLocator(0.01))
        axis.set_xlabel('Wavelength [nm]')
        axis.set_ylabel('Spectral Brightness [R/nm]')

        axis.errorbar(wavelengths, spectrum_1d.value,
                      yerr=spectrum_1d_unc.value,
                      color='k', linewidth=1,
                      ecolor=color_dict['grey'], elinewidth=0.5)
        axis.plot(wavelengths, fit.best_fit,
                  color=color_dict['red'], linewidth=1)
        dy = fit.eval_uncertainty(x=wavelengths.value)
        axis.fill_between(wavelengths.value,
                          fit.best_fit - dy, fit.best_fit + dy,
                          color=color_dict['red'], alpha=0.5, linewidth=0)
        axis.annotate(f'Fitted brightness: '
                      f'{fitted_brightness} ± {fitted_uncertainty} R\n'
                      f'Observed brightness: '
                      f'{observed_brightness} ± {observed_uncertainty} R',
                      xy=(0, 1), xytext=(6, -6), xycoords='axes fraction',
                      textcoords='offset points', ha='left', va='top')
        savename = Path(self._retrieval_data_directory, save_directory,
                        file_name)
        make_directory(savename.parent)
        plt.savefig(savename)
        plt.close(fig)

    # noinspection DuplicatedCode
    def run_average(self, data: dict, wavelengths: u.Quantity, name: str,
                    trim_bottom: int, trim_top: int, seeing: float = 0.5,
                    line_ratios: [float] = None):
        """
        Retrieve the brightness of the average across the data frames. For now
        the average does not align using the trace because of edge effects. I
        may change this at a later date.
        """
        save_directory = name
        n_obs, n_spa, n_spe = data['data'].shape
        calibration = _FluxCalibration(
            reduced_data_directory=self._reduced_data_directory,
            wavelengths=wavelengths, order=data['order'],
            trim_bottom=trim_bottom, trim_top=trim_top)

        # trace_offsets = _TraceOffsets(self._reduced_data_directory,
        #                               order=data['order'],
        #                               trim_bottom=trim_bottom,
        #                               trim_top=trim_top)

        # average
        x, y, mask, edge = self._make_mask(
            shape=(n_spa, n_spe), header=data['headers'][0],
            seeing=seeing, locs=data['feature_locations'],
            vertical_position=None)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # average_data, average_unc = trace_offsets.align_and_average(
            #     data['data'], data['uncertainty'])
            average_data = np.nanmean(data['data'], axis=0)
            average_unc = np.sqrt(
                np.nansum(data['uncertainty'] ** 2, axis=0)) / n_obs
            average_slit_profile = np.nanmean(
                data['background_profiles'], axis=0)
            average_slit_profile /= np.nanmax(average_slit_profile)
            background = _Background(average_data, average_unc, mask,
                                     average_slit_profile)
            bgsub_data = average_data - background.background

        # calibrate data
        radius = self._get_target_radius(data['headers'][0])
        area = (np.pi * radius ** 2).to(u.sr)
        bgsub_data = bgsub_data * u.electron / u.s / area
        bgsub_unc = average_unc * u.electron / u.s / area
        calibrated_data = bgsub_data / calibration.calibration_factors

        # calculate calibrated uncertainty
        data_nsr = bgsub_unc / bgsub_data
        calibration_nsr = (calibration.calibration_factors_unc /
                           calibration.calibration_factors)
        calibrated_unc = np.abs(calibrated_data) * np.sqrt(
            data_nsr ** 2 + calibration_nsr ** 2)

        self._qa_graphic_2d(x, y, calibrated_data, calibrated_unc, edge,
                            data['headers'][0]['spescale'],
                            data['headers'][0]['spascale'],
                            f'{save_directory}/spectra_2d', 'average.jpg')

        # calculate 1D spectrum
        ind = np.where(np.isnan(np.mean(mask, axis=1)))[0]
        spectrum_1d = np.nansum(calibrated_data[ind], axis=0)
        spectrum_1d_unc = np.sqrt(np.sum(calibrated_unc[ind] ** 2, axis=0))
        dwavelength = np.gradient(data['wavelength_centers'].value)

        fit = self._fit_gaussian(
            wavelengths=data['wavelength_centers'].value,
            spectrum=spectrum_1d.value,
            spectrum_unc=spectrum_1d_unc.value,
            line_wavelengths=wavelengths,
            line_strengths=line_ratios,
            target_radius=radius.value/data['headers'][0]['SPESCALE'])
        fitted_brightness = np.nansum(
            (fit.best_fit - fit.params['constant_c'].value) * dwavelength)
        fitted_uncertainty = np.nansum(
            fit.eval_uncertainty(x=data['wavelength_centers'].value)
            * dwavelength)
        observed_brightness = np.nansum((spectrum_1d * dwavelength)).value
        observed_uncertainty = np.sqrt(
            np.nansum((spectrum_1d_unc * dwavelength) ** 2)).value

        # convert to appropriate significant figures
        fitted_brightness, fitted_uncertainty = format_uncertainty(
            fitted_brightness, fitted_uncertainty)
        observed_brightness, observed_uncertainty = format_uncertainty(
            observed_brightness, observed_uncertainty)

        self._qa_graphic_1d(
            data['wavelength_centers'], spectrum_1d, spectrum_1d_unc, fit,
            fitted_brightness, fitted_uncertainty, observed_brightness,
            observed_uncertainty,
            f'{save_directory}/spectra_1d', 'average.jpg')

        self._save_average_results(
            save_directory, fitted_brightness, fitted_uncertainty,
            observed_brightness, observed_uncertainty)

    # noinspection DuplicatedCode
    def run_individual(self, data: dict, wavelengths: u.Quantity, name: str,
                       trim_bottom: int, trim_top: int, seeing: float = 1.,
                       line_ratios: [float] = None):
        """
        Retrieve the brightness of an individual data frame. Unlike the average
        retrieval, this does use the trace alignment.
        """
        save_directory = name
        n_obs, n_spa, n_spe = data['data'].shape
        trace_offsets = _TraceOffsets(self._reduced_data_directory,
                                      order=data['order'],
                                      trim_bottom=trim_bottom,
                                      trim_top=trim_top)
        calibration = _FluxCalibration(
            reduced_data_directory=self._reduced_data_directory,
            wavelengths=wavelengths, order=data['order'],
            trim_bottom=trim_bottom, trim_top=trim_top)

        for obs in range(n_obs):
            filename = data['filenames'][obs].replace(
                '_reduced.fits.gz', '.jpg')
            x, y, mask, edge = self._make_mask(
                shape=(n_spa, n_spe), header=data['headers'][obs],
                seeing=seeing, locs=data['feature_locations'],
                vertical_position=trace_offsets.centers[obs])
            obs_data = data['data'][obs]
            obs_unc = data['uncertainty'][obs]
            obs_slit_profile = data['background_profiles'][obs]
            obs_slit_profile /= np.nanmax(obs_slit_profile)
            background = _Background(obs_data, obs_unc, mask, obs_slit_profile)
            bgsub_data = obs_data - background.background

            # calibrate data
            radius = self._get_target_radius(data['headers'][obs])
            area = (np.pi * radius ** 2).to(u.sr)
            bgsub_data = bgsub_data * u.electron / u.s / area
            bgsub_unc = obs_unc * u.electron / u.s / area
            calibrated_data = bgsub_data / calibration.calibration_factors

            # calculate calibrated uncertainty
            data_nsr = bgsub_unc / bgsub_data
            calibration_nsr = (calibration.calibration_factors_unc /
                               calibration.calibration_factors)
            calibrated_unc = np.abs(calibrated_data) * np.sqrt(
                data_nsr ** 2 + calibration_nsr ** 2)

            # calculate 1D spectrum
            ind = np.where(np.isnan(np.mean(mask, axis=1)))[0]
            spectrum_1d = np.nansum(calibrated_data[ind], axis=0)
            spectrum_1d_unc = np.sqrt(np.sum(calibrated_unc[ind] ** 2, axis=0))
            dwavelength = np.gradient(data['wavelength_centers'].value)

            fit = self._fit_gaussian(
                wavelengths=data['wavelength_centers'].value,
                spectrum=spectrum_1d.value,
                spectrum_unc=spectrum_1d_unc.value,
                line_wavelengths=wavelengths,
                line_strengths=line_ratios,
                target_radius=radius.value/data['headers'][0]['SPESCALE'])
            fitted_brightness = np.nansum(
                (fit.best_fit - fit.params['constant_c'].value) * dwavelength)
            fitted_uncertainty = np.nansum(
                fit.eval_uncertainty(x=data['wavelength_centers'].value)
                * dwavelength)
            observed_brightness = np.nansum((spectrum_1d * dwavelength)).value
            observed_uncertainty = np.sqrt(
                np.nansum((spectrum_1d_unc * dwavelength) ** 2)).value

            # convert to appropriate significant figures
            fitted_brightness, fitted_uncertainty = format_uncertainty(
                fitted_brightness, fitted_uncertainty)
            observed_brightness, observed_uncertainty = format_uncertainty(
                observed_brightness, observed_uncertainty)

            self._qa_graphic_2d(x, y, calibrated_data, calibrated_unc, edge,
                                data['headers'][0]['spescale'],
                                data['headers'][0]['spascale'],
                                f'{save_directory}/spectra_2d', filename)

            self._qa_graphic_1d(
                data['wavelength_centers'], spectrum_1d, spectrum_1d_unc, fit,
                fitted_brightness, fitted_uncertainty, observed_brightness,
                observed_uncertainty, f'{save_directory}/spectra_1d', filename)

            self._save_individual_results(
                save_directory, fitted_brightness, fitted_uncertainty,
                observed_brightness, observed_uncertainty,
                data['headers'][obs]['DATE-OBS']
            )

    def run_all(self, extended: bool = False, trim_bottom: int = 2,
                trim_top: int = 2, horizontal_offset: int = 0,
                seeing: float = 1/2):
        """
        Wrapper to retrieve both the average and individual aurora
        brightnesses.
        """
        wavelengths = aurora_line_wavelengths(extended=extended)
        line_strengths = emission_line_strengths(extended=extended)
        names = aurora_line_names(extended=extended)
        for wavelength, line_strength, name in zip(wavelengths, line_strengths,
                                                   names):
            try:
                data = self._get_data(wavelengths=wavelength,
                                      trim_bottom=trim_bottom,
                                      trim_top=trim_top,
                                      horizontal_offset=horizontal_offset)
            except WavelengthNotFoundError:
                print(f'{name} not found in available orders, skipping...')
                continue
            print(f'Retrieving {name} brightnesses...')
            self.run_average(data=data, wavelengths=wavelength,
                             name=name, line_ratios=line_strength,
                             trim_bottom=trim_bottom, trim_top=trim_top,
                             seeing=seeing)
            self.run_individual(data=data, wavelengths=wavelength,
                                name=name, line_ratios=line_strength,
                                trim_bottom=trim_bottom,
                                trim_top=trim_top,
                                seeing=seeing)


def run_retrieval(reduced_data_directory: str or Path, extended: bool = False,
                  trim_bottom: int = 2, trim_top: int = 2,
                  horizontal_offset: int = 0,
                  seeing: float = 1.):
    """
    Wrapper function to run brightness retrievals on a set of reduced
    observations.

    Parameters
    ----------
    reduced_data_directory : str or Path
        Absolute path to reduced data from the HIRES pipeline.
    extended : bool
        Whether or not to use the extended wavelength list (for Io).
    trim_bottom : int
        How many rows to trim from the bottom of each order.
    trim_top : int
        How many rows to trim from the top of each order.
    horizontal_offset : int
        Additional pixel offset for horizontal centering.
    seeing : float
        Atmospheric seeing, use as a proxy to increase the aperture from which
        the retrieval occurs.
    """
    retrieval = _Retrieval(reduced_data_directory=reduced_data_directory)
    retrieval.run_all(extended=extended, trim_bottom=trim_bottom,
                      trim_top=trim_top, horizontal_offset=horizontal_offset,
                      seeing=seeing)


if __name__ == "__main__":
    # run_retrieval(
    #     reduced_data_directory='/Users/zachariahmilby/Documents/School/'
    #                            'Planetary Sciences PhD/Projects/'
    #                            'Galilean Satellite Aurora (Katherine de Kleer)'
    #                            '/HIRES/Data/Io 2022-11-24/reduced',
    #     extended=True, trim_bottom=6, trim_top=8, seeing=2,
    #     horizontal_offset=0)

    run_retrieval(
        reduced_data_directory='/Users/zachariahmilby/Documents/School/'
                               'Planetary Sciences PhD/Projects/'
                               'Galilean Satellite Aurora (Katherine de Kleer)'
                               '/HIRES/Data/Ganymede 2021-06-08/reduced',
        extended=False, trim_bottom=2, trim_top=4, seeing=4/5,
        horizontal_offset=0)
