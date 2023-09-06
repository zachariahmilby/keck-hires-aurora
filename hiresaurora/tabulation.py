from pathlib import Path

import pandas as pd
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.table as mpltable
from matplotlib.backends.backend_pdf import PdfPages
from astropy.time import Time

from hiresaurora.general import AuroraLines, FuzzyQuantity

# noinspection PyProtectedMember
weird_error = np.core._exceptions._UFuncNoLoopError


class TabulatedResults:
    """
    Class to collect and tabulate results from data calibration brightness
    retrievals.
    """

    _superscripts = {'⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
                     '⁵': '⁵5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9',
                     '⁺': '+', '⁻': '-'}

    def __init__(self, calibrated_data_path: str or Path, excluded: [int],
                 extended: bool):
        """
        Parameters
        ----------
        calibrated_data_path : str or Path
            File path to directory containing wavelength-specific
            subdirectories with calibrated data files.
        excluded : [int]
            Indices of observations excluded from averaging.
        extended : bool
            Whether or not the pipeline used the extended Io line list.
        """
        self._calibrated_data_path = Path(calibrated_data_path)
        self._excluded = excluded
        self._aurora_lines = AuroraLines(extended=extended)

    def _make_mpl_table(self, results: pd.DataFrame) -> None:
        """
        Save a color-coded summary table.
        """
        columns = [s for s in results.columns
                   if s not in ['Observation Time', 'Excluded from Averaging',
                                'Distance from Plasma Sheet Equator [R_J]']]
        columnlabels = np.array([s for s in columns[0::3]])
        columns = np.array([s.replace(' [R]', '') for s in columns[0::3]])
        rows = results['Observation Time'].to_numpy(dtype=str)
        rowlabels = [Time(t, format='isot').to_datetime().strftime('%H:%M:%S')
                     for t in rows[:-2]]
        rowlabels = np.append(rowlabels, ['Average of Above'])
        rowlabels = np.append(rowlabels, ['Average Image'])
        data = np.zeros((rows.size, columns.size), dtype=object)
        detected = np.zeros((rows.size, columns.size), dtype=bool)
        excluded = np.zeros((rows.size, columns.size), dtype=bool)
        for i, row in enumerate(rows):
            subresult = results[results['Observation Time'] == row]
            if i < rows.size - 1:
                excluded[i] = \
                    subresult['Excluded from Averaging'].to_numpy(dtype=bool)
            for j, column in enumerate(columns):
                brightness = subresult[f'{column} [R]'].to_numpy()[0]
                try:
                    uncertainty = np.min(
                        [subresult[f'{column} Unc. [R]'].to_numpy()[0],
                         subresult[f'{column} Std. [R]'].to_numpy()[0]])
                    fuzz = FuzzyQuantity(brightness, uncertainty)
                    label = fuzz.printable
                    label = label.replace(' R', '')
                    label = label.replace('±', r'\pm')
                    if '×' in label:
                        power = f"×{label.split('×')[1]}"
                        label = label.replace(
                            power, fr'\times 10^{{{fuzz.magnitude}}}')
                    label = label.replace('×', r'\times')
                    label = f'${label}$'
                    data[i, j] = label
                    try:
                        snr = fuzz.value / fuzz.uncertainty
                    except ZeroDivisionError:
                        snr = np.nan
                    if snr >= 2.0:
                        detected[i, j] = True
                except weird_error:
                    data[i, j] = 'error'
                    detected[i, j] = False
        cell_colors = np.full(data.shape, fill_value='red', dtype=object)
        cell_colors[np.where(detected)] = 'green'

        cols = 6
        n_pages = np.ceil(columns.size / cols).astype(int)

        with PdfPages(Path(self._calibrated_data_path, 'results.pdf')) as pdf:
            for n in range(n_pages):
                figsize = (cols * 1.25, rows.size * 0.25)
                fig, axis = plt.subplots(figsize=figsize, layout='constrained',
                                         clear=True)
                axis.set_frame_on(False)
                axis.set_xticks([])
                axis.set_yticks([])

                # get data subset
                s_ = np.s_[n*cols:(n+1)*cols]
                col_labels = columnlabels[s_]
                subdata = data[:, s_]
                subdetected = detected[:, s_]

                # noinspection PyTypeChecker
                table = mpltable.table(ax=axis, cellText=subdata,
                                       rowLabels=rowlabels,
                                       colLabels=col_labels,
                                       loc='center')
                cells = table.get_celld()
                for i in range(rows.size):
                    cells[i+1, -1].set_edgecolor('none')
                    for j in range(col_labels.size):
                        cells[0, j].set_edgecolor('none')
                        cells[i+1, j].set_alpha(0.1)
                        if subdetected[i, j]:
                            cells[i+1, j].get_text().set_color('green')
                        else:
                            cells[i+1, j].get_text().set_color('red')
                        if excluded[i, j]:
                            cells[i+1, j].get_text().set_alpha(0.5)
                            cells[i+1, j].set_alpha(0.05)
                            cells[i+1, -1].get_text().set_alpha(0.5)
                            cells[i+1, -1].set_alpha(0.05)
                        cells[i+1, j].set_edgecolor('none')
                        cells[rows.size, j].set_edgecolor('none')
                        cells[rows.size, -1].set_edgecolor('none')
                        if subdetected[i, j]:
                            cells[i+1, j].set_facecolor('green')
                        else:
                            cells[i+1, j].set_facecolor('red')
                pdf.savefig()
                plt.close(fig)

    # noinspection DuplicatedCode
    def _calculate_averages_of_individuals(
            self, results: pd.DataFrame) -> pd.DataFrame:
        new_data = pd.DataFrame()
        rows = results['Observation Time'].to_numpy()
        rows = np.insert(rows, -1, 'Average of Above')
        new_data['Observation Time'] = rows
        key = 'Distance from Plasma Sheet Equator [R_J]'
        distances = results[key].to_numpy()
        distances = np.insert(distances, -1, distances[-1])
        new_data[key] = distances
        good = np.where(
            results['Excluded from Averaging'].to_numpy()[:-1] == 0)[0]
        n = good.size
        for name in self._aurora_lines.names:
            try:
                brightnesses = results[f'{name} [R]'].to_numpy()
                if 'error' in brightnesses.tolist():
                    avg_brightness = 'error'
                else:
                    avg_brightness = np.mean(brightnesses[good])
                brightnesses = np.insert(brightnesses, -1, avg_brightness)
                try:
                    new_data[f'{name} [R]'] = np.round(brightnesses, 4)
                except TypeError:
                    new_data[f'{name} [R]'] = brightnesses
                uncertainties = results[f'{name} Unc. [R]'].to_numpy()
                if 'error' in uncertainties.tolist():
                    avg_uncertainty = 'error'
                else:
                    avg_uncertainty = \
                        np.sqrt(np.sum(uncertainties[good]**2)) / n
                uncertainties = np.insert(uncertainties, -1, avg_uncertainty)
                try:
                    new_data[f'{name} Unc. [R]'] = np.round(uncertainties, 4)
                except TypeError:
                    new_data[f'{name} Unc. [R]'] = uncertainties
                stddevs = results[f'{name} Std. [R]'].to_numpy()
                if 'error' in stddevs.tolist():
                    avg_stddev = 'error'
                else:
                    avg_stddev = np.sqrt(np.sum(stddevs[good] ** 2)) / n
                stddevs = np.insert(stddevs, -1, avg_stddev)
                try:
                    new_data[f'{name} Std. [R]'] = np.round(stddevs, 4)
                except TypeError:
                    new_data[f'{name} Std. [R]'] = stddevs
            except KeyError:
                continue
        new_data['Excluded from Averaging'] = np.insert(
            results['Excluded from Averaging'].to_numpy(), -1, 0)
        return new_data

    def run(self):
        """
        Generate and save the tabulated data as a CSV.
        """
        savename = Path(self._calibrated_data_path, 'results.csv')
        results = pd.DataFrame()
        excluded = None
        for name in self._aurora_lines.names:
            directory = Path(self._calibrated_data_path, name)
            files = sorted(directory.glob('*.fits.gz'))
            if len(files) == 0:
                continue
            times = []
            brightnesses = []
            uncertainties = []
            stddevs = []
            distances = []
            if excluded is None:
                excluded = np.zeros(len(files)).astype(bool)
            for i, file in enumerate(files):
                with fits.open(file) as hdul:
                    primary_header = hdul['PRIMARY'].header
                    data_header = hdul['CALIBRATED'].header
                    if file.name == 'average.fits.gz':
                        times.append('Average Image')
                    else:
                        times.append(data_header['DATE-OBS'])
                    brightness = primary_header['BGHTNESS']
                    uncertainty = primary_header['BGHT_UNC']
                    stddev = primary_header['BGHT_STD']
                    try:
                        distance = primary_header['PS_DIST']
                    except KeyError:
                        distance = np.nan
                    brightnesses.append(brightness)
                    uncertainties.append(uncertainty)
                    stddevs.append(stddev)
                    distances.append(distance)
                if self._excluded is not None:
                    if i in self._excluded:
                        excluded[i] = True
                    if file.name == 'average.fits.gz':
                        excluded[i] = True
            distances = np.array(distances)
            distances[np.where(np.isnan(distances))] = np.nanmean(distances)
            distances = np.round(distances, 4)
            results['Observation Time'] = times
            results['Distance from Plasma Sheet Equator [R_J]'] = distances
            results[f'{name} [R]'] = brightnesses
            results[f'{name} Unc. [R]'] = uncertainties
            results[f'{name} Std. [R]'] = stddevs
        results['Excluded from Averaging'] = excluded
        results = self._calculate_averages_of_individuals(results=results)
        results.to_csv(savename, index=False, sep=',')
        self._excluded = excluded
        self._make_mpl_table(results=results)


def tabulate_results(calibrated_data_path: str or Path,
                     excluded: [int], extended: bool) -> TabulatedResults:
    """
    Wrapper function to tabulate aurora results.

    Parameters
    ----------
    calibrated_data_path : str or Path
        File path to directory containing wavelength-specific
        subdirectories with calibrated data files.
    excluded : [int]
        Indices of observations excluded from averaging.
    extended: bool
        Whether or not the pipeline used the extended Io line list.

    Returns
    -------
    None.
    """
    tabulated_results = TabulatedResults(
        calibrated_data_path=calibrated_data_path, excluded=excluded,
        extended=extended)
    tabulated_results.run()
    return tabulated_results
