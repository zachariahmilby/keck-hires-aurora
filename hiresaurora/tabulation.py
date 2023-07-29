from pathlib import Path

import pandas as pd
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.table as mpltable
from astropy.time import Time

from hiresaurora.general import format_uncertainty, AuroraLines


class TabulatedResults:
    """
    Class to collect and tabulate results from data calibration brightness
    retrievals.
    """
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
        columns = np.array(self._aurora_lines.names)
        columnlabels = [f'{c} [R]' for c in columns]
        rows = results['Observation Time'].to_numpy(dtype=str)
        rowlabels = [Time(t, format='isot').to_datetime().strftime('%H:%M:%S')
                     for t in rows[:-1]]
        rowlabels = np.append(rowlabels, ['Average'])
        data = np.zeros((rows.size, columns.size), dtype=object)
        detected = np.zeros((rows.size, columns.size), dtype=bool)
        excluded = np.zeros((rows.size, columns.size), dtype=bool)
        for i, row in enumerate(rows):
            subresult = results[results['Observation Time'] == row]
            if i < rows.size - 1:
                excluded[i] = \
                    subresult['Excluded from Averaging'].to_numpy(dtype=bool)
            for j, column in enumerate(columns):
                try:
                    brightness = subresult[f'{column} [R]'].to_numpy()[0]
                    uncertainty = np.min(
                        [subresult[f'{column} Unc. [R]'].to_numpy()[0],
                         subresult[f'{column} Std. [R]'].to_numpy()[0]])
                    val, unc = format_uncertainty(quantity=brightness,
                                                  uncertainty=uncertainty)
                    data[i, j] = f'{val} ± {unc}'
                    snr = brightness / uncertainty
                    if snr >= 2.0:
                        detected[i, j] = True
                except KeyError:
                    data[i, j] = 'none'
        cell_colors = np.full(data.shape, fill_value='red', dtype=object)
        cell_colors[np.where(detected)] = 'green'

        figsize = (columns.size * 1.25, rows.size * 0.25)
        fig, axis = plt.subplots(figsize=figsize, layout='constrained')
        axis.set_frame_on(False)
        axis.set_xticks([])
        axis.set_yticks([])

        # noinspection PyTypeChecker
        table = mpltable.table(ax=axis, cellText=data,
                               rowLabels=rowlabels, colLabels=columnlabels,
                               loc='center')
        cells = table.get_celld()
        for i in range(rows.size):
            cells[i+1, -1].set_edgecolor('none')
            for j in range(columns.size):
                cells[0, j].set_edgecolor('none')
                cells[i+1, j].set_alpha(0.1)
                if detected[i, j]:
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
                if detected[i, j]:
                    cells[i+1, j].set_facecolor('green')
                else:
                    cells[i+1, j].set_facecolor('red')
        savename = Path(self._calibrated_data_path.parent, 'results.pdf')
        plt.savefig(savename)

    def run(self):
        """
        Generate and save the tabulated data as a CSV.
        """
        savename = Path(self._calibrated_data_path.parent, 'results.csv')
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
            if excluded is None:
                excluded = np.zeros(len(files)).astype(bool)
            for i, file in enumerate(files):
                with fits.open(file) as hdul:
                    primary_header = hdul['PRIMARY'].header
                    data_header = hdul['CALIBRATED'].header
                    if file.name == 'average.fits.gz':
                        times.append('Average')
                    else:
                        times.append(data_header['DATE-OBS'])
                    brightness = primary_header['BGHTNESS']
                    uncertainty = primary_header['BGHT_UNC']
                    stddev = primary_header['BGHT_STD']
                    brightnesses.append(brightness)
                    uncertainties.append(uncertainty)
                    stddevs.append(stddev)
                if self._excluded is not None:
                    if i in self._excluded:
                        excluded[i] = True
                    if file.name == 'average.fits.gz':
                        excluded[i] = True
            results['Observation Time'] = times
            results[f'{name} [R]'] = brightnesses
            results[f'{name} Unc. [R]'] = uncertainties
            results[f'{name} Std. [R]'] = stddevs
        results['Excluded from Averaging'] = excluded
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
    print('Tabulating results...')
    tabulated_results = TabulatedResults(
        calibrated_data_path=calibrated_data_path, excluded=excluded,
        extended=extended)
    tabulated_results.run()
    return tabulated_results
