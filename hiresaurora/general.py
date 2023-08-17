from pathlib import Path

import astropy.constants as c
import astropy.units as u
import numpy as np

_package_directory = Path(__file__).resolve().parent

rcparams = Path(_package_directory, 'anc', 'rcparams.mplstyle')

color_dict = {'red': '#D62728', 'orange': '#FF7F0E', 'yellow': '#FDB813',
              'green': '#2CA02C', 'blue': '#0079C1', 'violet': '#9467BD',
              'cyan': '#17BECF', 'magenta': '#D64ECF', 'brown': '#8C564B',
              'darkgrey': '#3F3F3F', 'grey': '#7F7F7F', 'lightgrey': '#BFBFBF'}


naif_codes = {'Jupiter': '599', 'Io': '501', 'Europa': '502',
              'Ganymede': '503', 'Callisto': '504', 'Maunakea': '568'}


# noinspection PyUnresolvedReferences
def _doppler_shift_wavelengths(wavelengths: u.Quantity, velocity):
    """
    Apply Doppler shift to wavelengths.
    """
    return (wavelengths * (1 - velocity.si / c.c)).si.to(u.nm)


class _EmissionLine:
    """
    Class to hold emission line information.
    """
    def __init__(self, wavelengths: u.Quantity):
        self._wavelengths = wavelengths

    @property
    def wavelengths(self) -> u.Quantity:
        return self._wavelengths


_emission_lines = {
    '[K I] 364.9 nm': _EmissionLine(wavelengths=[364.8985] * u.nm),
    '[Na I] 342.7 nm': _EmissionLine(wavelengths=[342.6858] * u.nm),
    '[Na I] 388.4 nm': _EmissionLine(wavelengths=[388.3903] * u.nm),
    '[S II] 406.9 nm': _EmissionLine(wavelengths=[406.8600, 407.6349] * u.nm),
    '[S II] 407.6 nm': _EmissionLine(wavelengths=[407.6349] * u.nm),
    'H I 434.0 nm': _EmissionLine(
        wavelengths=[434.0431, 434.0500, 434.0427, 434.0494, 434.0496] * u.nm),
    '[S I] 458.9 nm': _EmissionLine(wavelengths=[458.926] * u.nm),
    '[C I] 462.7 nm': _EmissionLine(wavelengths=[462.7344] * u.nm),
    '[K I] 464.2 nm': _EmissionLine(wavelengths=[464.2373] * u.nm),
    'H I 486.1 nm': _EmissionLine(
        wavelengths=[486.1288, 486.1375, 486.1279, 486.1362, 486.1365] * u.nm),
    '[O I] 557.7 nm': _EmissionLine(wavelengths=[557.7339] * u.nm),
    'Na I 589.0 nm': _EmissionLine(wavelengths=[588.9950] * u.nm),
    'Na I 589.6 nm': _EmissionLine(wavelengths=[589.5924] * u.nm),
    '[O I] 630.0 nm': _EmissionLine(wavelengths=[630.0304] * u.nm),
    '[O I] 636.4 nm': _EmissionLine(wavelengths=[636.3776] * u.nm),
    'H I 656.3 nm': _EmissionLine(
        wavelengths=[656.2752, 656.2909, 656.2710, 656.2852, 656.2867] * u.nm),
    '[S II] 671.6 nm': _EmissionLine(wavelengths=[671.6338] * u.nm),
    '[S II] 673.1 nm': _EmissionLine(wavelengths=[673.0713] * u.nm),
    '[Na I] 751.5 nm': _EmissionLine(
        wavelengths=[750.7464, 751.7172, 752.0333] * u.nm),
    'K I 766.4 nm': _EmissionLine(wavelengths=[766.4899] * u.nm),
    'K I 769.9 nm': _EmissionLine(wavelengths=[769.8965] * u.nm),
    '[O II] 731.9 nm': _EmissionLine(wavelengths=[731.8811, 731.9878] * u.nm),
    '[O II] 733.0 nm': _EmissionLine(wavelengths=[732.9554, 733.0624] * u.nm),
    '[S I] 772.5 nm': _EmissionLine(wavelengths=[772.5046] * u.nm),
    'O I 777.4 nm': _EmissionLine(
        wavelengths=[777.1944, 777.4166, 777.5388] * u.nm),
    'Na I 818.3 nm': _EmissionLine(wavelengths=[818.3256] * u.nm),
    'Na I 819.5 nm': _EmissionLine(wavelengths=[819.4790, 819.4824] * u.nm),
    'Cl I 837.6 nm': _EmissionLine(wavelengths=[837.5943] * u.nm),
    'O I 844.6 nm': _EmissionLine(
        wavelengths=[844.6247, 844.6359, 844.6758] * u.nm),
    '[C I] 872.7 nm': _EmissionLine(wavelengths=[872.7131] * u.nm),
    'S I 921.3 nm': _EmissionLine(wavelengths=[921.2865] * u.nm),
    'S I 922.8 nm': _EmissionLine(wavelengths=[922.8092] * u.nm),
    'S I 923.8 nm': _EmissionLine(wavelengths=[923.7538] * u.nm,),
}


icy_satellite_lines = ['[O I] 557.7 nm', '[O I] 630.0 nm', '[O I] 636.4 nm',
                       'H I 656.3 nm', 'O I 777.4 nm', 'O I 844.6 nm']


class AuroraLines:
    """
    Class to hold aurora emission line information.
    """
    def __init__(self, extended: bool = False):
        """
        Parameters
        ----------
        extended : bool
            Whether or not to include the extended Io line sets.
        """
        self._extended = extended
        self._aurora_line_wavelengths = self._get_aurora_line_wavelengths()
        self._aurora_line_names = self._get_aurora_line_names()

    def __str__(self):
        print_str = 'Aurora lines'
        if self._extended:
            print_str += ' (extended set):'
        else:
            print_str += ':'
        for name in self._aurora_line_names:
            print_str += '\n' + f'   {name}'
        return print_str

    def _get_aurora_line_wavelengths(self) -> [u.Quantity]:
        """
        Retrieve a list of all of the aurora wavelengths. Each is in a sublist
        to keep closely-spaced doublets and triplets together.
        """

        if self._extended:
            return [_emission_lines[key].wavelengths
                    for key in _emission_lines.keys()]
        else:
            return [_emission_lines[key].wavelengths
                    for key in _emission_lines.keys()
                    if key in icy_satellite_lines]

    def _get_aurora_line_names(self) -> [str]:
        """
        Get the atom name and wavelength to 1 decimal place. In astronomer
        notation (I for neutral, II for singly-ionized, etc., with square
        brackets for forbidden transitions).
        """

        if self._extended:
            return list(_emission_lines.keys())
        else:
            return icy_satellite_lines

    @property
    def wavelengths(self) -> [u.Quantity]:
        return self._aurora_line_wavelengths

    @property
    def names(self) -> [str]:
        return self._aurora_line_names


def format_uncertainty(quantity: int | float,
                       uncertainty: int | float) -> (float, float):
    """
    Reformats a quantity and its corresponding uncertainty to a proper number
    of decimal places. For uncertainties starting with 1, it allows two
    significant digits in the uncertainty. For 2-9, it allows only one. It
    scales the value to match the precision of the uncertainty.

    Parameters
    ----------
    quantity : float
        A measured quantity.
    uncertainty : float
        The measured quantity's uncertainty.

    Returns
    -------
    The correctly-formatted value and uncertainty.

    Examples
    --------
    Often fitting algorithms will report uncertainties with way more precision
    than appropriate:
    >>> format_uncertainty(1.023243, 0.563221)
    (1.0, 0.6)

    If the uncertainty is larger than 1.9, it returns the numbers as
    appropriately-rounded integers instead of floats, to avoid giving the
    impression of greater precision than really exists:
    >>> format_uncertainty(134523, 122)
    (134520, 120)

    It can handle positive or negative quantities (but uncertainties should
    always be positive by definition):
    >>> format_uncertainty(-10.2, 1.1)
    (-10.2, 1.1)

    >>> format_uncertainty(10.2, -2.1)
    Traceback (most recent call last):
     ...
    ValueError: Uncertainty must be a positive number.

    If the uncertainty is larger than the value, the precision is still set by
    the uncertainty.
    >>> format_uncertainty(0.023, 4.322221)
    (0, 4)
    """
    if np.sign(uncertainty) == -1.0:
        raise ValueError('Uncertainty must be a positive number.')
    if f'{uncertainty:#.1e}'[0] == '1':
        unc = float(f'{uncertainty:#.1e}')
        one_more = 1
        order = int(f'{uncertainty:#.1e}'.split('e')[1])
    else:
        unc = float(f'{uncertainty:#.0e}')
        one_more = 0
        order = int(f'{uncertainty:#.0e}'.split('e')[1])
    mag_quantity = int(f'{quantity:.3e}'.split('e')[1])
    mag_uncertainty = int(f'{uncertainty:.3e}'.split('e')[1])
    mag_diff = mag_quantity - mag_uncertainty
    if mag_diff < 0:
        fmt = one_more
    else:
        fmt = mag_diff + one_more
    val = f'{quantity:.{fmt}e}'
    if (np.sign(order) == -1) or ((order == 0) & (one_more == 1)):
        return float(val), float(unc)
    else:
        if int(float(unc)) == 0:
            return '---', '---'
        else:
            return int(float(val)), int(float(unc))
