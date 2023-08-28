import re
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


class FuzzyQuantity:
    """
    A class for proper formatting of values with uncertainties. I've only
    tested this on single numbers/quantities with associated uncertainty. I
    don't believe it will work with arrays, besides, each entry in an array
    might have a different formatting anyway, and this is really only useful
    when you are printing out uncerainties or otherwise reporting them in
    print.
    """
    _superscripts = {'0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
                     '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
                     '+': '⁺', '-': '⁻'}

    def __init__(self, value: int or float or u.Quantity,
                 uncertainty: int or float or u.Quantity):
        """
        This should work for values and uncertainties that are ints, floats or
        Astropy Quantities (wih associated units). If either is an Astropy
        Quantity object, the cother uncertainty must also be a Quantity object
        with equivalent units (so, value in meters and uncertainty in
        centimeters is fine, but value in ergs and uncertainty in Watts is
        not).

        If either is a NaN, both are returned as NaNs.

        Parameters
        ----------
        value : int or float or u.Quantity
            A measured value.
        uncertainty : int or float or u.Quantity
            The measured value's uncertainty.
        """
        if (isinstance(value, u.Quantity) and
                isinstance(uncertainty, u.Quantity)):
            self._check_units(value=value, uncertainty=uncertainty)
            self._value = float(value.value)
            self._unc = float(uncertainty.to(value.unit).value)
            self._unit = value.unit
        else:
            self._value = float(value)
            self._unc = float(uncertainty)
            self._unit = None
        if np.isnan(self._value) or np.isnan(self._unc):
            self._formatted_value = np.nan
            self._formatted_unc = np.nan
            self._substring_value, self._substring_unc = 'nan', 'nan'
        else:
            self._check_uncertainty_sign()
            self._value_magnitude, self._unc_magnitude = \
                self._get_magnitudes()
            self._magnitude = np.max(
                [self._value_magnitude, self._unc_magnitude])
            self._scale = 10 ** self._magnitude
            self._precision, self._unc_decimals = self._get_precision()
            self._formatted_value, self._formatted_unc = self._parse_numbers()
            self._substring_value, self._substring_unc = \
                self._make_sub_strings()

    def __str__(self):
        left = ''
        right = ''
        power = ''
        unit = ''
        if np.isnan(self._formatted_value) or np.isnan(self._formatted_unc):
            value = 'nan'
            unc = 'nan'
        elif (self._scale == 100) and self._precision >= 2:
            fmt = self._precision - 2
            value = f'{self._formatted_value:.{fmt}f}'
            unc = f'{self._formatted_unc:.{fmt}f}'
        elif (self._scale == 100) and self._precision < 2:
            value = f'{self._formatted_value:.0f}'
            unc = f'{self._formatted_unc:.0f}'
        elif (self._scale == 10) and self._precision >= 1:
            fmt = self._precision - 1
            value = f'{self._formatted_value:.{fmt}f}'
            unc = f'{self._formatted_unc:.{fmt}f}'
        elif (self._scale == 10) and self._precision < 1:
            value = f'{self._formatted_value:.0f}'
            unc = f'{self._formatted_unc:.0f}'
        elif (self._scale == 1) and self._precision > 0:
            fmt = self._precision
            value = f'{self._formatted_value:.{fmt}f}'
            unc = f'{self._formatted_unc:.{fmt}f}'
        elif (self._scale == 1) and self._precision == 0:
            value = f'{self._formatted_value:.0f}'
            unc = f'{self._formatted_unc:.0f}'
        elif (self._scale == 0.1) and self._precision >= 1:
            fmt = self._precision + 1
            value = f'{self._formatted_value:.{fmt}f}'
            unc = f'{self._formatted_unc:.{fmt}f}'
        elif (self._scale == 0.1) and self._precision == 0:
            value = f'{self._formatted_value:.1f}'
            unc = f'{self._formatted_unc:.1f}'
        elif (self._scale == 0.01) and self._precision >= 1:
            fmt = self._precision + 2
            value = f'{self._formatted_value:.{fmt}f}'
            unc = f'{self._formatted_unc:.{fmt}f}'
        elif (self._scale == 0.01) and self._precision == 0:
            value = f'{self._formatted_value:.2f}'
            unc = f'{self._formatted_unc:.2f}'
        else:
            left = '('
            right = ')'
            if self._value_magnitude < self._unc_magnitude:
                fmt = self._precision + self._unc_decimals
            else:
                fmt = self._precision
            unc = f'{self._formatted_unc / self._scale:.{fmt}f}'
            value = f'{self._formatted_value / self._scale:.{fmt}f}'
            exponent = f'{int(self._magnitude)}'
            for key, val in self._superscripts.items():
                exponent = exponent.replace(key, val)
            power = f' × 10{exponent}'
        if self._unit is not None:
            left = '('
            right = ')'
            unit = f" {self._unit.to_string('fits')}"
            for key, val in self._superscripts.items():
                unit = unit.replace(key, val)
        print_str = f'{left}{value} ± {unc}{right}{power}{unit}'
        return print_str

    @staticmethod
    def _check_units(value: u.Quantity, uncertainty: u.Quantity):
        try:
            uncertainty.to(value.unit)
        except u.UnitConversionError:
            raise Exception(
                'Value and uncertainty must have equivalent units!')

    def _check_uncertainty_sign(self):
        if np.sign(self._unc) == -1.0:
            raise ValueError('Uncertainty must be a positive number.')

    def _get_magnitudes(self) -> (int, int):
        value_magnitude = np.floor(np.log10(np.abs(self._value)))
        uncertainty_magnitude = np.floor(np.log10(np.abs(self._unc)))
        return value_magnitude, uncertainty_magnitude

    def _get_precision(self) -> (int, int):
        unc = f'{self._unc:.1e}'.split('e')[0]
        if unc[0] == '1':
            extra = 1
        else:
            extra = 0
        magnitude_difference = self._value_magnitude - self._unc_magnitude
        precision = np.max([int(magnitude_difference + extra), 0])
        return precision, extra

    def _parse_numbers(self) -> (float, float):
        value = float(f'{self._value:.{self._precision}e}')
        unc = float(f'{self._unc:.{self._unc_decimals}e}')
        return value, unc

    def _convert_to_latex(self) -> str:
        string = self.__str__()
        exponent = f'{int(self._magnitude)}'
        for key, val in self._superscripts.items():
            exponent = exponent.replace(key, val)
        power = f' × 10{exponent}'
        latex_power = fr' \times 10^{{{int(self._magnitude)}}}'
        string = string.replace(power, latex_power)
        string = string.replace('±', r'\pm')
        if self._unit is not None:
            unit_str = self._unit.to_string('latex_inline').replace('$', '')
            string = string.replace(f' {self._unit}', fr'\,{unit_str}')
        return string

    def _make_sub_strings(self) -> (str, str):
        string = self.__str__().replace('(', '').replace(')', '')
        parts = re.split(' ± | × ', string)
        if len(parts) == 2:
            return parts[0], parts[1]
        elif len(parts) == 3:
            return f'{parts[0]} × {parts[2]}', f'{parts[1]} × {parts[2]}'
        elif len(parts) == 4:
            return f'{parts[0]} × {parts[2]} {parts[3]}', \
                   f'{parts[1]} × {parts[2]} {parts[3]}'

    @property
    def value(self) -> float:
        return self._value

    @property
    def uncertainty(self) -> float:
        return self._unc

    @property
    def magnitude(self) -> int:
        return int(self._magnitude)

    @property
    def value_formatted(self) -> float:
        return self._formatted_value

    @property
    def uncertainty_formatted(self) -> float:
        return self._formatted_unc

    @property
    def value_printable(self) -> str:
        return self._substring_value

    @property
    def uncertainty_printable(self) -> str:
        return self._substring_unc

    @property
    def printable(self) -> str:
        return self.__str__()

    @property
    def latex(self) -> str:
        return self._convert_to_latex()


if __name__ == "__main__":
    fuzz = FuzzyQuantity(value=1.2, uncertainty=np.nan)
    print(fuzz.uncertainty_printable)
