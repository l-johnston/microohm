"""Quantity"""
from fractions import Fraction
from numbers import Number
import numpy as np
from microohm.factors import Factors
from microohm.parser import parse
from microohm.units_definitions import REVLU


class Quantity:
    """Quantity - a physical quantity converted to SI base units as a floating point number

    Parameters
    ----------
    value: float or str
    units: str
        if 'value' is numeric, provide the units as a string, otherwise leave as default None

    Usage
    -----
    >>> Quantity(1, "mm")
    Quantity('0.001 m')
    >>> Quantity("1 kΩ")
    Quantity('1000.0 Ω')
    >>> q = Quantity(1, "mW")
    >>> q.display(".1f dBm")
    0.0 dBm
    >>> q = Quantity(1, "m/ms")
    >>> print(f"{q:.1f km/s}")
    1.0 km/s
    """

    def __init__(self, value, units=None):
        if units:
            f = parse(units) if isinstance(units, str) else units
            self._value = float(value) * f.multiplier + f.offset
            self._factors = Factors(1, 0, f.m, f.kg, f.s, f.A, f.K, f.mol, f.cd)
        elif isinstance(value, str):
            vl = value.split()
            if len(vl) > 1:
                v = float(vl[0])
                f = parse(vl[1])
                self._value = v * f.multiplier + f.offset
                self._factors = Factors(1, 0, f.m, f.kg, f.s, f.A, f.K, f.mol, f.cd)
            else:
                self._value = float(value)
                self._factors = Factors()
        else:
            self._value = float(value)
            self._factors = Factors()

    @property
    def value(self):
        """Return numerical value"""
        return self._value

    @property
    def real(self):
        """Return numerical value"""
        # mimic Python float.real
        return self._value

    @property
    def units(self) -> str:
        """Return units"""
        units = REVLU.get(self._factors, str(self._factors))
        return units

    def __repr__(self):
        return f"Quantity('{self!s}')"

    def __str__(self) -> str:
        value = repr(self._value)
        units = REVLU.get(self._factors, str(self._factors))
        return f"{value} {units}".rstrip("1").rstrip()

    def __format__(self, format_spec: str) -> str:
        nfmt = format_spec
        ufmt = None
        format_spec = format_spec.replace("u", " ")
        if " " in format_spec:
            fsl = format_spec.split()
            if len(fsl) > 1:
                nfmt = fsl[0]
                ufmt = fsl[1]
            else:
                nfmt = ""
                ufmt = fsl[0]
        if ufmt is None:
            out_n = self.real
            out_u = REVLU.get(self._factors, str(self._factors))
        else:
            if ufmt == "dBm" and self._factors == Factors(1, 0, 2, 1, -3):
                denominator = Fraction(1, 1000)  # W
                out_n = 10 * np.log10(self.real / denominator)
                out_u = ufmt
            else:
                ufmt_factors = parse(ufmt)
                out_units_factors_ratio = self._factors / ufmt_factors
                offset = 0
                if ufmt_factors.offset > 0:
                    offset = ufmt_factors.offset
                out_n = (self.real - offset) * out_units_factors_ratio.multiplier
                out_factors = Factors(
                    1,
                    0,
                    out_units_factors_ratio.m,
                    out_units_factors_ratio.kg,
                    out_units_factors_ratio.s,
                    out_units_factors_ratio.A,
                    out_units_factors_ratio.K,
                    out_units_factors_ratio.mol,
                    out_units_factors_ratio.cd,
                )
                out_u = str(out_factors)
                if out_u.isdecimal():
                    out_u = ufmt
                elif out_u[0].isdecimal():
                    out_u = out_u.split("*", maxsplit=1)[-1] + "*" + ufmt
                else:
                    out_u = out_u + "*" + ufmt
        out_u = " " + out_u if out_u != "°" else out_u
        return format(out_n, nfmt) + out_u

    def display(self, format_spec: str = ""):
        """display"""
        print(format(self, format_spec))

    def __mul__(self, other):
        if isinstance(other, Number):
            return Quantity(self._value * other, self._factors)
        if isinstance(other, Quantity):
            return Quantity(self._value * other._value, self._factors * other._factors)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Number):
            return Quantity(other * self._value, self._factors)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, Number):
            return Quantity(self._value / other, self._factors)
        if isinstance(other, Quantity):
            return Quantity(self._value / other._value, self._factors / other._factors)
        return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, Number):
            return Quantity(other / self._value, 1 / self._factors)
        return NotImplemented

    def __pow__(self, other):
        if isinstance(other, Number):
            return Quantity(self._value**other, self._factors ** Factors(other))
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, Quantity):
            if not self._factors.is_same_dimension(other._factors):
                raise ValueError(f"{other!r} not of same dimension as {self!r}")
            return Quantity(self._value + other._value, self._factors)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Quantity):
            if not self._factors.is_same_dimension(other._factors):
                raise ValueError(f"{other!r} not of same dimension as {self!r}")
            return Quantity(self._value - other._value, self._factors)
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, Quantity):
            if not self._factors.is_same_dimension(other._factors):
                raise ValueError(f"{other!r} not of same dimension as {self!r}")
            return self._value > other._value
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, Quantity):
            if not self._factors.is_same_dimension(other._factors):
                raise ValueError(f"{other!r} not of same dimension as {self!r}")
            return self._value < other._value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, Quantity):
            if not self._factors.is_same_dimension(other._factors):
                raise ValueError(f"{other!r} not of same dimension as {self!r}")
            return self._value >= other._value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, Quantity):
            if not self._factors.is_same_dimension(other._factors):
                raise ValueError(f"{other!r} not of same dimension as {self!r}")
            return self._value <= other._value
        return NotImplemented
