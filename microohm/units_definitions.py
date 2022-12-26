"""Units definitions based on NIST SP811"""
import math
from fractions import Fraction
from microohm.factors import Factors

PREFIXES = {
    "Y": Fraction(10**24),
    "Z": Fraction(10**21),
    "E": Fraction(10**18),
    "P": Fraction(10**15),
    "T": Fraction(10**12),
    "G": Fraction(10**9),
    "M": Fraction(10**6),
    "k": Fraction(10**3),
    "h": Fraction(10**2),
    "da": Fraction(10**1),
    "": Fraction(1),
    "d": Fraction(1, 10**1),
    "c": Fraction(1, 10**2),
    "m": Fraction(1, 10**3),
    "µ": Fraction(1, 10**6),
    "n": Fraction(1, 10**9),
    "p": Fraction(1, 10**12),
    "f": Fraction(1, 10**15),
    "a": Fraction(1, 10**18),
    "z": Fraction(1, 10**21),
    "y": Fraction(1, 10**24),
}

SIUNITS = {
    "m": Factors(m=1),
    "kg": Factors(kg=1),
    "g": Factors(multiplier=Fraction(1, 1000), kg=1),
    "s": Factors(s=1),
    "A": Factors(A=1),
    "K": Factors(K=1),
    "mol": Factors(mol=1),
    "cd": Factors(cd=1),
    "rad": Factors(),
    "sr": Factors(),
    "deg": Factors(multiplier=Fraction(math.pi) / 180),
    "°": Factors(multiplier=Fraction(math.pi) / 180),
    "Hz": Factors(s=-1),
    "N": Factors(m=1, kg=1, s=-2),
    "Pa": Factors(m=-1, kg=1, s=-2),
    "J": Factors(m=2, kg=1, s=-2),
    "W": Factors(m=2, kg=1, s=-3),
    "C": Factors(s=1, A=1),
    "V": Factors(m=2, kg=1, s=-3, A=-1),
    "F": Factors(m=-2, kg=-1, s=4, A=2),
    "Ω": Factors(m=2, kg=1, s=-3, A=-2),
    "S": Factors(m=-2, kg=-1, s=-2, A=-1),
    "Wb": Factors(m=2, kg=1, s=-2, A=-1),
    "T": Factors(kg=1, s=-2, A=-1),
    "H": Factors(m=2, kg=1, s=-2, A=-2),
    "degC": Factors(offset=273.15, K=1),
    "°C": Factors(offset=273.15, K=1),
    "delta_degC": Factors(K=1),
    "Δ°C": Factors(K=1),
    "lm": Factors(cd=1),
    "lx": Factors(m=-2, cd=1),
    "Bq": Factors(s=-1),
    "Gy": Factors(m=2, s=-2),
    "Sv": Factors(m=2, s=-2),
    "kat": Factors(s=-1, mol=1),
    "L": Factors(multiplier=Fraction(1, 1000), m=3),
    "Np": Factors(),
    "B": Factors(),
    "Bm": Factors(),
    "Bc": Factors(),
}

NONSIUNITS = {
    "Å": Factors(multiplier=1e-10, m=1),
    "ua": Factors(multiplier=1.495979e11, m=1),
    "ch": Factors(multiplier=2.011684e1, m=1),
    "fathom": Factors(multiplier=1.828804, m=1),
    "fermi": Factors(multiplier=1e-15, m=1),
    "ft": Factors(multiplier=3.048e-1, m=1),
    "in": Factors(multiplier=2.54e-2, m=1),
    "µ": Factors(multiplier=1e-6, m=1),
    "mil": Factors(multiplier=2.54e-5, m=1),
    "mi": Factors(multiplier=1.609344e3, m=1),
    "yd": Factors(multiplier=9.144e-1, m=1),
    "oz": Factors(multiplier=2.834952e-2, kg=1),
    "lb": Factors(multiplier=4.535924e-1, kg=1),
    "d": Factors(multiplier=8.64e4, s=1),
    "h": Factors(multiplier=3.6e3, s=1),
    "min": Factors(multiplier=60, s=1),
    "degF": Factors(
        multiplier=Fraction(5, 9), offset=Fraction(4492555643909279, 17592186044416), K=1
    ),
    "°F": Factors(
        multiplier=Fraction(5, 9), offset=Fraction(4492555643909279, 17592186044416), K=1
    ),
    "delta_degF": Factors(multiplier=Fraction(5, 9), offset=0, K=1),
    "Δ°F": Factors(multiplier=Fraction(5, 9), offset=0, K=1),
    "degR": Factors(multiplier=Fraction(10, 18), K=1),
    # BTU IT international table
    "BTU": Factors(multiplier=1.05505585262e3, m=2, kg=1, s=-2),
    # cal IT international table
    "cal": Factors(multiplier=4.1868, m=2, kg=1, s=-2),
    "eV": Factors(multiplier=1.602176e-19, m=2, kg=1, s=-2),
    "lbf": Factors(multiplier=4.448222, m=1, kg=1, s=-2),
    "horsepower": Factors(multiplier=7.46e2, m=2, kg=1, s=-3),
    "atm": Factors(multiplier=1.01325e5, m=-1, kg=1, s=-2),
    "bar": Factors(multiplier=1e5, m=-1, kg=1, s=-2),
    "inHg": Factors(multiplier=3.386389e3, m=-1, kg=1, s=-2),
    "psi": Factors(multiplier=6.894757e3, m=-1, kg=1, s=-2),
    "torr": Factors(multiplier=1.333224e2, m=-1, kg=1, s=-2),
    "rem": Factors(multiplier=1e-2, m=2, s=-2),
    "gal": Factors(multiplier=3.785412e-3, m=3),
}

SIUNITS2 = SIUNITS.copy()
SIUNITS2.pop("rad")
SIUNITS2.pop("sr")
SIUNITS2.pop("degC")
SIUNITS2.pop("delta_degC")
SIUNITS2.pop("Δ°C")
SIUNITS2.pop("Bq")
SIUNITS2.pop("Sv")
SIUNITS2.pop("Np")
SIUNITS2.pop("B")
REVLU = {v: k for k, v in SIUNITS2.items()}
REVLU[Factors()] = "1"  # dimensionless quantity
