"""SI base unit factors"""
from fractions import Fraction
from numbers import Number


class Factors:
    """SI base unit factors"""

    __slots__ = ["multiplier", "offset", "m", "kg", "s", "A", "K", "mol", "cd"]

    def __init__(self, multiplier=1, offset=0, m=0, kg=0, s=0, A=0, K=0, mol=0, cd=0):
        self.multiplier = Fraction(multiplier)
        self.offset = Fraction(offset)
        self.m = Fraction(m)
        self.kg = Fraction(kg)
        self.s = Fraction(s)
        self.A = Fraction(A)
        self.K = Fraction(K)
        self.mol = Fraction(mol)
        self.cd = Fraction(cd)

    def __mul__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(f"{other} must of type {self.__class__}")
        if self.offset != 0 and other.offset != 0:
            raise ValueError("offsets must be zero")
        res = Factors()
        for k in self.__slots__:
            if k == "multiplier":
                setattr(res, k, getattr(self, k) * getattr(other, k))
            else:
                setattr(res, k, getattr(self, k) + getattr(other, k))
        return res

    def __truediv__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(f"{other} must of type {self.__class__}")
        if self.offset != 0 and other.offset != 0:
            raise ValueError("offsets must be zero")
        res = Factors()
        for k in self.__slots__:
            if k == "multiplier":
                setattr(res, k, getattr(self, k) / getattr(other, k))
            elif k == "offset":
                continue
            else:
                setattr(res, k, getattr(self, k) - getattr(other, k))
        return res

    def __rtruediv__(self, other):
        if not isinstance(other, Number):
            raise TypeError(f"{other!r} not a number")
        if self.offset != 0:
            raise ValueError("offset must be zero")
        res = Factors()
        for k in self.__slots__:
            if k == "multiplier":
                setattr(res, k, other / getattr(self, k))
            elif k == "offset":
                continue
            else:
                setattr(res, k, -getattr(self, k))
        return res

    def __neg__(self):
        self.multiplier = -self.multiplier
        return self

    def __pos__(self):
        return self

    def __pow__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(f"{other} must of type {self.__class__}")
        if self.offset != 0 and other.offset != 0:
            raise ValueError("offsets must be zero")
        multiplier = other.multiplier
        res = Factors()
        for k in self.__slots__:
            if k in ["offset", "slots"]:
                continue
            if k == "multiplier":
                setattr(res, k, getattr(self, k) ** multiplier)
            else:
                setattr(res, k, getattr(self, k) * multiplier)
        return res

    def __repr__(self):
        params = ",".join([f"{k}={getattr(self, k)}" for k in self.__slots__])
        return "Factors(" + params + ")"

    def __str__(self):
        res = ""
        for factor in self.__slots__:
            if factor == "offset":
                continue
            fraction = getattr(self, factor)
            if factor == "multiplier":
                if fraction.denominator > 1:
                    res += f"({fraction})"
                elif fraction.numerator > 1:
                    res += f"{fraction.numerator}"
            else:
                if fraction.numerator == 0:
                    continue
                if abs(fraction.denominator) > 1 or fraction < 0:
                    res += f"*{factor}**({fraction})"
                elif abs(fraction.numerator) > 1:
                    res += f"*{factor}**{fraction.numerator}"
                else:
                    res += f"*{factor}"
        if res == "":
            res = "1"
        elif res.startswith("*"):
            res = res[1:]
        return res

    def __eq__(self, other):
        if not isinstance(other, Factors):
            return False
        return all(
            [
                self.multiplier == other.multiplier,
                self.offset == other.offset,
                self.m == other.m,
                self.kg == other.kg,
                self.s == other.s,
                self.A == other.A,
                self.K == other.K,
                self.mol == other.mol,
                self.cd == other.cd,
            ]
        )

    def __hash__(self) -> int:
        return hash(repr(self))

    def copy(self):
        """Return a copy"""
        return Factors(
            self.multiplier, self.offset, self.m, self.kg, self.s, self.A, self.K, self.mol, self.cd
        )

    def is_same_dimension(self, other) -> bool:
        """Return whether 'other' has same SI base unit factors"""
        if not isinstance(other, Factors):
            raise TypeError(f"{other!r} not a Factors")
        return all(
            [
                self.m == other.m,
                self.kg == other.kg,
                self.s == other.s,
                self.A == other.A,
                self.K == other.K,
                self.mol == other.mol,
                self.cd == other.cd,
            ]
        )

    def is_dimensionless(self) -> bool:
        """Return whether dimensionless"""
        return all(
            [
                self.m == 0,
                self.kg == 0,
                self.s == 0,
                self.A == 0,
                self.K == 0,
                self.mol == 0,
                self.cd == 0,
            ]
        )
