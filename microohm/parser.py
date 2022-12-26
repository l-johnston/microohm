"""Units expression parser"""
import functools
from fractions import Fraction
from microohm.character_set import NUMBERS
from microohm.exceptions import TokenError
from microohm.factors import Factors
from microohm.lexer import TokenStream
from microohm.units_definitions import PREFIXES, SIUNITS, NONSIUNITS


def singleton(cls):
    """Decorator function to make class 'cls' a singleton"""

    @functools.wraps(cls)
    def single_cls(*args, **kwargs):
        if single_cls.instance is None:
            single_cls.instance = cls(*args, **kwargs)
        return single_cls.instance

    single_cls.instance = None
    return single_cls


@singleton
class Parser:
    """A recursive-decent parser for unit expressions
    Converts unit expression string to SI base unit Factors
    """

    def __init__(self):
        self.ts = TokenStream("")

    def get_expression(self) -> Factors:
        """Expression"""
        t = self.ts.get()
        if t.value[0] in NUMBERS:
            self.ts.putback(t.value)
            f = self.get_numberterm()
        else:
            self.ts.putback(t.value)
            f = self.get_term()
        t = self.ts.get()
        t = t.value
        while True:
            if t == "*":
                f *= self.get_term()
            elif t == "/":
                f /= self.get_term()
            elif t == "":
                break
            else:
                self.ts.putback(t)
                break
            t = self.ts.get()
            t = t.value
        return f

    def get_term(self) -> Factors:
        """Term"""
        f = self.get_unit()
        t = self.ts.get()
        t = t.value
        while True:
            if t == "**":
                f **= self.get_numberterm()
            else:
                self.ts.putback(t)
                break
            t = self.ts.get()
            t = t.value
        return f

    def get_unit(self) -> Factors:
        """Unit"""
        t = self.ts.get()
        t = t.value
        f = Factors()
        if t in NONSIUNITS:
            return NONSIUNITS[t].copy()
        if t in SIUNITS:
            return SIUNITS[t].copy()
        prefixlen = 1
        if t.startswith("da"):
            prefixlen = 2
        if t[:prefixlen] in PREFIXES and t[prefixlen:] in SIUNITS:
            prefix = t[:prefixlen]
            prefix_value = PREFIXES[prefix]
            unit = t[prefixlen:]
            f = SIUNITS[unit].copy()
            # bel (B) is maintained as decibel
            # while its impermissible to attach info to units (SP811 7.5)
            # RF engineers demand dBm and dBc
            if unit in ["B", "Bm", "Bc"]:
                prefix_value *= Fraction(10**1, 1)
            f.multiplier *= prefix_value
            return f
        if t == "(":
            f = self.get_expression()
            t = self.ts.get()
            t = t.value
            if t != ")":
                raise TokenError(f"expect ')', not '{t}'")
            return f
        raise TokenError(f"unknown unit '{t}'")

    def get_numberterm(self, inpar=False) -> Factors:
        """NumberTerm"""
        f = self.get_number()
        t = self.ts.get()
        t = t.value
        while True:
            if inpar and t == "/":
                f /= self.get_number()
            elif inpar and t == "*":
                f *= self.get_number()
            else:
                self.ts.putback(t)
                break
            t = self.ts.get()
            t = t.value
        return f

    def get_number(self) -> Factors:
        """Number"""
        t = self.ts.get()
        t = t.value
        if t == "(":
            f = self.get_numberterm(inpar=True)
            t = self.ts.get()
            if t.value != ")":
                raise TokenError(f"expect ')', not '{t}'")
            return f
        if t.isdecimal():
            return Factors(multiplier=int(t))
        if t == "-":
            return -self.get_number()
        if t == "+":
            return self.get_number()
        raise TokenError(f"expected a number, not {t}")

    @functools.lru_cache(maxsize=128)
    def __call__(self, units: str):
        self.ts = TokenStream(units)
        return self.get_expression()


parse = Parser()
