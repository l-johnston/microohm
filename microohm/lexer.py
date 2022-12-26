"""Units expression lexical analyzer"""
from microohm.character_set import LENMAXINT, LENMAXSTR, NUMBERS, LETTERS
from microohm.exceptions import TokenError


class Token:
    """Token"""

    def __init__(self, ch: str):
        self.value = ch

    def __repr__(self):
        return f"Token('{self.value}')"

    def __str__(self):
        return self.value

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(f"{other} not a Token")
        return self.value == other.value

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(f"{other} not a Token")
        return Token(self.value + other.value)

    def __len__(self):
        return len(self.value)


class TokenStream:
    """Extract tokens from unit expression string"""

    def __init__(self, expression: str):
        self.expression = expression

    def __repr__(self):
        return f"TokenStream('{self.expression}')"

    def __str__(self):
        return f"{self.expression}"

    def putback(self, ch: str) -> None:
        """put token into the buffer because 'ch' starts a different type of token"""
        self.expression = ch + self.expression

    def _get_numbers(self, token: Token) -> Token:
        """compose a numeric token"""
        if len(self.expression) == 0:
            return token
        ch = self.expression[0]
        self.expression = self.expression[1:]
        if ch not in NUMBERS:
            self.putback(ch)
            return token
        token = token + Token(ch)
        if len(token) > LENMAXINT:
            raise TokenError(f"{token} exceeds max int")
        return self._get_numbers(token)

    def _get_letters(self, token: Token) -> Token:
        """compose an alphabetic token"""
        if len(self.expression) == 0:
            return token
        ch = self.expression[0]
        self.expression = self.expression[1:]
        if ch not in LETTERS:
            self.putback(ch)
            return token
        token = token + Token(ch)
        if len(token) > LENMAXSTR:
            raise TokenError(f"{token} exceeds max str length")
        return self._get_letters(token)

    def get(self) -> Token:
        """get a token"""
        if len(self.expression) == 0:
            return Token("")
        ch = self.expression[0]
        self.expression = self.expression[1:]
        if ch in ["-", "+", "/", "(", ")"]:
            return Token(ch)
        if ch == "*":
            if self.expression.startswith("*"):
                ch += "*"
                self.expression = self.expression[1:]
            return Token(ch)
        if ch in NUMBERS:
            return self._get_numbers(Token(ch))
        if ch in LETTERS:
            return self._get_letters(Token(ch))
        raise TokenError(f"{ch} not a simple token")

    def __iter__(self):
        while True:
            if len(self.expression) == 0:
                return None
            yield self.get()
