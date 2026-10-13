from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

def round_half_away(x: float, digits: int) -> str:
    d = Decimal(repr(round(x, 9))).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
    return str(abs(d) if d.is_zero() else d)

def half_width(est: float, interval, digits: int, scale: float = 1) -> str:
    lo, hi = interval
    return round_half_away(scale * max(est - lo, hi - est), digits)
