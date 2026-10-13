from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)

def town_length(text: str) -> int:
    return len(nfc(text))

def first_initial(text: str) -> str | None:
    value = nfc(text).casefold()
    return value[0] if value and value[0].isalpha() else None

@dataclass(frozen=True)
class IntegerInterval:
    low: int
    high: int | None
    class_id: str
    answer: str

    def contains(self, value: int) -> bool:
        return value >= self.low and (self.high is None or value <= self.high)

def fixed_length_intervals(lengths: Sequence[int], target_classes: int) -> list[IntegerInterval]:
    if target_classes < 1 or not lengths:
        raise ValueError("need lengths and target_classes >= 1")
    values = sorted(lengths)
    cuts: list[int] = []
    for i in range(1, target_classes):
        pos = math.ceil(i * len(values) / target_classes)
        while pos < len(values) and values[pos - 1] == values[pos]:
            pos += 1
        if 0 < pos < len(values) and values[pos - 1] not in cuts:
            cuts.append(values[pos - 1])
    lows = [min(values)] + [x + 1 for x in cuts]
    highs: list[int | None] = cuts + [None]
    ordinal = ("shortest", "short", "medium", "long", "longest")
    return [IntegerInterval(lo, hi, f"len-{lo}-{hi or 'plus'}",
                            f"{ordinal[i] if len(lows) == 5 else f'length band {i + 1}'} ({lo}–{hi} characters)" if hi
                            else f"{ordinal[i] if len(lows) == 5 else f'length band {i + 1}'} ({lo}+ characters)")
            for i, (lo, hi) in enumerate(zip(lows, highs))]

def assign_interval(value: int, intervals: Sequence[IntegerInterval]) -> IntegerInterval:
    for interval in intervals:
        if interval.contains(value):
            return interval
    raise ValueError(f"value {value} outside intervals")

def latitude_bands(values: Sequence[float], target_classes: int, min_support: int) -> list[tuple[float, float]]:
    if not values or target_classes < 1 or min_support < 1:
        raise ValueError("need values and positive target_classes/min_support")
    groups = sorted(Counter(float(v) for v in values).items())
    total = sum(n for _, n in groups)
    bands: list[list[tuple[float, int]]] = [[]]
    running = 0
    next_cut = total / target_classes
    for value, count in groups:
        if bands[-1] and running >= next_cut and len(bands) < target_classes:
            bands.append([])
            next_cut = total * len(bands) / target_classes
        bands[-1].append((value, count))
        running += count
    while len(bands) > 1:
        small = next((i for i, b in enumerate(bands) if sum(n for _, n in b) < min_support), None)
        if small is None:
            break
        neighbour = 1 if small == 0 else small - 1 if small == len(bands) - 1 else (
            small - 1 if sum(n for _, n in bands[small - 1]) <= sum(n for _, n in bands[small + 1]) else small + 1)
        lo, hi = sorted((small, neighbour))
        bands[lo].extend(bands[hi]); del bands[hi]
    return [(b[0][0], b[-1][0]) for b in bands]

def assign_latitude(value: float, bands: Sequence[tuple[float, float]]) -> int:
    for i, (low, high) in enumerate(bands):
        if low <= value <= high:
            return i
    raise ValueError(f"latitude {value} outside bands")

def fitting_rows(total_rows: int, outer_folds: int, inner_folds: int) -> int:
    if min(total_rows, outer_folds, inner_folds) < 2:
        raise ValueError("total rows and fold counts must be >= 2")
    outer_train = total_rows - math.ceil(total_rows / outer_folds)
    return outer_train - math.ceil(outer_train / inner_folds)

def q_feasible(total_rows: int, q: int, outer_folds: int, inner_folds: int) -> bool:
    return fitting_rows(total_rows, outer_folds, inner_folds) >= q + 1

UNIT_TO_METRES = {"Q11573": 1.0, "Q3710": 0.3048}

_WIKIDATA_TIME = re.compile(r"^([+-]\d+)-(.*)$")

def claim_value(claim: dict) -> object | None:
    snak = claim.get("mainsnak") or {}
    if snak.get("snaktype") != "value":
        return None
    return (snak.get("datavalue") or {}).get("value")

def point_in_time(claim: dict) -> str | None:
    for qualifier in (claim.get("qualifiers") or {}).get("P585", []):
        if qualifier.get("snaktype") != "value":
            continue
        time = ((qualifier.get("datavalue") or {}).get("value") or {}).get("time")
        if isinstance(time, str) and time:
            return time
    return None

def time_key(time: str) -> tuple[int, str]:
    match = _WIKIDATA_TIME.match(time)
    if not match:
        raise ValueError(f"not a Wikidata time literal: {time!r}")
    return int(match.group(1)), match.group(2)

def usable_claims(claims: Iterable[dict] | None) -> list[dict]:
    return [c for c in (claims or [])
            if c.get("rank") != "deprecated" and claim_value(c) is not None]

def select_claim(claims: Iterable[dict] | None) -> tuple[dict | None, str]:
    usable = usable_claims(claims)
    if not usable:
        return None, "no_claim"
    preferred = [c for c in usable if c.get("rank") == "preferred"]
    if len(preferred) == 1:
        return preferred[0], "preferred_rank"
    pool = preferred or usable
    dated = [(time_key(t), c) for c in pool for t in [point_in_time(c)] if t]
    if dated:
        latest = max(key for key, _ in dated)
        tied = [c for key, c in dated if key == latest]
        if len(tied) > 1:
            return None, "ambiguous_claim"
        return tied[0], "latest_point_in_time"
    if len(pool) == 1:
        return pool[0], "sole_claim"
    return None, "ambiguous_claim"

def quantity(claim: dict) -> tuple[float | None, str | None]:
    value = claim_value(claim)
    if not isinstance(value, dict) or "amount" not in value:
        return None, None
    unit = str(value.get("unit") or "").rsplit("/", 1)[-1]
    return float(value["amount"]), unit

def metres(claim: dict) -> tuple[float | None, str]:
    amount, unit = quantity(claim)
    if amount is None:
        return None, "no_value"
    factor = UNIT_TO_METRES.get(unit)
    if factor is None:
        return None, "unit_unsupported"
    return amount * factor, "resolved"

def count(claim: dict) -> tuple[float | None, str]:
    amount, _unit = quantity(claim)
    if amount is None:
        return None, "no_value"
    return amount, "resolved"

def item_candidates(claims: Iterable[dict] | None) -> list[dict]:
    usable = [c for c in usable_claims(claims)
              if isinstance(claim_value(c), dict) and claim_value(c).get("id")]
    preferred = [c for c in usable if c.get("rank") == "preferred"]
    pool = preferred or usable
    out, seen = [], set()
    for claim in pool:
        qid = claim_value(claim)["id"]
        if qid in seen:
            continue
        seen.add(qid)
        out.append({"qid": qid, "rank": claim.get("rank")})
    return out

def first_item(claims: Iterable[dict] | None) -> tuple[str | None, str]:
    candidates = item_candidates(claims)
    if not candidates:
        return None, "no_claim"
    rank = "preferred_rank" if candidates[0]["rank"] == "preferred" else "first_listed"
    return candidates[0]["qid"], rank

def speaker_count(claims: Iterable[dict] | None) -> tuple[float | None, str]:
    amounts = []
    for claim in usable_claims(claims):
        amount, _unit = quantity(claim)
        if amount is not None:
            amounts.append((point_in_time(claim), amount))
    if not amounts:
        return None, "no_claim"
    dated = [(time_key(t), a) for t, a in amounts if t]
    if dated:
        latest = max(key for key, _ in dated)
        return max(a for key, a in dated if key == latest), "latest_point_in_time"
    return max(a for _, a in amounts), "largest_amount"

def official_language(
    claims: Iterable[dict] | None, speakers: dict[str, float | None]
) -> tuple[str | None, str, list[dict]]:
    candidates = [dict(c, speakers=speakers.get(c["qid"]))
                  for c in item_candidates(claims)]
    if not candidates:
        return None, "no_claim", []
    if len(candidates) > 1:
        return None, "ambiguous_language", candidates
    return candidates[0]["qid"], "sole_official_language", candidates

def english_label(entity: dict | None) -> str | None:
    if not isinstance(entity, dict):
        return None
    label = ((entity.get("labels") or {}).get("en") or {}).get("value")
    return label if isinstance(label, str) and label.strip() else None

def class_slug(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    parts = [p for p in re.split(r"[^A-Za-z0-9]+", folded.lower()) if p]
    return "-".join(parts) or "unnamed"
