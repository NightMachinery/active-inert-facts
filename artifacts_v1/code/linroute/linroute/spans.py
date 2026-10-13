from __future__ import annotations

from dataclasses import dataclass

import torch

@dataclass(frozen=True)
class Encoded:
    ids: tuple[int, ...]
    span: tuple[int, int]

def fill(template: str, value: str) -> tuple[str, int, int]:
    if template.count("{}") != 1:
        raise ValueError(f"template must contain exactly one '{{}}': {template!r}")
    head, tail = template.split("{}")
    return head + value + tail, len(head), len(head) + len(value)

def append(text: str, value: str, sep: str = " ") -> tuple[str, int, int]:
    start = len(text) + len(sep)
    return text + sep + value, start, start + len(value)

def token_span(
    offsets: list[tuple[int, int]], char_start: int, char_end: int
) -> tuple[int, int]:
    hits = [
        i
        for i, (a, b) in enumerate(offsets)
        if a < char_end and b > char_start and b > a
    ]
    if not hits:
        raise ValueError(f"no token covers characters [{char_start}, {char_end})")
    return hits[0], hits[-1] + 1

def encode_span(tokenizer, text: str, char_start: int, char_end: int) -> Encoded:
    if not tokenizer.is_fast:
        raise ValueError("a fast tokenizer is required for offset mapping")
    enc = tokenizer(text, return_offsets_mapping=True)
    return Encoded(
        ids=tuple(enc["input_ids"]),
        span=token_span(enc["offset_mapping"], char_start, char_end),
    )

def span_matches(tokenizer, encoded: Encoded, value: str) -> bool:
    text = tokenizer.decode(list(encoded.ids[encoded.span[0] : encoded.span[1]]))
    return text.strip() == value.strip()

def pad_left(
    sequences: list[tuple[int, ...]], pad_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(s) for s in sequences)
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(sequences), width), dtype=torch.long)
    for i, s in enumerate(sequences):
        ids[i, width - len(s) :] = torch.tensor(s, dtype=torch.long)
        mask[i, width - len(s) :] = 1
    return ids, mask
