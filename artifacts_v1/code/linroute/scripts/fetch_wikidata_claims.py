#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from linroute.engineered import (
    first_item, item_candidates, usable_claims)

REPO_ROOT = Path(__file__).resolve().parents[1]

API = "https://www.wikidata.org/w/api.php"

USER_AGENT = (
    "linroute-engineered-relations/1.0 "
    "(research code; engineered town relations)"
)

BATCH = 50

COUNTRY = "P17"
OFFICIAL_LANGUAGE = "P37"

def batch_digest(ids: list[str]) -> str:
    return hashlib.sha256("|".join(ids).encode("utf-8")).hexdigest()[:10]

def town_qids(records: Path) -> list[str]:
    seen: dict[str, None] = {}
    with open(records, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            qid = (json.loads(line).get("coordinate_provenance") or {}).get("qid")
            if isinstance(qid, str) and qid.startswith("Q"):
                seen.setdefault(qid, None)
    return list(seen)

def fetch(ids: list[str], *, timeout: float) -> bytes:
    query = urllib.parse.urlencode({
        "action": "wbgetentities",
        "ids": "|".join(ids),
        "props": "claims|labels",
        "languages": "en",
        "format": "json",
    })
    request = urllib.request.Request(
        f"{API}?{query}", headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()

def run_pass(ids: list[str], kind: str, raw: Path, args) -> dict[str, dict]:
    entities: dict[str, dict] = {}
    for n, start in enumerate(range(0, len(ids), args.batch)):
        chunk = ids[start:start + args.batch]
        path = raw / f"wbget-{kind}-{n}-{batch_digest(chunk)}.json"
        if path.exists() and not args.refetch:
            print(f"  {path.name}: already cached ({len(chunk)} ids)")
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            if n or kind != "towns":
                time.sleep(args.sleep)
            print(f"  {path.name}: fetching {len(chunk)} ids")
            body = fetch(chunk, timeout=args.timeout)
            payload = json.loads(body.decode("utf-8"))
            if "entities" not in payload:
                raise SystemExit(
                    f"{path.name}: the response carries no 'entities' key; "
                    f"nothing was written.  Server said: "
                    f"{json.dumps(payload)[:400]}"
                )
            path.write_bytes(body)
            path.with_suffix(".ids").write_text(
                "\n".join(chunk) + "\n", encoding="utf-8"
            )
        for qid, entity in (payload.get("entities") or {}).items():
            if entity.get("missing") is None:
                entities[qid] = entity
    return entities

def items_pointed_at(entities: dict[str, dict], prop: str) -> list[str]:
    seen: dict[str, None] = {}
    for entity in entities.values():
        claims = (entity.get("claims") or {}).get(prop)
        if not usable_claims(claims):
            continue
        qid, _method = first_item(claims)
        if qid:
            seen.setdefault(qid, None)
    return list(seen)

def items_offered_by(entities: dict[str, dict], prop: str) -> list[str]:
    seen: dict[str, None] = {}
    for entity in entities.values():
        for candidate in item_candidates((entity.get("claims") or {}).get(prop)):
            seen.setdefault(candidate["qid"], None)
    return list(seen)

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--records",
        default=str(REPO_ROOT / "data/relations/engineered/latitude/records.jsonl"),
        help="the frozen records whose coordinate_provenance.qid supplies the town ids",
    )
    p.add_argument(
        "--raw-out", required=True,
        help="the fresh-subjects cache's raw/ directory, beside the wdqs-*.json files",
    )
    p.add_argument("--batch", type=int, default=BATCH,
                   help="ids per request; the API's anonymous limit is 50")
    p.add_argument("--sleep", type=float, default=1.0,
                   help="seconds between requests, as a courtesy to the service")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--refetch", action="store_true",
                   help="ask again for batches already cached, overwriting them")
    args = p.parse_args(argv)

    raw = Path(args.raw_out)
    raw.mkdir(parents=True, exist_ok=True)

    towns = town_qids(Path(args.records))
    print(f"towns: {len(towns)} distinct QIDs")
    town_entities = run_pass(towns, "towns", raw, args)

    countries = items_pointed_at(town_entities, COUNTRY)
    print(f"countries: {len(countries)} distinct QIDs")
    country_entities = run_pass(countries, "countries", raw, args)

    languages = items_offered_by(country_entities, OFFICIAL_LANGUAGE)
    print(f"languages: {len(languages)} distinct candidate QIDs")
    language_entities = run_pass(languages, "languages", raw, args)

    print(
        f"cached {len(town_entities)} towns, {len(country_entities)} countries "
        f"and {len(language_entities)} languages into {raw}"
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
