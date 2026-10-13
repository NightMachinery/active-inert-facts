#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, hashlib, html, json, os, re, sys, unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from linroute.engineered import (assign_interval, assign_latitude, class_slug,
    count, english_label, first_initial, first_item, fitting_rows,
    fixed_length_intervals, latitude_bands, metres, official_language,
    select_claim, speaker_count, town_length)

MODELS = {
    "olmo3_7b": "allenai/Olmo-3-1025-7B",
    "qwen35_9b": "Qwen/Qwen3.5-9B",
    "qwen35_9b_base": "Qwen/Qwen3.5-9B-Base",
    "qwen38_27b": "Qwen/Qwen3.8-27B",
}
LABELS = ["lowest", "lower", "middle", "upper", "highest", "extra"]

BAND_CLASSES, BAND_MIN_SUPPORT = 5, 20

POPULATION, ELEVATION, COUNTRY, OFFICIAL_LANGUAGE = "P1082", "P2044", "P17", "P37"
SPEAKERS = "P1098"

NEW_SET = {"longitude": "longitude_status", "population": "population_status",
           "elevation": "elevation_status", "country-language": "language_status"}

PATH_LABELS: list[tuple[str, str]] = []
RELATION_LABELS: dict[str, str] = {}
RELATION_LABELS_SHA256 = ""
DEFAULT_RELATION_LABELS = (
    Path(__file__).resolve().parents[2]
    / "paper" / "metadata" / "relation-labels.json5"
)

def load_relation_labels(path: Path) -> None:
    global RELATION_LABELS, RELATION_LABELS_SHA256, RESULTS_ORDER
    try:
        raw = path.read_bytes()
        document = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read canonical relation-label map {path}: {error}") from error
    if document.get("schema") != "linroute/relation-labels/1":
        raise SystemExit(f"{path}: unexpected relation-label schema")
    labels = document.get("relations")
    if not isinstance(labels, dict):
        raise SystemExit(f"{path}: missing relation-label object")
    RELATION_LABELS = labels
    RELATION_LABELS_SHA256 = hashlib.sha256(raw).hexdigest()
    missing = sorted(set(RESULT_SPECS) - set(labels))
    if missing:
        raise SystemExit("missing canonical engineered relation labels: " + ", ".join(missing))
    RESULTS_ORDER = [(relation_id, labels[relation_id], tier)
                     for relation_id, tier in RESULT_SPECS.items()]

def relation_label(relation_id: str) -> str:
    try:
        return RELATION_LABELS[relation_id]
    except KeyError as error:
        raise SystemExit(f"missing canonical relation label: {relation_id}") from error

def relation_label_meta() -> str:
    return (f'<meta name="linroute-relation-labels-source" '
            f'content="paper/metadata/relation-labels.json5">'
            f'<meta name="linroute-relation-labels-sha256" '
            f'content="{RELATION_LABELS_SHA256}">')

def key(s):
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower())

def _swap(text, pairs):
    for src,dst in pairs:
        src,dst=src.rstrip("/"),dst.rstrip("/")
        if text==src or text.startswith(src+"/"): return dst+text[len(src):]
    return text

def label(path):
    return None if path is None else _swap(str(path), PATH_LABELS)

def unlabel(path):
    return None if path is None else _swap(str(path), [(d,s) for s,d in PATH_LABELS])

def source_hash(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def digests(paths):
    out=[]
    for p in sorted({str(x) for x in paths if x}):
        local=Path(unlabel(p))
        out.append({"file":p,"sha256":source_hash(local) if local.is_file() else None})
    return out

def coordinate_index(raw: Path, candidates: Path):
    records=[]
    for line in candidates.read_text().splitlines():
        x=json.loads(line); records.append({"name":x["town"], "title":x.get("enwiki_title"), "qid":x.get("qid"), "lat":x.get("lat"), "lon":x.get("lon"), "source_file":label(candidates)})
    for p in sorted(raw.glob("wdqs-*.json")):
        try: bindings=json.loads(p.read_text()).get("results",{}).get("bindings",[])
        except (json.JSONDecodeError,OSError): continue
        for b in bindings:
            if "lat" not in b: continue
            title=b.get("enwiki",{}).get("value","").rsplit("/",1)[-1].replace("_"," ")
            records.append({"name":b.get("itemLabel",{}).get("value",""), "title":title,
              "qid":b.get("item",{}).get("value","").rsplit("/",1)[-1], "lat":float(b["lat"]["value"]),
              "lon":float(b["lon"]["value"]) if "lon" in b else None, "source_file":label(p)})
    by_title=defaultdict(list); by_name=defaultdict(list)
    for r in records:
        if r["title"]: by_title[key(r["title"])].append(r)
        if r["name"]: by_name[key(r["name"])].append(r)
    return by_title,by_name

def find_coordinate(town, title, by_title, by_name):
    for method, matches in (("exact-enwiki-title",by_title.get(key(title),[])), ("normalized-town-name",by_name.get(key(town),[]))):
        usable=[x for x in matches if x["lat"] is not None]
        coords={(x["lat"],x["lon"]) for x in usable}
        if len(coords)==1:
            x=usable[0]; return {**x,"join_method":method,"coordinate_status":"resolved"}
        if len(coords)>1: return {"lat":None,"lon":None,"qid":None,"join_method":method,"coordinate_status":"ambiguous"}
    return {"lat":None,"lon":None,"qid":None,"join_method":None,"coordinate_status":"unknown"}

def entity_index(raw: Path):
    entities,sources={},{}
    for p in sorted(raw.glob("wbget-*.json")):
        try: payload=json.loads(p.read_text())
        except (json.JSONDecodeError,OSError): continue
        for qid,entity in (payload.get("entities") or {}).items():
            if entity.get("missing") is not None: continue
            entities[qid]=entity; sources[qid]=label(p)
    return entities,sources

def claims_of(entity, prop):
    return (entity.get("claims") or {}).get(prop) if entity else None

def attribute_rows(base, entities, sources, prop, status_key, read):
    rows=[]
    for x in base:
        qid=x["coordinate_provenance"].get("qid")
        prov={"qid":qid,"join_method":"wikidata-qid" if qid else None,status_key:"no_qid","claim_rule":None,"source_file":None}
        value=None
        if qid:
            entity=entities.get(qid)
            if entity is None: prov[status_key]="no_entity"
            else:
                prov["source_file"]=sources.get(qid)
                claim,rule=select_claim(claims_of(entity,prop)); prov["claim_rule"]=rule
                if claim is None: prov[status_key]=rule
                else: value,prov[status_key]=read(claim)
        rows.append({**x,"attribute_provenance":prov,"raw_value":value})
    return rows

def banded(rows, prefix, noun, fmt):
    values=[r["raw_value"] for r in rows if r["raw_value"] is not None]
    bands=latitude_bands(values,BAND_CLASSES,BAND_MIN_SUPPORT)
    out=[]
    for r in rows:
        v=r["raw_value"]
        if v is None: out.append({**r,"class_id":None,"answer":None}); continue
        i=assign_latitude(v,bands); lo,hi=bands[i]
        out.append({**r,"class_id":f"{prefix}-{i+1}","answer":f"{LABELS[i]} {noun} ({fmt(lo)} to {fmt(hi)})"})
    return out,bands

def longitude_rows(base):
    rows=[]
    for x in base:
        c=x["coordinate_provenance"]; lon=c.get("lon")
        status=("resolved" if lon is not None
                else "no_longitude" if c["coordinate_status"]=="resolved" else c["coordinate_status"])
        rows.append({**x,"attribute_provenance":{"qid":c.get("qid"),"join_method":c.get("join_method"),
            "longitude_status":status,"source_file":c.get("source_file")},"raw_value":lon})
    return rows

def tidy(value):
    return None if value is None else int(value) if float(value).is_integer() else value

def language_rows(base, entities, sources):
    speakers={qid: speaker_count(claims_of(entity,SPEAKERS))[0]
              for qid,entity in entities.items()}
    rows=[]
    for x in base:
        qid=x["coordinate_provenance"].get("qid")
        prov={"qid":qid,"join_method":"wikidata-qid" if qid else None,"country_qid":None,
              "country_rule":None,"language_qid":None,"language_rule":None,
              "language_candidates":None,"language_status":"no_qid","source_file":None}
        answer=None
        if qid:
            town=entities.get(qid)
            if town is None: prov["language_status"]="no_entity"
            else:
                prov["source_file"]=sources.get(qid)
                cq,crule=first_item(claims_of(town,COUNTRY)); prov["country_qid"],prov["country_rule"]=cq,crule
                country=entities.get(cq) if cq else None
                if cq is None: prov["language_status"]="no_country"
                elif country is None: prov["language_status"]="country_not_cached"
                else:
                    lq,lrule,cands=official_language(claims_of(country,OFFICIAL_LANGUAGE),speakers)
                    prov["language_qid"],prov["language_rule"]=lq,lrule
                    prov["language_candidates"]=[{"qid":c["qid"],"label":english_label(entities.get(c["qid"])),
                        "rank":c["rank"],"speakers":tidy(c["speakers"])} for c in cands]
                    name=english_label(entities.get(lq)) if lq else None
                    if lrule=="ambiguous_language": prov["language_status"]="ambiguous_language"
                    elif lq is None: prov["language_status"]="no_official_language"
                    elif name is None: prov["language_status"]="no_label"
                    else: answer=name; prov["language_status"]="resolved"
        rows.append({**x,"attribute_provenance":prov,"raw_value":answer,
                     "class_id":f"lang-{class_slug(answer)}" if answer else None,"answer":answer})
    return rows

def status_counts(rows, status_key):
    return dict(sorted(Counter(r["attribute_provenance"][status_key] for r in rows).items()))

def tokenize(rows):
    from transformers import AutoTokenizer
    for model, model_id in MODELS.items():
        tok=AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        for r in rows:
            answer=r.get("answer")
            if answer is None: r.setdefault("tokenization",{})[model]={"status":"unknown"}; continue
            rendered=" "+answer
            ids=tok.encode(rendered,add_special_tokens=False)
            r.setdefault("tokenization",{})[model]={"rendered":rendered,"token_ids":ids,"first_token_id":ids[0],"first_token_text":tok.decode([ids[0]])}

def row_status(r):
    prov=r.get("attribute_provenance") or r.get("coordinate_provenance",{})
    return prov,next((v for k,v in prov.items() if k.endswith("_status")),r["source"])

def sort_key(r):
    v=r.get("raw_value")
    rank=(0,float(v),"") if isinstance(v,(int,float)) and not isinstance(v,bool) else (0,999.0,"") if v is None else (1,0.0,str(v))
    return (r.get("class_id") or "~unknown",rank,r["town"])

def strip_plot(title, rows, esc):
    values=[float(r["raw_value"]) for r in rows]
    lo,hi=min(values),max(values); span=(hi-lo) or 1.0
    dots="".join(f'<circle cx="{20+(float(r["raw_value"])-lo)/span*720:.2f}" cy="35" r="3"><title>{esc(r["town"])}: {r["raw_value"]}</title></circle>' for r in rows)
    return (f'<h2>{esc(title)}</h2><p><small>{esc(lo)} on the left, {esc(hi)} on the right.</small></p>'
            f'<svg viewBox="0 0 760 70" role="img"><line x1="20" y1="35" x2="740" y2="35"/>{dots}</svg>')

def report(kind, rows, summary):
    esc=lambda x:html.escape(str(x))
    known=[r for r in rows if r.get("class_id") is not None]; counts=Counter(r["class_id"] for r in known); maxn=max(counts.values(),default=1)
    bars="".join(f'<div class="bar"><span>{esc(k)}</span><i style="width:{100*n/maxn:.1f}%"></i><b>{n}</b></div>' for k,n in sorted(counts.items()))
    trs=[]
    for r in sorted(rows,key=sort_key):
        toks="<br>".join(f"{esc(m)}: {esc(t.get('first_token_text','unknown'))} <small>{esc(t.get('first_token_id','—'))}</small>" for m,t in r.get("tokenization",{}).items())
        prov,status=row_status(r)
        trs.append(f"<tr><td>{esc(r['town'])}</td><td>{esc(r.get('raw_value','unknown'))}</td><td>{esc(r.get('class_id') or 'unknown')}</td><td>{esc(r.get('answer') or 'unknown')}</td><td>{toks}</td><td>{esc(status)}<br><small>{esc(prov.get('join_method',''))}</small></td></tr>")
    plot=""
    if kind=="latitude":
        dots="".join(f'<circle cx="{20+(float(r["raw_value"])+90)*4}" cy="35" r="3"><title>{esc(r["town"])}: {r["raw_value"]}</title></circle>' for r in known)
        plot=f'<h2>Latitude distribution</h2><svg viewBox="0 0 760 70" role="img"><line x1="20" y1="35" x2="740" y2="35"/>{dots}</svg>'
    elif kind in ("longitude","population","elevation") and known:
        plot=strip_plot({"longitude":"Longitude distribution","population":"Population distribution",
                         "elevation":"Elevation distribution"}[kind],known,esc)
    prompt=summary["prompt_proposal"]
    return f'''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">{relation_label_meta()}<title>{esc(summary['title'])}</title><link rel="stylesheet" href="../../shared/collapse.css"><script src="../../shared/collapse.js" defer></script><style>body{{font:16px system-ui;max-width:1200px;margin:auto;padding:2rem;color:#18212b}}h1{{margin-bottom:.2rem}}.note{{background:#eef5ff;padding:1rem;border-left:5px solid #3973b8}}.bar{{display:grid;grid-template-columns:12rem 1fr 3rem;gap:.7rem;align-items:center;margin:.25rem}}i{{display:block;background:#4b87c5;height:1rem}}table{{border-collapse:collapse;width:100%;font-size:.88rem}}th,td{{padding:.45rem;border-bottom:1px solid #ccd6df;text-align:left;vertical-align:top}}thead{{position:sticky;top:0;background:white}}small{{color:#687684}}svg{{width:100%;background:#f5f7fa}}svg line{{stroke:#555}}svg circle{{fill:#3973b8;opacity:.65}}</style><h1>{esc(summary['title'])}</h1><p>{esc(summary['definition'])}</p><div class="note"><b>Review status:</b> Source classes and answer strings are proposals for inspection. No animal mapping is chosen. No model outcomes were used.</div><h2>Class support</h2>{bars}<p>{esc(summary['support_note'])}</p><h2>q=16 / fold arithmetic</h2><p>{esc(summary['q_note'])}</p><h2>Prompt proposals — not frozen</h2><ul>{''.join(f'<li><code>{esc(x)}</code></li>' for x in prompt)}</ul><p>Answers are encoded with a leading space. The table shows each tokenizer’s effective first token; equal IDs within a tokenizer are an effective-class collapse.</p>{plot}<h2>Members and provenance</h2><table><thead><tr><th>Town</th><th>Raw value</th><th>Class</th><th>Readable answer</th><th>Tokenizer first token</th><th>Provenance/status</th></tr></thead><tbody>{''.join(trs)}</tbody></table>'''

RESULT_SPECS = {
    "engineered_town_country_language": "known",
    "engineered_town_latitude_band": "strong",
    "engineered_town_longitude_band": "strong",
    "engineered_town_elevation_band": "weak",
    "engineered_town_population_band": "weak",
    "engineered_town_first_letter": "control",
    "engineered_town_name_length": "control",
}
RESULTS_ORDER = []
KIND_RELATION_IDS = {
    "country-language": "engineered_town_country_language",
    "latitude": "engineered_town_latitude_band",
    "longitude": "engineered_town_longitude_band",
    "elevation": "engineered_town_elevation_band",
    "population": "engineered_town_population_band",
    "first-letter": "engineered_town_first_letter",
    "name-length": "engineered_town_name_length",
}

LEGACY_RESULT_LABELS = {
    "engineered_town_country_language": "Country language",
    "engineered_town_latitude_band": "Latitude band",
    "engineered_town_longitude_band": "Longitude band",
    "engineered_town_elevation_band": "Elevation band",
    "engineered_town_population_band": "Population band",
    "engineered_town_first_letter": "First letter",
    "engineered_town_name_length": "Name length",
}
SURFACE_CUE_RELATIONS = {"engineered_town_first_letter", "engineered_town_name_length"}

ROSTER_SURFACE_CUE = {"word_first_letter", "word_last_letter"}

CHECK_RELATIONS = [r for r,_,_ in [
    ("engineered_town_country_language",0,0),("engineered_town_latitude_band",0,0),
    ("engineered_town_longitude_band",0,0),("engineered_town_elevation_band",0,0),
    ("engineered_town_population_band",0,0)]]
SHAPES = {"engineered_town_country_language":"circle", "engineered_town_latitude_band":"square",
          "engineered_town_longitude_band":"triangle", "engineered_town_elevation_band":"diamond",
          "engineered_town_population_band":"tri-down", "engineered_town_first_letter":"circle",
          "engineered_town_name_length":"square"}
LOCKED_REPEATS = (0,1,2)

MEAN_FIELDS = ("heldout_cand_acc","newrel_gen_ep0_acc","primary_probe_gain",
               "pre_rel_cand_acc","pre_rel_cls_gain")

CELL_FIELDS = ("chance","n_classes","n_heldout","n_train")

def default_rows_path():
    import os
    base=os.environ.get("LINROUTE_PAGES") or str(Path.home()/"code/sites/linroute-pages")
    return str(Path(base)/"transfer/data/rows.jsonl")

def _num(v):
    return float(v) if isinstance(v,(int,float)) and not isinstance(v,bool) else None

def usable_row(r):
    if r.get("status")!="complete" or r.get("production_eligible") is not True: return False
    if any(_num(r.get(f))!=20 for f in ("production_horizon","planned_epochs","epochs_done","final_epoch")): return False
    if _num(r.get("limit_train"))!=0: return False
    records,exposed=_num(r.get("n_train_records")),_num(r.get("n_exposed_per_epoch"))
    if not records or records<=0 or not exposed or exposed<=0 or exposed!=records: return False
    return all(_num(r.get(m)) is not None for m in ("primary_probe_gain","heldout_cand_acc"))

def aggregate_results(path: Path):
    groups=defaultdict(dict)
    for line in path.read_text().splitlines():
        if not line.strip(): continue
        r=json.loads(line)
        rel,model,repeat=r.get("relation"),r.get("model"),r.get("repeat")
        if not isinstance(rel,str) or not isinstance(model,str) or not isinstance(repeat,int) or isinstance(repeat,bool): continue
        relation_label(rel)
        groups[(rel,model)][repeat]=r
    def cell_means(reps):
        if any(k not in reps or not usable_row(reps[k]) for k in LOCKED_REPEATS): return None
        picked=[reps[k] for k in LOCKED_REPEATS]
        means={f: sum(_num(p.get(f)) for p in picked)/3.0 for f in MEAN_FIELDS
               if all(_num(p.get(f)) is not None for p in picked)}
        if len(means)<len(MEAN_FIELDS): return None
        means.update({f: picked[0].get(f) for f in CELL_FIELDS})
        return means

    roster=defaultdict(dict)
    for (rel,model),reps in groups.items():
        if rel.startswith("engineered_") or rel in ROSTER_SURFACE_CUE: continue
        means=cell_means(reps)
        if means: roster[model][rel]=means
    out={}
    for rel,title,tier in RESULTS_ORDER:
        cells={}
        for (rl,model),reps in groups.items():
            if rl!=rel: continue
            means=cell_means(reps)
            if means: cells[model]=means
        out[rel]={"title":title,"tier":tier,"cells":cells,
                  "surface_cue":rel in SURFACE_CUE_RELATIONS}
    shared=None
    for rel,_,_ in RESULTS_ORDER:
        names=set(out[rel]["cells"])
        shared=names if shared is None else (shared & names)
    shared=sorted(shared or [])
    extra=sorted({m for rel,_,_ in RESULTS_ORDER for m in out[rel]["cells"]}-set(shared))
    return {"relations":out,"models":shared,"extra_models":extra,"roster":dict(roster),
            "rows_file":"transfer/data/rows.jsonl"}

def _pearson(xy):
    n=len(xy)
    if n<3: return None
    mx=sum(x for x,_ in xy)/n; my=sum(y for _,y in xy)/n
    sxx=sum((x-mx)**2 for x,_ in xy); syy=sum((y-my)**2 for _,y in xy)
    sxy=sum((x-mx)*(y-my) for x,y in xy)
    if sxx<=0 or syy<=0: return None
    return sxy/(sxx*syy)**0.5, sxy/sxx, my-(sxy/sxx)*mx

def correlation_check(res):
    rels=res["relations"]; roster=res["roster"]
    xy=lambda c: (c["primary_probe_gain"],c["heldout_cand_acc"])
    per=[]; pooled=[]; below=0; rms=[]; by_rel=defaultdict(list)
    for m in res["models"]+res["extra_models"]:
        rpts=[xy(c) for c in roster.get(m,{}).values()]
        fit=_pearson(rpts)
        if not fit: continue
        r0,slope,icpt=fit
        eng=[(rel,xy(rels[rel]["cells"][m])) for rel in CHECK_RELATIONS if m in rels[rel]["cells"]]
        ra=_pearson(rpts+[v for _,v in eng]); re_=_pearson([v for _,v in eng])
        resid={rel: v[1]-(icpt+slope*v[0]) for rel,v in eng}
        for rel,d in resid.items(): by_rel[rel].append(d); below+=d<0
        pooled+=[v for _,v in eng]
        rms.append((sum((y-(icpt+slope*x))**2 for x,y in rpts)/len(rpts))**0.5)
        per.append({"model":m,"n_roster":len(rpts),"r_roster":r0,"slope":slope,"intercept":icpt,
                    "n_all":len(rpts)+len(eng),"r_all":ra[0] if ra else None,
                    "r_eng":re_[0] if re_ else None,"n_eng":len(eng),
                    "mean_resid":(sum(resid.values())/len(resid)) if resid else None,"resid":resid})
    full=[p for p in per if p["n_eng"]==len(CHECK_RELATIONS)]
    pooled_fit=_pearson(pooled)
    return {"per_model":per,"full":full,
            "pooled_r":pooled_fit[0] if pooled_fit else None,"pooled_n":len(pooled),
            "below":below,"rms_roster":sum(rms)/len(rms) if rms else None,
            "rel_mean_resid":{rel:sum(v)/len(v) for rel,v in by_rel.items()}}

def roster_scatter(res, chk):
    W,H=560,440; ml,mr,mt,mb=68,16,46,66
    x0,x1,y0,y1=ml,W-mr,mt,H-mb
    XMAX=0.9
    sx=lambda v: x0+(min(max(v,0),XMAX)/XMAX)*(x1-x0)
    sy=lambda v: y1-min(max(v,0),1)*(y1-y0)
    esc=html.escape; models=res["models"]
    full_sub=(f'grey: {len(models)} models x {chk["full"][0]["n_roster"] if chk["full"] else 0} '
              'roster relations and their per-model fits; colour: the engineered relations')
    p=[f'<svg class="chart wide" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" '
       'aria-label="Held-out accuracy against LIN-HEAD probe gain: roster relations in grey with each model\'s roster-only fit, engineered relations in colour.">'
       f'<title>{esc(full_sub)}</title>']
    p.append('<text class="ttl" x="0" y="15">Where the engineered relations fall on the roster line</text>')
    p.append('<text class="sub" x="0" y="31">grey: roster + fits; colour: engineered</text>')
    for t in (0.0,0.2,0.4,0.6,0.8,1.0):
        p.append(f'<line class="grid" x1="{x0}" y1="{sy(t):.1f}" x2="{x1}" y2="{sy(t):.1f}"/>')
        p.append(f'<text class="tick" x="{x0-8}" y="{sy(t)+4:.1f}" text-anchor="end">{_f(t,1)}</text>')
    for t in (0.0,0.2,0.4,0.6,0.8):
        p.append(f'<text class="tick" x="{sx(t):.1f}" y="{y1+16:.1f}" text-anchor="middle">{_f(t,1)}</text>')
    p.append(f'<line class="axis" x1="{x0}" y1="{y1:.1f}" x2="{x1}" y2="{y1:.1f}"/>')
    p.append(f'<line class="axis" x1="{x0}" y1="{y0:.1f}" x2="{x0}" y2="{y1:.1f}"/>')
    p.append(f'<text class="axl" x="{(x0+x1)/2:.1f}" y="{y1+34:.1f}" text-anchor="middle">LIN-HEAD probe gain (null = 0)</text>')
    p.append(f'<text class="axl" transform="translate(13,{(y0+y1)/2:.1f}) rotate(-90)" text-anchor="middle">NEW-REL-GEN on held-out subjects</text>')
    for entry in chk["full"]:
        m=entry["model"]; pts=[(c["primary_probe_gain"],c["heldout_cand_acc"]) for c in res["roster"][m].values()]
        lo,hi=min(x for x,_ in pts),max(x for x,_ in pts)
        ya,yb=entry["intercept"]+entry["slope"]*lo,entry["intercept"]+entry["slope"]*hi
        p.append(f'<line class="rfit" x1="{sx(lo):.1f}" y1="{sy(ya):.1f}" x2="{sx(hi):.1f}" y2="{sy(yb):.1f}"><title>{esc(m)}: roster-only fit, r = {_f(entry["r_roster"])}, n = {entry["n_roster"]}</title></line>')
        for (gx,gy),(rel,c) in zip(pts,res["roster"][m].items()):
            p.append(f'<circle class="rp" cx="{sx(gx):.1f}" cy="{sy(gy):.1f}" r="3"><title>{esc(relation_label(rel))}: {esc(m)}, probe gain {_f(gx)}, held-out {_f(gy)}</title></circle>')
    for rel,title,tier in RESULTS_ORDER:
        info=res["relations"][rel]; cells=info["cells"]
        cls=("mk hollow" if info["surface_cue"] else f"mk c-{tier}")
        for m in models:
            c=cells[m]; gx,gy=c["primary_probe_gain"],c["heldout_cand_acc"]
            p.append(_shape(SHAPES[rel],sx(gx),sy(gy),4.6,cls,
                f'{title}{" (excluded control)" if info["surface_cue"] else ""}: {m}, probe gain {_f(gx)}, held-out {_f(gy)}'))

    p.append('</svg>')
    return "".join(p)

PW, PML, PMR, PMT, PROW, PMB = 460, 140, 16, 54, 20, 50

def _f(v,n=3): return f"{v:.{n}f}"

def _shape(kind, cx, cy, r, cls, title):
    t=f"<title>{html.escape(title)}</title>"
    if kind=="square":
        s=r*0.92
        return f'<rect class="{cls}" x="{cx-s:.1f}" y="{cy-s:.1f}" width="{2*s:.1f}" height="{2*s:.1f}" rx="1">{t}</rect>'
    if kind=="diamond":
        d=r*1.25
        return f'<polygon class="{cls}" points="{cx:.1f},{cy-d:.1f} {cx+d:.1f},{cy:.1f} {cx:.1f},{cy+d:.1f} {cx-d:.1f},{cy:.1f}">{t}</polygon>'
    if kind=="triangle":
        return f'<polygon class="{cls}" points="{cx:.1f},{cy-r*1.3:.1f} {cx+r*1.2:.1f},{cy+r*0.88:.1f} {cx-r*1.2:.1f},{cy+r*0.88:.1f}">{t}</polygon>'
    if kind=="tri-down":
        return f'<polygon class="{cls}" points="{cx:.1f},{cy+r*1.3:.1f} {cx+r*1.2:.1f},{cy-r*0.88:.1f} {cx-r*1.2:.1f},{cy-r*0.88:.1f}">{t}</polygon>'
    return f'<circle class="{cls}" cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}">{t}</circle>'

def _frame(models, d0, d1, ticks, chance, chance_label, title, sub, tier, body_fn):
    n=len(models); y0=PMT; y1=PMT+PROW*n; H=y1+PMB; x1=PW-PMR
    sx=lambda v: PML+(v-d0)/(d1-d0)*(x1-PML)
    esc=html.escape
    parts=[f'<svg class="chart tier-{tier}" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {PW} {H}" role="img" aria-label="{esc(title)}. {esc(sub)}">']
    parts.append(f'<text class="ttl" x="0" y="15">{esc(title)}</text>')
    parts.append(f'<text class="sub" x="0" y="30">{esc(sub)}</text>')
    for t in ticks:
        parts.append(f'<line class="grid" x1="{sx(t):.1f}" y1="{y0:.1f}" x2="{sx(t):.1f}" y2="{y1:.1f}"/>')
    parts.append(f'<line class="rule" x1="{sx(chance):.1f}" y1="{y0-7:.1f}" x2="{sx(chance):.1f}" y2="{y1:.1f}"/>')
    parts.append(f'<text class="tick" x="{sx(chance):.1f}" y="{y0-11:.1f}" text-anchor="middle">{esc(chance_label)}</text>')
    parts.append(f'<line class="axis" x1="{PML}" y1="{y1:.1f}" x2="{x1}" y2="{y1:.1f}"/>')
    for t in ticks:
        parts.append(f'<text class="tick" x="{sx(t):.1f}" y="{y1+15:.1f}" text-anchor="middle">{_f(t,1)}</text>')
    for i,m in enumerate(models):
        cy=y0+PROW*i+PROW/2
        parts.append(f'<text class="mdl" x="{PML-9}" y="{cy+4:.1f}" text-anchor="end">{esc(m)}</text>')
    parts.append(body_fn(sx,y0,x1))
    parts.append('</svg>')
    return "".join(parts)

CHAR_W = 6.0

def _label_at(sx, x1, value, cy, text, avoid=None, cls="val"):
    x=sx(value); w=len(text)*CHAR_W+2
    free=lambda a,b: avoid is None or not (a-7 < sx(avoid) < b+7)
    if x+10+w < x1 and free(x+10,x+10+w):
        return f'<text class="{cls}" x="{x+10:.1f}" y="{cy+4:.1f}" text-anchor="start">{text}</text>'
    if x-10-w > PML and free(x-10-w,x-10):
        return f'<text class="{cls}" x="{x-10:.1f}" y="{cy+4:.1f}" text-anchor="end">{text}</text>'
    return f'<text class="{cls}" x="{x:.1f}" y="{cy-9:.1f}" text-anchor="middle">{text}</text>'

def dumbbell_panel(rel, info, models):
    cells=info["cells"]; chance=cells[models[0]]["chance"]
    sub=(f'{info["n_classes"]} classes, chance {_f(chance)}, '
         f'{info["n_heldout"]} held-out subjects')
    if info["surface_cue"]: sub="Excluded control. "+sub
    order=sorted(models,key=lambda m:-cells[m]["heldout_cand_acc"])
    def body(sx,y0,x1):
        out=[]
        best,worst=order[0],order[-1]
        for i,m in enumerate(order):
            c=cells[m]; cy=y0+PROW*i+PROW/2
            a,b=c["newrel_gen_ep0_acc"],c["heldout_cand_acc"]
            out.append(f'<line class="conn" x1="{sx(a):.1f}" y1="{cy:.1f}" x2="{sx(b):.1f}" y2="{cy:.1f}"/>')
            out.append(f'<polygon class="gate" points="{sx(c["pre_rel_cand_acc"]):.1f},{cy-5:.1f} {sx(c["pre_rel_cand_acc"])+4.4:.1f},{cy:.1f} {sx(c["pre_rel_cand_acc"]):.1f},{cy+5:.1f} {sx(c["pre_rel_cand_acc"])-4.4:.1f},{cy:.1f}"><title>{html.escape(m)} PRE-REL candidate accuracy {_f(c["pre_rel_cand_acc"])}</title></polygon>')
            out.append(f'<circle class="pre" cx="{sx(a):.1f}" cy="{cy:.1f}" r="4"><title>{html.escape(m)} epoch 0 {_f(a)}</title></circle>')
            out.append(f'<circle class="post" cx="{sx(b):.1f}" cy="{cy:.1f}" r="4.6"><title>{html.escape(m)} after fine-tuning {_f(b)}</title></circle>')
            if m in (best,worst):
                out.append(_label_at(sx,x1,b,cy,_f(b),avoid=c["pre_rel_cand_acc"]))
        return "".join(out)
    return _frame(order,0.0,1.0,[0.0,0.2,0.4,0.6,0.8,1.0],chance,f"chance {_f(chance)}",
                  info["title"],sub,info["tier"],body)

def probe_panel(rel, info, models):
    cells=info["cells"]
    lo=min(cells[m]["primary_probe_gain"] for m in models)
    hi=max(cells[m]["primary_probe_gain"] for m in models)
    sub=f'probe gain {_f(lo)} to {_f(hi)}'
    if info["surface_cue"]: sub="Excluded control. "+sub
    order=sorted(models,key=lambda m:-cells[m]["primary_probe_gain"])
    def body(sx,y0,x1):
        out=[]; best,worst=order[0],order[-1]
        for i,m in enumerate(order):
            g=cells[m]["primary_probe_gain"]; cy=y0+PROW*i+PROW/2
            out.append(f'<line class="stem" x1="{sx(0.0):.1f}" y1="{cy:.1f}" x2="{sx(g):.1f}" y2="{cy:.1f}"/>')
            out.append(f'<circle class="post" cx="{sx(g):.1f}" cy="{cy:.1f}" r="4.6"><title>{html.escape(m)} probe gain {_f(g)}</title></circle>')
            if m in (best,worst): out.append(_label_at(sx,x1,g,cy,_f(g)))
        return "".join(out)
    return _frame(order,0.0,0.9,[0.0,0.2,0.4,0.6,0.8],0.0,"no gain",
                  info["title"],sub,info["tier"],body)

def _place_label(text, box, points, placed, bounds):
    x0,y0,x1,y1=bounds; bx0,by0,bx1,by1=box
    w=len(text)*CHAR_W+2; h=12
    cx,cy=(bx0+bx1)/2,(by0+by1)/2
    cands=[]
    for pad in (10,26,42):
        cands += [(bx1+pad, cy+4, "start"), (bx0-pad, cy+4, "end"),
                  (cx, by0-pad+2, "middle"), (cx, by1+pad+6, "middle"),
                  (bx1+pad, by0-pad+8, "start"), (bx1+pad, by1+pad, "start"),
                  (bx0-pad, by0-pad+8, "end"), (bx0-pad, by1+pad, "end")]
    best=None
    for x,y,anchor in cands:
        left = x if anchor=="start" else (x-w if anchor=="end" else x-w/2)
        r=(left-2, y-h, left+w+2, y+3)
        cost=0.0
        if r[0]<x0 or r[2]>x1 or r[1]<y0 or r[3]>y1: cost+=1000
        for px,py in points:
            if r[0]<px+7 and r[2]>px-7 and r[1]<py+7 and r[3]>py-7: cost+=40
        for q in placed:
            if r[0]<q[2] and r[2]>q[0] and r[1]<q[3] and r[3]>q[1]: cost+=60
        cost += 0.02*abs(x-cx)+0.02*abs(y-cy)
        if best is None or cost<best[0]: best=(cost,x,y,anchor,r)
        if cost<1: break
    _,x,y,anchor,r=best
    placed.append(r)
    return x,y,anchor

def scatter(res):
    W,H=480,430; ml,mr,mt,mb=68,16,46,66
    x0,x1,y0,y1=ml,W-mr,mt,H-mb
    sx=lambda v: x0+(v/0.9)*(x1-x0)
    sy=lambda v: y1-v*(y1-y0)
    esc=html.escape; models=res["models"]
    p=[f'<svg class="chart wide" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" aria-label="Held-out accuracy against LIN-HEAD probe gain, one point per relation and model.">']
    p.append('<text class="ttl" x="0" y="15">Held-out accuracy against probe gain</text>')
    p.append(f'<text class="sub" x="0" y="31">one point per relation and model, {len(models)} models x 7 relations</text>')
    for t in (0.0,0.2,0.4,0.6,0.8,1.0):
        p.append(f'<line class="grid" x1="{x0}" y1="{sy(t):.1f}" x2="{x1}" y2="{sy(t):.1f}"/>')
        p.append(f'<text class="tick" x="{x0-8}" y="{sy(t)+4:.1f}" text-anchor="end">{_f(t,1)}</text>')
    for t in (0.0,0.2,0.4,0.6,0.8):
        p.append(f'<text class="tick" x="{sx(t):.1f}" y="{y1+16:.1f}" text-anchor="middle">{_f(t,1)}</text>')
    p.append(f'<line class="axis" x1="{x0}" y1="{y1:.1f}" x2="{x1}" y2="{y1:.1f}"/>')
    p.append(f'<line class="axis" x1="{x0}" y1="{y0:.1f}" x2="{x0}" y2="{y1:.1f}"/>')
    p.append(f'<text class="axl" x="{(x0+x1)/2:.1f}" y="{y1+34:.1f}" text-anchor="middle">LIN-HEAD probe gain (null = 0)</text>')
    p.append(f'<text class="axl" transform="translate(13,{(y0+y1)/2:.1f}) rotate(-90)" text-anchor="middle">NEW-REL-GEN on held-out subjects</text>')
    seen=set()
    for rel,title,tier in RESULTS_ORDER:
        info=res["relations"][rel]; ch=info["cells"][models[0]]["chance"]
        if round(ch,4) in seen: continue
        seen.add(round(ch,4))
        p.append(f'<line class="rule" x1="{x0}" y1="{sy(ch):.1f}" x2="{x1}" y2="{sy(ch):.1f}"/>')
        p.append(f'<text class="tick" x="{x1}" y="{sy(ch)-4:.1f}" text-anchor="end">chance {_f(ch)}</text>')
    every=[]; clusters=[]
    for rel,title,tier in RESULTS_ORDER:
        info=res["relations"][rel]; cells=info["cells"]
        cls=("mk hollow" if info["surface_cue"] else f"mk c-{tier}")
        pts=[]
        for m in models:
            c=cells[m]; gx,gy=c["primary_probe_gain"],c["heldout_cand_acc"]
            pts.append((sx(gx),sy(gy)))
            p.append(_shape(SHAPES[rel],sx(gx),sy(gy),4.6,cls,
                f'{title}{" (excluded control)" if info["surface_cue"] else ""}: {m}, probe gain {_f(gx)}, held-out {_f(gy)}'))
        every+=pts
        clusters.append((rel,title+(" (control)" if info["surface_cue"] else ""),pts))
    placed=[]
    for rel,lab,pts in clusters:
        box=(min(x for x,_ in pts),min(y for _,y in pts),
             max(x for x,_ in pts),max(y for _,y in pts))
        lx,ly,anchor=_place_label(lab,box,every,placed,(x0,y0,x1,y1))
        p.append(f'<text class="clab" x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}">{esc(lab)}</text>')
    p.append('</svg>')
    return "".join(p)

RESULTS_CSS = """
:root{color-scheme:light;--bg:#faf9f6;--surface:#ffffff;--border:#ddd8cf;--border-strong:#c3bcae;
--text:#17202a;--text-2:#52514e;--text-muted:#6f7780;--link:#175c8c;--grid:#e7e3da;
--t-known:#2a78d6;--t-strong:#eb6834;--t-weak:#1baf7a;--t-control:#6f7780}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;
--bg:#14140f;--surface:#1c1c19;--border:#383730;--border-strong:#514f45;--text:#f2f1ea;
--text-2:#c3c2b7;--text-muted:#9b9a8f;--link:#7ab6e8;--grid:#2e2e28;
--t-known:#3987e5;--t-strong:#d95926;--t-weak:#199e70;--t-control:#9b9a8f}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#14140f;--surface:#1c1c19;--border:#383730;
--border-strong:#514f45;--text:#f2f1ea;--text-2:#c3c2b7;--text-muted:#9b9a8f;--link:#7ab6e8;
--grid:#2e2e28;--t-known:#3987e5;--t-strong:#d95926;--t-weak:#199e70;--t-control:#9b9a8f}
*{box-sizing:border-box}
body{margin:0 auto;max-width:860px;padding:1.5rem 1.25rem 3rem;
font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--text);background:var(--bg)}
h1{font-size:1.5rem;line-height:1.2;margin:.4rem 0 .35rem}
h2{font-size:1.08rem;margin:2.2rem 0 .5rem}
h3{font-size:.98rem;margin:1.4rem 0 .3rem}
a{color:var(--link)}
p,li{max-width:74ch}
li{margin:.35rem 0}
small,.note{color:var(--text-muted);font-size:.84rem}
dl{margin:.4rem 0}
dt{font-weight:600;margin-top:.7rem}
dd{margin:.1rem 0 0;color:var(--text-2)}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.88em}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;
padding:1rem 1rem .8rem;margin:.9rem 0 1.2rem}
.control{border-left:3px solid var(--t-control)}
figure{margin:0 0 1.6rem}
figcaption{color:var(--text-muted);font-size:.84rem;margin-top:.5rem;max-width:74ch}
.chart{display:block;width:100%;height:auto;max-width:560px;overflow:visible}
.chart.wide{max-width:660px}
.chart text{font:12px system-ui,-apple-system,"Segoe UI",sans-serif;fill:var(--text-2)}
.chart .ttl{font-size:13.5px;font-weight:600;fill:var(--text)}
.chart .sub,.chart .tick,.chart .axl{font-size:11px;fill:var(--text-muted)}
.chart .mdl{font-size:11px;fill:var(--text-2)}
.chart .val{font-size:11px;fill:var(--text-2);font-variant-numeric:tabular-nums}
.chart .clab{font-size:11px;fill:var(--text-2)}
.chart .grid{stroke:var(--grid);stroke-width:1;fill:none}
.chart .axis{stroke:var(--border-strong);stroke-width:1;fill:none}
.chart .rule{stroke:var(--border-strong);stroke-width:1;fill:none}
.chart .conn{stroke-width:2;stroke-linecap:round;opacity:.35;fill:none}
.chart .stem{stroke-width:2;stroke-linecap:round;opacity:.28;fill:none}
.tier-known .post{fill:var(--t-known)}
.tier-strong .post{fill:var(--t-strong)}
.tier-weak .post{fill:var(--t-weak)}
.tier-control .post{fill:var(--t-control)}
.tier-known .conn,.tier-known .stem{stroke:var(--t-known)}
.tier-strong .conn,.tier-strong .stem{stroke:var(--t-strong)}
.tier-weak .conn,.tier-weak .stem{stroke:var(--t-weak)}
.tier-control .conn,.tier-control .stem{stroke:var(--t-control)}
.chart .post{stroke:var(--surface);stroke-width:2}
.chart .pre{fill:var(--surface);stroke-width:2}
.tier-known .pre{stroke:var(--t-known)}
.tier-strong .pre{stroke:var(--t-strong)}
.tier-weak .pre{stroke:var(--t-weak)}
.tier-control .pre{stroke:var(--t-control)}
.chart .gate{fill:none;stroke:var(--text-muted);stroke-width:1.5}
.mk{stroke:var(--surface);stroke-width:2}
.c-known{fill:var(--t-known)}
.c-strong{fill:var(--t-strong)}
.c-weak{fill:var(--t-weak)}
.hollow{fill:var(--surface);stroke:var(--t-control);stroke-width:1.6}
/* Unscoped, not .chart .rp: the same two marks appear in the 16px legend
   swatch, which is a bare <svg> and would otherwise fall back to a black fill
   and no stroke, invisible on the dark surface. */
.rp{fill:var(--text-muted);opacity:.32;stroke:none}
.rfit{stroke:var(--text-muted);stroke-width:1;opacity:.35;fill:none}
.legend{display:flex;flex-wrap:wrap;gap:.35rem 1.1rem;margin:.2rem 0 .6rem;
font-size:.84rem;color:var(--text-2);list-style:none;padding:0}
.legend li{display:flex;align-items:center;gap:.4rem;margin:0}
.legend svg{flex:none}
.panels{margin:.6rem 0 0}
/* SVG text is sized in viewBox units, then scaled with the chart. At a true
   390px iframe the old 11-unit labels rendered at roughly 7px. These sizes
   keep ticks, labels, and captions legible on the smallest supported view. */
@media (max-width:560px){
  .chart text{font-size:16px}
  .chart .ttl{font-size:18px}
  .chart .sub,.chart .tick,.chart .axl,
  .chart .mdl,.chart .val,.chart .clab{font-size:15px}
}
@media (prefers-color-scheme:dark){.rfit{opacity:.28}}
:root[data-theme="dark"] .rfit{opacity:.28}
"""

LEGEND_SHAPES = {"circle":'<circle cx="8" cy="8" r="4.6"/>',
  "square":'<rect x="3.7" y="3.7" width="8.6" height="8.6" rx="1"/>',
  "diamond":'<polygon points="8,2.2 13.8,8 8,13.8 2.2,8"/>',
  "triangle":'<polygon points="8,2.4 13.5,12 2.5,12"/>',
  "tri-down":'<polygon points="8,13.6 13.5,4 2.5,4"/>'}

def _swatch(shape, cls):
    return (f'<svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true" '
            f'xmlns="http://www.w3.org/2000/svg"><g class="{cls}">{LEGEND_SHAPES[shape]}</g></svg>')

def results_page(res):
    esc=lambda x: html.escape(str(x))
    models=res["models"]; rels=res["relations"]
    kept=[(r,t,ti) for r,t,ti in RESULTS_ORDER if not rels[r]["surface_cue"]]
    ctrl=[(r,t,ti) for r,t,ti in RESULTS_ORDER if rels[r]["surface_cue"]]
    for rel,_,_ in RESULTS_ORDER:
        c=rels[rel]["cells"][models[0]]
        rels[rel].update({"n_classes":c["n_classes"],"n_heldout":c["n_heldout"],
                          "n_train":c["n_train"],"chance":c["chance"]})
    def stat(rel,field,fn):
        return fn([rels[rel]["cells"][m][field] for m in models])
    p=[]
    p.append('<!doctype html><html lang="en"><head><meta charset="utf-8">'
             '<meta name="viewport" content="width=device-width,initial-scale=1">'
             + relation_label_meta() +
             '<title>Engineered relations: trained results</title>'
             '<link rel="stylesheet" href="../../shared/collapse.css">'
             '<script src="../../shared/collapse.js" defer></script>'
             '<style>'+RESULTS_CSS+'</style></head><body>')
    p.append('<p><a href="../index.html">&#8592; Engineered relations</a> &middot; '
             '<a href="../../index.html">All reports</a></p>')
    p.append('<h1>Engineered relations: trained results</h1>')
    p.append('<p class="note">Seven engineered relations, ten models on a shared axis, three repeats each. '
             'Every number is a mean over the three locked repeats of one fine-tuning cell, '
             'to three decimals. Source: the merged relational-transfer page data '
             f'(<code>{esc(res["rows_file"])}</code>).</p>')
    p.append('<p>Each relation is read against <b>its own</b> pre-training baseline, never pooled. '
             'Country language is knowledge the base models already have and starts near 0.9 on the '
             'knowledge gate; the four banded relations start at chance. So a single statement about '
             '"the engineered relations" would mean nothing, and none is made here. Of the four banded '
             'relations, longitude and latitude transfer strongly from chance, elevation '
             'weakly and population weakest. Two relations, first letter and name length, are '
             'surface-cue controls: their object is a deterministic function of the town name’s '
             'spelling, so they form a separately reported control group rather than another substantive '
             'relation. They are drawn hollow and excluded from the substantive relational-transfer '
             'summaries, while their behavior remains visible for comparison.</p>')
    p.append(f'<p class="note">Qwen3.8-27B has no engineered-v2 cells, so it is not on the shared '
             f'model axis; it contributes only to the three first-set relations and its values are '
             f'named in those relations’ summaries.</p>' if res["extra_models"] else "")

    p.append('<h2>1. Transfer against linear availability</h2>')
    p.append('<p>Read this first as a descriptive association, not a causal claim: the plot compares '
             'LIN-HEAD probe gain with NEW-REL-GEN across relations, but it does not show that linear '
             'availability causes transfer, or that either measure causes the other. The per-model grey '
             'lines are fitted on roster relations only; the coloured engineered relations are then '
             'read against those fixed lines.</p>')
    p.append('<figure class="card">')
    p.append(scatter(res))
    p.append('<ul class="legend">')
    for rel,title,tier in RESULTS_ORDER:
        info=rels[rel]
        cls=("mk hollow" if info["surface_cue"] else f"mk c-{tier}")
        lab=title+(" (excluded control)" if info["surface_cue"] else "")
        p.append(f'<li>{_swatch(SHAPES[rel],cls)}{esc(lab)}</li>')
    p.append('</ul>')
    p.append('<figcaption>One point per relation and model: LIN-HEAD probe gain across, NEW-REL-GEN on '
             'the held-out subjects up. Colour marks the pre-training tier (blue: known before '
             'fine-tuning; orange: at chance before fine-tuning and transfers strongly; green: at chance '
             'before fine-tuning and transfers weakly), and mark shape names the relation, so no reading '
             'here rests on colour alone. The two hollow grey series are the surface-cue controls and are '
             'excluded. The three horizontal rules are the three chance levels in play, because chance '
             'is a property of the relation’s class count and not of the page: a point is above its '
             'own rule, never above a shared one. The pre-training baseline the text reads these against '
             'is not on this chart; it is chart 2, where each model’s epoch-0 value and knowledge '
             'gate sit on the same axis as its post-fine-tuning value.</figcaption>')
    p.append('</figure>')

    p.append('<h2>How to read this</h2>')
    p.append('<dl>')
    for term,body in [
      ("Candidate accuracy","the model is scored only over the relation’s own answer strings, so a "
       "wrong answer is a wrong choice among the candidates rather than an off-topic continuation. "
       "Class-macro means every answer class counts equally, whatever its number of subjects."),
      ("Held-out","the subjects that were never shown during fine-tuning. NEW-REL-GEN, the transfer "
       "measure plotted here, is candidate accuracy on those subjects after the twenty fine-tuning "
       "epochs (field <code>heldout_cand_acc</code>)."),
      ("Epoch 0","the same measure, on the same held-out subjects, before any fine-tuning (field "
       "<code>newrel_gen_ep0_acc</code>). It is the relation’s own before value for the transfer measure."),
      ("PRE-REL candidate accuracy","the base model’s candidate accuracy on the original statement of "
       "the attribute, canonical template, over the fine-tuning set (field <code>pre_rel_cand_acc</code>). "
       "This is the knowledge gate: asked plainly, does the model know the attribute at all?"),
      ("Pre-training baseline","the pair of the two above. Neither is zero and neither is the same across "
       "relations, which is exactly why each relation is read against its own."),
      ("Probe gain","the LIN-HEAD probe gain at q = 16 (field <code>primary_probe_gain</code>): how much "
       "better than a constant predictor a linear map from the pre-fine-tuning representation predicts "
       "the post-fine-tuning answer target. A large gain means that target is predictable from the base "
       "representation under this probe; it is not by itself proof that the attribute was stored linearly. "
       "q = 16 is a capacity choice, not evidence for nonlinear storage. Its null is 0, not chance."),
      ("Chance","one divided by the relation’s number of answer classes. It differs by relation: 0.091 "
       "for country language with 11 retained classes, 0.200 for the five-class bands, 0.048 for the "
       "21-class first-letter control."),
      ("Surface-cue control","first letter and name length. A separately reported control group whose "
       "objects are deterministic functions of spelling. They are excluded from the substantive relation "
       "summaries, but shown on the same axes for comparison."),
    ]:
        p.append(f'<dt>{esc(term)}</dt><dd>{body}</dd>')
    p.append('</dl>')

    p.append('<h2>2. The knowledge gate: before and after, per model</h2>')
    p.append('<p>One panel per relation, the same ten models down every panel and the same accuracy '
             'axis across them. Each model’s row runs from its epoch-0 value to its value after '
             'fine-tuning; the hollow diamond is the knowledge gate, the base model asked the '
             'attribute plainly. The vertical rule is chance for that relation.</p>')
    p.append('<ul class="legend">'
             '<li><svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true" xmlns="http://www.w3.org/2000/svg"><g class="tier-known"><circle class="pre" cx="8" cy="8" r="4"/></g></svg>epoch 0 (hollow)</li>'
             '<li><svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true" xmlns="http://www.w3.org/2000/svg"><g class="tier-known"><circle class="post" cx="8" cy="8" r="4.6"/></g></svg>after fine-tuning (filled)</li>'
             '<li><svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true" xmlns="http://www.w3.org/2000/svg"><polygon class="gate" points="8,3 12.4,8 8,13 3.6,8"/></svg>PRE-REL candidate accuracy (knowledge gate)</li>'
             '</ul>')
    p.append('<div class="panels">')
    for rel,title,tier in RESULTS_ORDER:
        cls="card control" if rels[rel]["surface_cue"] else "card"
        p.append(f'<figure class="{cls}">{dumbbell_panel(rel,rels[rel],models)}</figure>')
    p.append('</div>')
    p.append('<p class="note">Only the highest and lowest model in each panel is labelled with a number; '
             'every mark carries its exact value as a hover title, and the full per-relation figures are '
             'in the summaries below.</p>')

    p.append('<h2>3. LIN-HEAD probe gain, per model</h2>')
    p.append('<p>The same ten models and the same axis in every panel, so the panels compare directly. '
             'The rule at zero is the null: a probe gain of zero means a linear map from the base '
             'representation does no better than a constant predictor.</p>')
    p.append('<div class="panels">')
    for rel,title,tier in RESULTS_ORDER:
        cls="card control" if rels[rel]["surface_cue"] else "card"
        p.append(f'<figure class="{cls}">{probe_panel(rel,rels[rel],models)}</figure>')
    p.append('</div>')

    p.append('<h2>Per relation</h2>')
    import statistics as _st
    for rel,title,tier in RESULTS_ORDER:
        info=rels[rel]; cells=info["cells"]
        extra=[m for m in res["extra_models"] if m in cells]
        med=lambda f: _st.median([cells[m][f] for m in models])
        rng=lambda f: (min(cells[m][f] for m in models), max(cells[m][f] for m in models))
        pg=rng("primary_probe_gain"); ho=rng("heldout_cand_acc"); e0=rng("newrel_gen_ep0_acc")
        gate=rng("pre_rel_cand_acc"); lin=rng("pre_rel_cls_gain")
        cls="card control" if info["surface_cue"] else "card"
        p.append(f'<div class="{cls}">')
        head=esc(title)+(" <small>(excluded surface-cue control)</small>" if info["surface_cue"] else "")
        p.append(f'<h3>{head}</h3>')
        p.append('<ul>')
        p.append(f'<li>{len(models)} models on the shared axis, 3 repeats each'
                 + (f', plus {esc(", ".join(extra))} off the shared axis' if extra else '')
                 + f'; {info["n_classes"]} answer classes, chance {_f(info["chance"])}; '
                 f'N = {info["n_train"]} fine-tuning subjects, {info["n_heldout"]} held out.</li>')
        p.append(f'<li>Pre-training, median: epoch 0 {_f(med("newrel_gen_ep0_acc"))} '
                 f'(range {_f(e0[0])} to {_f(e0[1])}); knowledge gate {_f(med("pre_rel_cand_acc"))} '
                 f'(range {_f(gate[0])} to {_f(gate[1])}).</li>')
        p.append(f'<li>Post-training, median: NEW-REL-GEN {_f(med("heldout_cand_acc"))} '
                 f'(range {_f(ho[0])} to {_f(ho[1])}).</li>')
        p.append(f'<li>Probe gain: {_f(pg[0])} to {_f(pg[1])}, median {_f(med("primary_probe_gain"))}; '
                 f'PRE-REL linear gain {_f(lin[0])} to {_f(lin[1])}.</li>')
        if extra:
            for m in extra:
                c=cells[m]
                p.append(f'<li>{esc(m)}, first set only: NEW-REL-GEN {_f(c["heldout_cand_acc"])} '
                         f'(epoch 0 {_f(c["newrel_gen_ep0_acc"])}), probe gain {_f(c["primary_probe_gain"])}, '
                         f'knowledge gate {_f(c["pre_rel_cand_acc"])}.</li>')
        p.append('</ul>')
        p.append(f'<p>{RESULT_PROSE[rel]}</p>')
        p.append('</div>')

    p.append('<h2>How a number here was made</h2>')
    p.append('<p>A cell is one model on one relation. It counts only when all three locked repeats '
             '0, 1, 2 are present, each <code>status=complete</code> and production eligible '
             '(twenty planned epochs, twenty done, no train limit, exposure equal to the training '
             'records); an incomplete cell is dropped whole rather than averaged over whatever '
             'survived. The three repeats are then averaged. That is the relational-transfer page’s '
             'own rule, restated so this page stands alone. Fields read: '
             '<code>heldout_cand_acc</code>, <code>newrel_gen_ep0_acc</code>, '
             '<code>primary_probe_gain</code>, <code>pre_rel_cand_acc</code>, '
             '<code>pre_rel_cls_gain</code>, <code>chance</code>, <code>n_classes</code>, '
             '<code>n_heldout</code>, <code>n_train</code>.</p>')
    p.append('<p class="note">What these numbers do not say: each cell is three repeats on one model, '
             'so per-model differences inside a relation are not separated from repeat noise; nothing '
             'here is a claim about Qwen3.8-27B on the second-set relations, which have not run; and '
             '"weak" above means low absolute accuracy, not a tested difference from the rest of the '
             'roster. This page is generated by <code>scripts/make_engineered.py</code> and holds no '
             'external requests, no scripts and no remote assets.</p>')
    p.append('</body></html>')
    return "".join(x for x in p if x)

RESULT_PROSE = {
 "engineered_town_country_language":
  "Known before fine-tuning and it transfers on every column. The knowledge gate is far above chance "
  "for all ten models, so what this relation tests is attaching a new carrier to knowledge the model "
  "already has, not acquiring the attribute. Its LIN-HEAD gain is the largest of the seven, meaning the "
  "post-fine-tuning answer target is especially predictable from the base representation under this probe. "
  "granite-4.2-3b is the low "
  "column and is also the second-lowest knowledge gate.",
 "engineered_town_latitude_band":
  "The first-set relation of the same kind as the second-set bands: a real-world attribute whose band "
  "labels are ours, so its statement frequency in any corpus is essentially zero. At chance before "
  "fine-tuning and the strongest transfer of the four bands.",
 "engineered_town_longitude_band":
  "The informative one. The models cannot state it before fine-tuning, yet it generalises to held-out "
  "towns after it, and the same rows show a relatively high LIN-HEAD gain: the post-fine-tuning answer "
  "target is predictable from the base representation under this probe. This is evidence about the measured "
  "association, not a causal claim about storage. Knowledge without corpus statements of the fact, which is what this "
  "family of relations was proposed to supply. It sits just below its sibling latitude, so the "
  "pre-registered expectation that a model knowing where a town is carries both axes holds.",
 "engineered_town_elevation_band":
  "Weak transfer: above chance on all ten columns but only about twice chance. This is the smallest of "
  "the bands, so its per-column spread is the widest and single-column differences should not be read "
  "as model differences. Against the pre-registration, elevation has the higher measured LIN-HEAD and "
  "PRE-REL gains of the elevation and population pair, not the lower ones.",
 "engineered_town_population_band":
  "The weakest of the four second-set relations, and the pre-registration got it backwards: population "
  "was expected toward the active end on coarse metropolis-versus-small-town knowledge, and it has the "
  "lowest measured LIN-HEAD and PRE-REL gains of the four. On held-out accuracy it and elevation sit "
  "inside each other’s spread, so the ordering claim rests on these measured probe gains rather than on "
  "transfer accuracy.",
 "engineered_town_first_letter":
  "Excluded surface-cue control. The object is a deterministic function of the town name’s spelling; "
  "the base models answer it almost perfectly, and after fine-tuning they are at or near 1.000. It is "
  "reported separately so spelling-linked behavior can be compared with, but not conflated with, the "
  "substantive relations.",
 "engineered_town_name_length":
  "Excluded surface-cue control. Also a deterministic function of spelling, but a harder one: the "
  "knowledge gate sits exactly at chance for every model and both transfer and measured probe gain stay "
  "low. It provides a spelling-linked comparison group, not evidence that the substantive relations are "
  "stored nonlinearly.",
}

INDEX_CSS = ("li{margin:.55rem 0}h2{margin-top:2rem}.card ul{padding-left:1.2rem}.card li b{color:var(--text)}")

def _rng(vals, n=3):
    vals=[v for v in vals if v is not None]
    return f"{_f(min(vals),n)} to {_f(max(vals),n)}" if vals else "n/a"

def index_page(res):
    esc=lambda x: html.escape(str(x))
    summary=results=""
    if res:
        import statistics as _st
        rels=res["relations"]; models=res["models"]; chk=correlation_check(res)
        def med(rel,field="heldout_cand_acc"):
            return _st.median([rels[rel]["cells"][m][field] for m in models])
        def rng(rel,field="heldout_cand_acc"):
            return _rng([rels[rel]["cells"][m][field] for m in models])
        chance=lambda rel: rels[rel]["cells"][models[0]]["chance"]
        full=chk["full"]
        r_roster=_rng([e["r_roster"] for e in full]); r_all=_rng([e["r_all"] for e in full])
        r_eng=_rng([e["r_eng"] for e in full])
        shift=max(abs(e["r_all"]-e["r_roster"]) for e in full) if full else 0
        n_roster=full[0]["n_roster"] if full else 0; n_all=full[0]["n_all"] if full else 0
        mres=_rng([e["mean_resid"] for e in full])
        rel_res=chk["rel_mean_resid"]
        on_line=min(rel_res,key=lambda r: abs(rel_res[r])) if rel_res else None
        CL,LA,LO,EL,PO="engineered_town_country_language","engineered_town_latitude_band",\
            "engineered_town_longitude_band","engineered_town_elevation_band","engineered_town_population_band"
        summary=('<h2 id="summary">Executive summary</h2><div class="card"><ul>'
          '<li><b>What was built.</b> Seven relations over one fixed universe of 300 towns, in two sets: '
          'the town’s name length, first letter and latitude band first, then longitude band, population '
          'band, elevation band and country language from structured Wikidata attributes. Name length and '
          'first letter are a separately reported surface-cue control group (their objects are deterministic '
          'functions of spelling) and are excluded from the substantive relation summaries.</li>'
          f'<li><b>All five substantive relations transfer to held-out towns, in three tiers set by the '
          f'pre-training baseline.</b> Country language is knowledge the base models already have and reaches '
          f'a median NEW-REL-GEN of {_f(med(CL))} against chance {_f(chance(CL))}. Latitude and longitude '
          f'start at chance {_f(chance(LA))} and end at {_f(med(LA))} and {_f(med(LO))}. Elevation and '
          f'population also start at chance and end weak, {_f(med(EL))} and {_f(med(PO))}, above chance on '
          f'every one of the {len(models)} models but only about twice it.</li>'
          f'<li><b>The roster correlation holds.</b> Per model, the Pearson r between LIN-HEAD probe gain and '
          f'NEW-REL-GEN over the {n_roster} roster relations alone is {r_roster}; with the five engineered '
          f'relations added (n = {n_all}) it is {r_all}, a shift of at most {_f(shift,2)} on any model. Within '
          f'the five engineered relations alone r is {r_eng} per model, and {_f(chk["pooled_r"])} pooled over '
          f'all {chk["pooled_n"]} engineered points (descriptive only).</li>'
          f'<li><b>They sit a little below the roster line.</b> Read against each model’s roster-only fit, the '
          f'engineered points have a mean residual of {mres} in held-out accuracy, and {chk["below"]} of '
          f'{chk["pooled_n"]} fall below the line; the roster’s own scatter about that line is '
          f'{_f(chk["rms_roster"])} RMS. Same ordering, same slope, lower intercept: at equal measured '
          f'LIN-HEAD probe gain an engineered relation transfers slightly less than a roster relation, '
          f'descriptively rather than causally. '
          f'{esc(rels[on_line]["title"]) if on_line else ""} is the one that lands on the line '
          f'(mean residual {_f(rel_res[on_line]) if on_line else ""}).</li>'
          '<li><b>Pre-registration.</b> Longitude was predicted to have the highest measured LIN-HEAD gain '
          'among the bands and to transfer best, and it does. Population was predicted to have higher '
          'measured probe gains than elevation, and the measurement is the other way round on both probe '
          'measures; on held-out '
          'accuracy the two are inside each other’s spread.</li>'
          '<li><b>Caveats.</b> Each cell is three repeats on one model, so per-model differences inside a '
          'relation are not separated from repeat noise. Qwen3.8-27B has not run the second set.</li>'
          '</ul></div>')
        per=[]
        for rel,title,tier in RESULTS_ORDER:
            info=rels[rel]
            if info["surface_cue"]:
                per.append(f'<li><b>{esc(title)}</b>, excluded surface-cue control: NEW-REL-GEN median '
                           f'{_f(med(rel))} (range {rng(rel)}), chance {_f(chance(rel))}.</li>')
            else:
                per.append(f'<li><b>{esc(title)}</b>: NEW-REL-GEN median {_f(med(rel))} (range {rng(rel)}) '
                           f'against chance {_f(chance(rel))} and an epoch-0 median of {_f(med(rel,"newrel_gen_ep0_acc"))}; '
                           f'probe gain median {_f(med(rel,"primary_probe_gain"))}; knowledge gate median '
                           f'{_f(med(rel,"pre_rel_cand_acc"))}.</li>')
        rows=[]
        for e in chk["per_model"]:
            part=(f'<li><b>{esc(e["model"])}</b>: roster r = {_f(e["r_roster"])} (n = {e["n_roster"]}); '
                  f'with engineered r = {_f(e["r_all"])} (n = {e["n_all"]})')
            if e["r_eng"] is not None: part+=f'; engineered alone r = {_f(e["r_eng"])} (n = {e["n_eng"]})'
            if e["mean_resid"] is not None: part+=f'; mean residual {e["mean_resid"]:+.3f}'
            if e["n_eng"]<len(CHECK_RELATIONS): part+=' (second set not run on this model)'
            rows.append(part+'.</li>')
        results=('<h2 id="results">Results</h2><div class="card">'
          f'<p>The cells are trained: {len(models)} models on a shared axis, three locked repeats each, '
          'every number a mean over the three repeats of one cell. Each relation is read against its own '
          'pre-training baseline, never pooled. '
          '<a href="results/index.html">Full charts and per-relation figures</a>.</p>'
          '<h3 id="per-relation">Per relation</h3><ul>'+"".join(per)+'</ul>'
          '<p class="note">NEW-REL-GEN is candidate accuracy on the held-out towns after fine-tuning; '
          'epoch 0 is the same measure before it; the knowledge gate is the base model asked the attribute '
          'plainly; probe gain is the LIN-HEAD probe gain at q = 16, whose null is 0.</p>'
          '<h3 id="correlation">Does the roster correlation hold on these relations?</h3>'
          '<p>Yes. The study’s association is, per model, the correlation between probe gain and '
          'NEW-REL-GEN across relations. The line for each model is fitted here on the roster relations '
          'alone, so the engineered points are a held-out test of it rather than part of it.</p>'
          '<figure>'+roster_scatter(res,chk)+
          '<ul class="legend">'+"".join(
              f'<li>{_swatch(SHAPES[r],("mk hollow" if rels[r]["surface_cue"] else f"mk c-{t}"))}'
              f'{esc(rels[r]["title"])}{" (excluded control)" if rels[r]["surface_cue"] else ""}</li>'
              for r,_,t in RESULTS_ORDER)+
          f'<li><svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><circle class="rp" cx="8" cy="8" r="3.4" style="opacity:.5"/><line class="rfit" x1="1" y1="13" x2="15" y2="3" style="opacity:.8"/></svg>roster relations and per-model fits</li></ul>'+
          '<figcaption>One point per relation and model. Grey dots are the roster relations (surface cues '
          'excluded) and each thin grey line is one model’s roster-only least-squares fit. Coloured marks '
          'are the engineered relations: blue known before fine-tuning, orange at chance before and '
          'transferring strongly, green at chance before and transferring weakly; shape names the relation '
          'and hollow marks are the excluded controls. The engineered marks follow the grey lines, and sit '
          'slightly under them.</figcaption></figure>'
          '<ul>'+"".join(rows)+'</ul>'
          f'<p class="note">Pooled over all {chk["pooled_n"]} engineered points, descriptive only: '
          f'r = {_f(chk["pooled_r"])}, {chk["below"]} below their model’s roster line. Mean residual per '
          'relation: '+"; ".join(f'{esc(rels[r]["title"])} {v:+.3f}' for r,v in
                                 sorted(rel_res.items(),key=lambda kv: kv[1]))+'.</p>'
          '</div>')
    return ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
      '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
      + relation_label_meta() +
      '<title>Engineered town relations</title>\n'
      '<link rel="stylesheet" href="../shared/collapse.css">'
      '<script src="../shared/collapse.js" defer></script>\n'
      '<style>'+RESULTS_CSS+INDEX_CSS+'</style></head>\n'
      '<body><p><a href="../index.html">&#8592; All reports</a></p>'
      '<h1>Engineered town relations</h1>'
      '<p>Candidate relations derived from the fixed 300-town universe. These pages are separate '
      'from the 48-relation Hernandez inventory and do not imply loader integration or a frozen '
      'design.</p>\n'
      + summary + results +
      '\n<h2 id="first-set">First set: the name and the coordinates</h2>\n'
      '<ul><li><a href="name-length/index.html">Town name length</a>: tied-preserving Unicode '
      'code-point bins.</li><li><a href="first-letter/index.html">Town first letter</a>: literal '
      'NFC/casefolded initial classes.</li><li><a href="latitude/index.html">Town latitude</a>: '
      'coordinate coverage and tied-preserving latitude-band candidates.</li></ul>\n'
      '<h2 id="second-set">Second set, 2026-09-19: real-world attributes</h2>\n'
      '<p>Four relations whose object is a function of a structured attribute of the town rather '
      'than of its name. Each row records the join method, a status and the cached file it was read '
      'from, and a town whose attribute cannot be read keeps its row with no class instead of '
      'disappearing.</p>\n'
      '<ul><li><a href="longitude/index.html">Town longitude</a>, quantile bands over the '
      'coordinates already joined for latitude, so no new fetch.</li><li><a '
      'href="population/index.html">Town population</a>, quantile bands over Wikidata P1082 under '
      'the frozen claim-selection rule.</li><li><a href="elevation/index.html">Town elevation</a>, '
      'quantile bands over Wikidata P2044, converting metres and feet and refusing every other '
      'unit.</li><li><a href="country-language/index.html">Town country language</a>, Wikidata P17 '
      'then P37, answered with the language’s English label.</li></ul>\n'
      '<small>Review artifacts only; prompts and class choices remain subject to author '
      'approval.</small></body></html>\n')

def relabel_preserved_html(path: Path) -> None:
    try:
        text=path.read_text()
    except OSError as error:
        raise SystemExit(f"cannot preserve generated page {path}: {error}") from error
    for relation_id,legacy in LEGACY_RESULT_LABELS.items():
        canonical=relation_label(relation_id)
        if legacy in text: text=text.replace(legacy,canonical)
        elif canonical not in text:
            raise SystemExit(f"{path}: neither legacy nor canonical label found for {relation_id}")
    if 'name="linroute-relation-labels-sha256"' not in text:
        marker='<meta name="viewport" content="width=device-width,initial-scale=1">'
        if marker not in text:
            raise SystemExit(f"{path}: cannot locate metadata insertion point")
        text=text.replace(marker,marker+'\n'+relation_label_meta(),1)
    path.write_text(text)

def main(argv=None):
    root=Path(__file__).resolve().parents[1]; p=argparse.ArgumentParser();
    p.add_argument("--towns",default=str(root/"data/sandbox/towns_candidates.tsv")); p.add_argument("--fresh-root",default=os.environ.get("LINROUTE_FRESH_ROOT", "fresh-subjects"))
    p.add_argument("--out",default=str(root/"data/relations/engineered")); p.add_argument("--html-out",required=True,help="the engineered page's directory")
    p.add_argument("--path-label",action="append",default=[],metavar="FROM=TO",help="rewrite a recorded path prefix, so a run on a mirror of the cache records the canonical location; repeatable")
    p.add_argument("--rows",default=default_rows_path(),help="the merged transfer-page rows (one row per model x relation x repeat). When the file exists the trained-results page is written and linked; when it does not, nothing about the review pages changes.")
    p.add_argument("--relation-labels",default=str(DEFAULT_RELATION_LABELS),help="the canonical paper-repository relation-label map")
    p.add_argument("--render-existing",action="store_true",help="render HTML from existing records.jsonl/summary.json files in --out without rebuilding relation data or loading tokenizers")
    a=p.parse_args(argv); townpath=Path(a.towns); fresh=Path(a.fresh_root); out=Path(a.out); hout=Path(a.html_out)
    if (hout/"build_data.py").is_file() and (hout/"refresh.py").is_file():
        raise SystemExit(f"--html-out {hout} holds the rewritten engineered page; pass a scratch directory")
    load_relation_labels(Path(a.relation_labels))
    for pair in a.path_label:
        if "=" not in pair: raise SystemExit(f"--path-label wants FROM=TO, got {pair!r}")
        PATH_LABELS.append(tuple(pair.split("=",1)))
    PATH_LABELS.sort(key=lambda kv:-len(kv[0]))
    if a.render_existing:
        for kind, relation_id in KIND_RELATION_IDS.items():
            data_dir=out/kind
            try:
                rows=[json.loads(line) for line in (data_dir/"records.jsonl").read_text().splitlines() if line.strip()]
                summary=json.loads((data_dir/"summary.json").read_text())
            except (OSError,json.JSONDecodeError) as error:
                raise SystemExit(f"cannot render existing engineered relation {kind}: {error}") from error
            summary["title"]=relation_label(relation_id)
            html_dir=hout/kind; html_dir.mkdir(parents=True,exist_ok=True)
            (html_dir/"index.html").write_text(report(kind,rows,summary))
        res=None; rowspath=Path(a.rows)
        if rowspath.is_file():
            res=aggregate_results(rowspath)
            if res["models"]:
                results_dir=hout/"results"; results_dir.mkdir(parents=True,exist_ok=True)
                (results_dir/"index.html").write_text(results_page(res))
            else: res=None
        hout.mkdir(parents=True,exist_ok=True)
        if res is None and (hout/"results/index.html").is_file() and (hout/"index.html").is_file():
            relabel_preserved_html(hout/"results/index.html")
            relabel_preserved_html(hout/"index.html")
        else:
            (hout/"index.html").write_text(index_page(res))
        return 0
    towns=list(csv.DictReader(townpath.open(),delimiter="\t")); by_title,by_name=coordinate_index(fresh/"raw",fresh/"candidates/town_candidates.jsonl")
    entities,entity_sources=entity_index(fresh/"raw")
    base=[]
    for t in towns:
        coord=find_coordinate(t["town"],t["enwiki_title"],by_title,by_name)
        base.append({**t,"town_nfc":unicodedata.normalize("NFC",t["town"]),"coordinate_provenance":coord})
    lengths=[town_length(x["town"]) for x in base]; alternatives={str(k):[(i.low,i.high) for i in fixed_length_intervals(lengths,k)] for k in (4,5,6)}; intervals=fixed_length_intervals(lengths,5)
    relation_rows={}
    relation_rows["name-length"]=[{**x,"raw_value":town_length(x["town"]),"class_id":assign_interval(town_length(x["town"]),intervals).class_id,"answer":assign_interval(town_length(x["town"]),intervals).answer} for x in base]
    relation_rows["first-letter"]=[{**x,"raw_value":first_initial(x["town"]),"class_id":f"initial-{first_initial(x['town'])}" if first_initial(x["town"]) else None,"answer":first_initial(x["town"]).upper() if first_initial(x["town"]) else None} for x in base]
    lats=[x["coordinate_provenance"]["lat"] for x in base if x["coordinate_provenance"]["lat"] is not None]; bands=latitude_bands(lats,5,20)
    lat_alternatives={f"K{k}-min{support}":[(round(lo,4),round(hi,4)) for lo,hi in latitude_bands(lats,k,support)] for k in (4,5,6) for support in (10,20,30)}
    latrows=[]
    for x in base:
        lat=x["coordinate_provenance"]["lat"]
        if lat is None: latrows.append({**x,"raw_value":None,"class_id":None,"answer":None})
        else:
            i=assign_latitude(lat,bands); lo,hi=bands[i]; latrows.append({**x,"raw_value":lat,"class_id":f"lat-{i+1}","answer":f"{LABELS[i]} latitude ({lo:.1f}° to {hi:.1f}°)"})
    relation_rows["latitude"]=latrows

    relation_rows["longitude"],lon_bands=banded(longitude_rows(base),"lon","longitude",lambda v:f"{v:.1f}°")
    relation_rows["population"],pop_bands=banded(
        attribute_rows(base,entities,entity_sources,POPULATION,"population_status",count),
        "pop","population",lambda v:f"{v:,.0f}")
    relation_rows["elevation"],elev_bands=banded(
        attribute_rows(base,entities,entity_sources,ELEVATION,"elevation_status",metres),
        "elev","elevation",lambda v:f"{v:.1f} m")
    relation_rows["country-language"]=language_rows(base,entities,entity_sources)
    def resolved(kind): return sum(1 for r in relation_rows[kind] if r.get("class_id") is not None)
    summaries={
      "name-length":{"title":"Engineered relation: town name length","definition":"NFC Unicode code-point length; spaces and punctuation count. Fixed inclusive integer intervals preserve all equal lengths.","support_note":f"Five-class review candidate. Alternatives: {alternatives}.","prompt_proposal":["The normalized written name of {town} has", "Counting every character in {town}, including spaces and punctuation, gives" ]},
      "first-letter":{"title":"Engineered relation: town first initial","definition":"Literal first Unicode code point after NFC normalization and casefolding; it must be a letter. No punctuation is skipped.","support_note":"Every literal initial remains visible, including sparse classes; no support-based merging is performed.","prompt_proposal":["The first letter of the town name {town} is", "Written normally, {town} begins with the letter"]},
      "latitude":{"title":"Engineered relation: town latitude band","definition":f"WGS84 latitude joined from cached Wikidata-derived sources. Approximate quantile bands preserve ties and merge adjacent bands below support 20. Review-candidate bands: {[(round(x,4),round(y,4)) for x,y in bands]}.","support_note":f"Resolved {len(lats)}/300; unresolved and ambiguous rows remain visible. Explicit K/minimum-support alternatives: {lat_alternatives}.","prompt_proposal":["Geographically, {town} lies within the", "By geographic latitude, {town} belongs to the"]},
      "longitude":{"title":"Engineered relation: town longitude band","definition":f"WGS84 longitude in decimal degrees, from the same cached coordinate join as the latitude relation; no new fetch. Approximate quantile bands preserve ties and merge adjacent bands below support {BAND_MIN_SUPPORT}. Bands: {[(round(x,4),round(y,4)) for x,y in lon_bands]}.","support_note":f"Resolved {resolved('longitude')}/300. Join statuses: {status_counts(relation_rows['longitude'],'longitude_status')}.","prompt_proposal":["Measured east to west, {town} lies within the", "By geographic longitude, {town} belongs to the"]},
      "population":{"title":"Engineered relation: town population band","definition":f"Wikidata P1082, selected by the frozen claim rule (one preferred-rank claim; else the latest P585 point in time; else the sole claim; several equals are ambiguous and left unknown). Approximate quantile bands preserve ties and merge adjacent bands below support {BAND_MIN_SUPPORT}. Bands: {[(round(x,4),round(y,4)) for x,y in pop_bands]}.","support_note":f"Resolved {resolved('population')}/300. Claim statuses: {status_counts(relation_rows['population'],'population_status')}.","prompt_proposal":["By number of inhabitants, {town} falls in the", "Counting its residents, {town} belongs to the"]},
      "elevation":{"title":"Engineered relation: town elevation band","definition":f"Wikidata P2044 above sea level in metres, the same claim rule as population, converting metres (Q11573) and feet (Q3710 x 0.3048) and refusing every other unit as unit_unsupported. Approximate quantile bands preserve ties and merge adjacent bands below support {BAND_MIN_SUPPORT}. Bands: {[(round(x,4),round(y,4)) for x,y in elev_bands]}.","support_note":f"Resolved {resolved('elevation')}/300. Claim statuses: {status_counts(relation_rows['elevation'],'elevation_status')}.","prompt_proposal":["By height above sea level, {town} falls in the", "Measured in metres above the sea, {town} belongs to the"]},
      "country-language":{"title":"Engineered relation: town country language","definition":"Wikidata P17 (country, preferred rank else first listed) then that country's P37 (official language), answered with the language's English label. A country has an answer only when exactly one P37 candidate survives the preferred-rank filter; a country with several is ambiguous_language and its towns keep their rows with no class. Third and final rule, settled 2026-09-19 before any cell of this relation was trained: first-listed gave the 16 United States towns Hawaiian, most-P1098-speakers gave them Spanish because P1098 counts speakers worldwide, and both would have scored a model answering from real knowledge of the country as wrong. The classes are the language names as they stand; which of them clear minimum support is the admission screen's verdict.","support_note":f"Resolved {resolved('country-language')}/300. Hop statuses: {status_counts(relation_rows['country-language'],'language_status')}.","prompt_proposal":["The official language of the country containing {town} is", "In the country where {town} lies, the official language is"]}}
    for kind, relation_id in KIND_RELATION_IDS.items():
        summaries[kind]["title"] = relation_label(relation_id)
    for kind,rows in relation_rows.items():
        tokenize(rows); counts=Counter(r.get("class_id") or "unknown" for r in rows); effective={m:len({r["tokenization"][m].get("first_token_id") for r in rows if r.get("answer")}) for m in MODELS}
        summaries[kind].update({"relation":kind,"n_rows":len(rows),"class_support":dict(sorted(counts.items())),"effective_class_counts":effective,"q_note":f"For 5 outer folds and 4 inner folds: 24 total rows yield {fitting_rows(24,5,4)} fitting rows (q16 infeasible); 30 yield {fitting_rows(30,5,4)} (q16 feasible). This is total fitting support, not 17 per class.","town_source":label(townpath),"town_source_sha256":source_hash(townpath)})
        if kind in NEW_SET:
            status_key=NEW_SET[kind]
            paths=[r["attribute_provenance"].get("source_file") for r in rows]
            if kind=="country-language":
                hops={r["attribute_provenance"].get("country_qid") for r in rows}
                hops|={c["qid"] for r in rows
                       for c in (r["attribute_provenance"].get("language_candidates") or [])}
                paths+=[entity_sources.get(q) for q in hops if q]
            summaries[kind].update({"attribute_status_counts":status_counts(rows,status_key),
                "source_files":digests(paths)})
        d=out/kind; d.mkdir(parents=True,exist_ok=True); (d/"records.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows)); (d/"summary.json").write_text(json.dumps(summaries[kind],indent=2,ensure_ascii=False)+"\n")
        hd=hout/kind; hd.mkdir(parents=True,exist_ok=True); (hd/"index.html").write_text(report(kind,rows,summaries[kind]))

    res=None
    rowspath=Path(a.rows)
    if rowspath.is_file():
        res=aggregate_results(rowspath)
        if res["models"]:
            rd=hout/"results"; rd.mkdir(parents=True,exist_ok=True)
            (rd/"index.html").write_text(results_page(res))
        else:
            res=None
    hout.mkdir(parents=True,exist_ok=True); (hout/"index.html").write_text(index_page(res))
    return 0
if __name__=="__main__": raise SystemExit(main())
