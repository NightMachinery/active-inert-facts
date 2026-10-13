NODE ?= node
PYTHON ?= python3
RSVG_CONVERT ?= rsvg-convert
SOURCE_DATE_EPOCH ?= 1789948800
PROBE_NAMES := ../../metadata/probe-names.json5
VARIANTS := $(shell $(NODE) -e 'const s=require("./spec.json");process.stdout.write(Object.values(s.variants).map(v=>v.file).join(" "))')

include ../shared/render-guard.mk

.PHONY: all render pdf preview validate synthetic real
all: pdf validate

synthetic: render-guard
	$(NODE) make-synthetic.mjs --spec spec.json

real: render-guard
	test -n "$(ROWS)" || { echo "set ROWS=<pool run rows jsonl ...>"; exit 1; }
	$(NODE) freeze.mjs --spec spec.json $(foreach r,$(ROWS),--rows $(r)) $(if $(CE),--ce $(CE))
	$(MAKE) all preview

render: render-guard
	$(NODE) render.mjs --spec spec.json --probe-names $(PROBE_NAMES) --aggregate aggregate.json

pdf: render-guard render
	for v in $(VARIANTS); do SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) $(RSVG_CONVERT) --format=pdf --output=$$v.pdf $$v.svg || exit 1; done

preview: render-guard render
	for v in $(VARIANTS); do $(RSVG_CONVERT) --width=1440 --output=$$v.png $$v.svg || exit 1; done

validate:
	$(PYTHON) validate.py --spec spec.json --probe-names $(PROBE_NAMES) --aggregate aggregate.json $(foreach v,$(VARIANTS),--svg $(v).svg)
