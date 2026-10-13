NODE ?= node
PYTHON ?= python3
RSVG_CONVERT ?= rsvg-convert
SOURCE_DATE_EPOCH ?= 1789948800
PROBE_NAMES := ../../metadata/probe-names.json5
SPEC ?= spec.json
AGG ?= aggregate.json
AGG_RANGEBARS ?= $(AGG:.json=-rangebars.json)
AGG_CORRELATION ?= $(AGG:.json=-correlation.json)
EXTRA = --aggregate-rangebars $(AGG_RANGEBARS) --aggregate-correlation $(AGG_CORRELATION)
VARIANTS = $(shell $(NODE) render.mjs --spec $(SPEC) --list-files)

include ../shared/render-guard.mk

.PHONY: all render pdf preview validate freeze interim phase1 check-phase1
all: pdf validate

freeze: render-guard
	$(NODE) freeze.mjs --spec $(SPEC)

render: render-guard
	$(NODE) render.mjs --spec $(SPEC) --probe-names $(PROBE_NAMES) --aggregate $(AGG) $(EXTRA)

pdf: render-guard render
	for v in $(VARIANTS); do SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) $(RSVG_CONVERT) --format=pdf --output=$$v.pdf $$v.svg || exit 1; done

preview: render-guard render
	for v in $(VARIANTS); do $(RSVG_CONVERT) --width=1440 --output=$$v.png $$v.svg || exit 1; done

validate:
	$(PYTHON) validate.py --spec $(SPEC) --probe-names $(PROBE_NAMES) --aggregate $(AGG) $(EXTRA) $(foreach v,$(VARIANTS),--svg $(v).svg)
	$(PYTHON) ../shared/model_order.py spec.json

interim: render-guard
	$(MAKE) freeze SPEC=spec-interim.json
	$(MAKE) all preview SPEC=spec-interim.json AGG=aggregate-interim.json

phase1:
	$(MAKE) all preview SPEC=spec-phase1.json AGG=aggregate-phase1.json

check-phase1:
	$(NODE) freeze.mjs --spec spec-phase1.json --check
