NODE ?= node
PYTHON ?= python3
RSVG_CONVERT ?= rsvg-convert
PDF_FORMAT ?= pdf1.5
SOURCE_DATE_EPOCH ?= 1789948800
PROBE_NAMES := ../../metadata/probe-names.json5
SPEC ?= spec.json
AGG ?= aggregate.json
SOURCE ?=
VARIANTS = $(shell $(NODE) render.mjs --spec $(SPEC) --list-files)
ROUNDS ?= rounds

include ../shared/render-guard.mk

.PHONY: all render pdf preview rounds validate freeze check-freeze
all: pdf rounds validate

freeze: render-guard
	$(PYTHON) freeze.py --spec $(SPEC) --source $(SOURCE)/summary.json --meta $(SOURCE)/meta.json

check-freeze:
	$(PYTHON) freeze.py --spec $(SPEC) --source $(SOURCE)/summary.json --meta $(SOURCE)/meta.json --check

render: render-guard
	$(NODE) render.mjs --spec $(SPEC) --probe-names $(PROBE_NAMES) --aggregate $(AGG)

pdf: render-guard render
	for v in $(VARIANTS); do SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) $(RSVG_CONVERT) --format=$(PDF_FORMAT) --output=$$v.pdf $$v.svg || exit 1; done

preview: render-guard render
	for v in $(VARIANTS); do $(RSVG_CONVERT) --width=1440 --output=$$v.png $$v.svg || exit 1; done

rounds: render-guard
	$(NODE) render.mjs --spec $(SPEC) --probe-names $(PROBE_NAMES) --aggregate $(AGG) --rounds $(ROUNDS)
	for v in $$(find $(ROUNDS) -name '*.svg' -not -path '*/current/*' | sed 's/\.svg$$//'); do \
	  SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) $(RSVG_CONVERT) --format=$(PDF_FORMAT) --output=$$v.pdf $$v.svg || exit 1; \
	  $(RSVG_CONVERT) --width=1440 --output=$$v.png $$v.svg || exit 1; \
	done

validate:
	$(PYTHON) validate.py --spec $(SPEC) --probe-names $(PROBE_NAMES) --aggregate $(AGG) $(foreach v,$(VARIANTS),--svg $(v).svg) $(if $(wildcard $(ROUNDS)),--rounds $(ROUNDS))
	$(PYTHON) ../shared/model_order.py $(SPEC)
