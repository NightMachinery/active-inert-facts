NODE ?= node
PYTHON ?= python3
RSVG_CONVERT ?= rsvg-convert
SOURCE_DATE_EPOCH ?= 1789948800
PAPER_EXTRA ?= predictor-dots-partial predictor-dots-partial-half predictor-dots-partial-stacked
VARIANTS ?= heatmap-recall-quantile-all-sets appendix-strips-recall-part1-sets appendix-strips-recall-part2-sets scatter-grid-sets scatter-grid-sets-nostats heatmap-recall-sets heatmap-recall-sqrt-sets heatmap-recall heatmap-neutral scatter-pooled-recall rank-agreement appendix-strips
NEW_VARIANTS ?= heatmap-recall-sets heatmap-recall-sqrt heatmap-recall-quantile heatmap-recall-quantile-all heatmap-recall-sqrt-sets heatmap-recall-quantile-sets heatmap-recall-quantile-all-sets appendix-strips-sets appendix-strips-recall-sets appendix-strips-recall-part1 appendix-strips-recall-part2 appendix-strips-recall-part1-sets appendix-strips-recall-part2-sets heatmap-recall-quantile-all-sets-by-gain heatmap-recall-quantile-all-sets-models-once
ENC_VARIANTS ?= $(NEW_VARIANTS) heatmap-recall-simple heatmap-recall-observed appendix-strips-simple appendix-strips-recall scatter-pooled heatmap heatmap-interleaved model-spread scatter-pooled-neutral scatter-pooled-model model-spread-highlight
PROTO_VARIANTS ?= scatter-grid-sets scatter-pooled heatmap heatmap-interleaved model-spread rank-agreement appendix-strips
PRED_VARIANTS ?= predictor-dots predictor-dots-partial predictor-dots-partial-half predictor-dots-partial-stacked predictor-centred predictor-added-variable
VIEWS ?= main:spec.json hb:spec-hb.json
PROTO_ROWS ?= ../ranked-strips/data/rows-scatter-20260922-c2660377.jsonl ../bear10-strips/data/rows-bear10-scatter-20260922-cb9bb66f.jsonl
DATE ?= $(shell date +%Y%m%d)

include ../shared/render-guard.mk

.PHONY: all explore predictors refresh prototype prototype-freeze synthetic preview clean-out

all: render-guard
	@set -e; for v in $(VARIANTS) $(PAPER_EXTRA); do \
	  $(NODE) render.mjs --spec spec.json --variant $$v --svg $$v.svg --aggregate aggregate.json; \
	  SOURCE_DATE_EPOCH=$(SOURCE_DATE_EPOCH) $(RSVG_CONVERT) --format=pdf --output=$$v.pdf $$v.svg; \
	done
	$(PYTHON) validate.py --spec spec.json --aggregate aggregate.json $(foreach v,$(VARIANTS) $(PAPER_EXTRA),--svg $(v).svg)
	$(PYTHON) ../shared/model_order.py spec.json

explore: render-guard
	@set -e; for vw in $(VIEWS); do name=$${vw%%:*}; sp=$${vw#*:}; d=out/explore/$$name; mkdir -p $$d; \
	  for v in $(VARIANTS) $(ENC_VARIANTS); do \
	    $(NODE) render.mjs --spec $$sp --variant $$v --svg $$d/$$v.svg --aggregate $$d/aggregate.json; \
	    $(RSVG_CONVERT) --width=1440 --output=$$d/$$v.png $$d/$$v.svg; \
	  done; \
	  $(PYTHON) validate.py --spec $$sp --aggregate $$d/aggregate.json $$(for v in $(VARIANTS) $(ENC_VARIANTS); do echo --svg $$d/$$v.svg; done); \
	done

predictors: render-guard
	@mkdir -p out/explore/main; set -e; for v in $(PRED_VARIANTS); do \
	  $(NODE) render.mjs --spec spec.json --variant $$v --svg out/explore/main/$$v.svg --aggregate out/explore/main/aggregate.json; \
	  $(RSVG_CONVERT) --width=1440 --output=out/explore/main/$$v.png out/explore/main/$$v.svg; \
	done
	$(PYTHON) validate.py --spec spec.json --aggregate out/explore/main/aggregate.json $(foreach v,$(PRED_VARIANTS),--svg out/explore/main/$(v).svg)

refresh: render-guard
	@test -n "$(ROWS)" || { echo "usage: make refresh ROWS=/path/to/rows.jsonl [ROWS2=...]"; exit 2; }
	@set -e; for vw in $(VIEWS); do $(NODE) freeze.mjs --spec $${vw#*:} --rows $(ROWS) $(if $(ROWS2),--rows $(ROWS2)) --date $(DATE); done
	$(MAKE) all preview

preview: render-guard
	@set -e; for v in $(VARIANTS) $(PAPER_EXTRA); do $(RSVG_CONVERT) --width=1440 --output=$$v.png $$v.svg; done

prototype-freeze: render-guard
	$(NODE) freeze.mjs --spec spec-prototype.json $(foreach r,$(PROTO_ROWS),--rows $(r)) --date 20260922

prototype: render-guard
	@mkdir -p out/prototype; set -e; for v in $(PROTO_VARIANTS); do \
	  $(NODE) render.mjs --spec spec-prototype.json --variant $$v --svg out/prototype/$$v.svg --aggregate out/prototype/aggregate.json; \
	  $(RSVG_CONVERT) --width=1440 --output=out/prototype/$$v.png out/prototype/$$v.svg; \
	done
	$(PYTHON) validate.py --spec spec-prototype.json --aggregate out/prototype/aggregate.json $(foreach v,$(PROTO_VARIANTS),--svg out/prototype/$(v).svg)

synthetic: render-guard
	@mkdir -p out/synthetic
	$(NODE) synth.mjs --spec spec-synthetic.json --out out/synthetic/rows-SYNTHETIC.jsonl
	$(NODE) freeze.mjs --spec spec-synthetic.json --rows out/synthetic/rows-SYNTHETIC.jsonl --out out/synthetic/cells-SYNTHETIC.jsonl --no-spec-update
	@set -e; for v in $(PROTO_VARIANTS); do \
	  $(NODE) render.mjs --spec spec-synthetic.json --data out/synthetic/cells-SYNTHETIC.jsonl --variant $$v --svg out/synthetic/$$v.svg --aggregate out/synthetic/aggregate.json; \
	  $(RSVG_CONVERT) --width=1440 --output=out/synthetic/$$v.png out/synthetic/$$v.svg; \
	done
	$(PYTHON) validate.py --spec spec-synthetic.json --data out/synthetic/cells-SYNTHETIC.jsonl --aggregate out/synthetic/aggregate.json $(foreach v,$(PROTO_VARIANTS),--svg out/synthetic/$(v).svg)

clean-out:
	$(RM) -r out
