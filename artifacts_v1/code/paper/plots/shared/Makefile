NODE ?= node
PYTHON ?= python3

.PHONY: test
test:
	$(NODE) --test relation-sets.test.mjs key.test.mjs
	NODE=$(NODE) $(PYTHON) test_relation_sets.py -v
	NODE=$(NODE) $(PYTHON) test_terms.py -v
	$(PYTHON) test_layer_order.py -v
	$(PYTHON) test_key_spacing.py -v
	NODE=$(NODE) $(PYTHON) test_json5.py -v
	NODE=$(NODE) $(PYTHON) test_half_width.py -v
