.PHONY: setup data-check train ablation baselines test clean

PYTHON ?= python3
DATA_DIR := 3_Mutation-Data-from-Frakenformer

setup:
	bash scripts/setup.sh

data-check:
	@test -f $(DATA_DIR)/encoder_v1_killed_binary.csv || (echo "MISSING: $(DATA_DIR)/encoder_v1_killed_binary.csv" && exit 1)
	@test -f $(DATA_DIR)/decoder_v1_killed_binary.csv || (echo "MISSING: $(DATA_DIR)/decoder_v1_killed_binary.csv" && exit 1)
	@test -f $(DATA_DIR)/encoder_absolute_filled_labeled.csv || (echo "MISSING: $(DATA_DIR)/encoder_absolute_filled_labeled.csv" && exit 1)
	@test -f $(DATA_DIR)/decoder_absolute_filled_labeled.csv || (echo "MISSING: $(DATA_DIR)/decoder_absolute_filled_labeled.csv" && exit 1)
	@echo "All required mutation datasets are present."

train:
	$(PYTHON) -m hierarchical_graph_category_rootcause.train --arch both

ablation:
	$(PYTHON) -m hierarchical_graph_category_rootcause.evaluate --arch both

baselines:
	$(PYTHON) 4_Baseline-comparison_with_defaultpp/run_baselines.py --arch both

test:
	$(PYTHON) -m pytest 7_DEFaultpp-code/tests/test_phase0_gate.py

clean:
	rm -rf results .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
