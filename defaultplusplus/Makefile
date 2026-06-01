.PHONY: setup data-check train ablation baselines all clean

PYTHON          ?= python3
DATA_DIR        := ../data
# The baseline comparison scripts live in the paper-artifact bundle,
# outside this code repository. Point BASELINE_SCRIPT at them if you
# have that bundle checked out alongside this repo.
BASELINE_SCRIPT ?= ../baselines/run_baselines.py

setup:
	bash scripts/setup.sh

data-check:
	@echo "Checking required data files..."
	@test -f $(DATA_DIR)/encoder_dataset.csv || (echo "MISSING: $(DATA_DIR)/encoder_dataset.csv" && exit 1)
	@test -f $(DATA_DIR)/decoder_dataset.csv || (echo "MISSING: $(DATA_DIR)/decoder_dataset.csv" && exit 1)
	@echo "All data files present."

train:
	$(PYTHON) -m hierarchical_graph_category_rootcause.train --arch both

ablation:
	$(PYTHON) -m hierarchical_graph_category_rootcause.evaluate --arch both

baselines:
	@test -f $(BASELINE_SCRIPT) || (echo "Baseline scripts not found at $(BASELINE_SCRIPT). They ship in the paper-artifact bundle, not this code repo. Set BASELINE_SCRIPT to their location." && exit 1)
	$(PYTHON) $(BASELINE_SCRIPT) --arch both

all: data-check train ablation
	@echo "Reproduction (train + ablation) complete. Run 'make baselines' separately if the paper-artifact bundle is available."

clean:
	rm -rf ../results/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
