.PHONY: setup data-check train ablation baselines all clean

PYTHON ?= python3

setup:
	bash scripts/setup.sh

data-check:
	@echo "Checking required data files..."
	@test -f data/encoder_v1_killed_binary.csv      || (echo "MISSING: data/encoder_v1_killed_binary.csv" && exit 1)
	@test -f data/decoder_v1_killed_binary.csv      || (echo "MISSING: data/decoder_v1_killed_binary.csv" && exit 1)
	@test -f data/encoder_absolute_filled_labeled.csv || (echo "MISSING: data/encoder_absolute_filled_labeled.csv" && exit 1)
	@test -f data/decoder_absolute_filled_labeled.csv || (echo "MISSING: data/decoder_absolute_filled_labeled.csv" && exit 1)
	@echo "All 4 data files present."

train:
	$(PYTHON) hierarchical_graph_category_rootcause/train.py --arch both

ablation:
	$(PYTHON) hierarchical_graph_category_rootcause/evaluate.py --arch both

baselines:
	$(PYTHON) baselines/run_baselines.py --arch both

all: data-check train ablation baselines
	@echo "Full reproduction complete."

clean:
	rm -rf results/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
