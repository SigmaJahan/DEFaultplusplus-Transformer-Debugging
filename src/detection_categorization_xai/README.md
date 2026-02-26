# Stage 1-2: Detection, Categorization, XAI

Pipeline module for Stage 1 detection and Stage 2 categorization/XAI.

## Module Contents

- `data/`: input datasets (`.pkl`, `.csv`)
- `preprocess.py`: preprocessing entry point
- `run_classifiers.py`: model training/evaluation
- `run_xai.py`: explanation generation
- `run_all.sh`, `run_parallel.sh`: batch runners

## Run

From repository root:

```bash
cd src/detection_categorization_xai
python preprocess.py
python run_classifiers.py --data data/enc_v1_detection.pkl --out ../../results/detection/enc_detection.json
python run_xai.py --data data/enc_v1_categorization.pkl --results ../../results/categorization/enc_categorization.json --out ../../results/explanations/xai_enc_categorization.json
```
