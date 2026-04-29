# Data Setup

This directory should contain 4 CSV files (~244 MB total). They are **not** tracked by git due to size.

## Required Files

| File | Size | Rows | Description | MD5 |
|------|------|------|-------------|-----|
| `encoder_v1_killed_binary.csv` | 84 MB | 9,560 | Encoder feature traces (543 features + metadata) | `0512722d154f93511427a53cbbdfbb76` |
| `decoder_v1_killed_binary.csv` | 37 MB | 9,310 | Decoder feature traces (215 features + metadata) | `ebdd57ced5f404da852cd807450c231d` |
| `encoder_absolute_filled_labeled.csv` | 85 MB | 9,560 | Encoder mutation testing labels (`killed` column) | `426a406f6e91f0d02157e9ad5ea4ed28` |
| `decoder_absolute_filled_labeled.csv` | 38 MB | 9,310 | Decoder mutation testing labels (`killed` column) | `c2958729daa4684f764e7437c5ff07fa` |

## Sources

The feature CSVs (`*_v1_killed_binary.csv`) come from the DEFault++ fault debugging study. The origin CSVs (`*_absolute_filled_labeled.csv`) come from the FrankenFormer mutation study.

Both are available in the companion data repository. Contact the authors if you need access.

## How the Data is Used

- The `v1_killed_binary.csv` files contain extracted runtime features and fault metadata (category, subcategory, layer index, severity parameters).
- The `absolute_filled_labeled.csv` files provide the `killed` column from mutation testing, which determines detection labels:
  - `killed=1` = **faulty** (mutation was detected by the test suite)
  - `killed=0` or `NaN` (baseline) = **clean**
- At load time, `train.py` merges the `killed` column into the feature CSV via the `Identifier` key.

## Expected Layout After Setup

```
data/
  README.md                              (this file)
  encoder_v1_killed_binary.csv           (84 MB)
  decoder_v1_killed_binary.csv           (37 MB)
  encoder_absolute_filled_labeled.csv    (85 MB)
  decoder_absolute_filled_labeled.csv    (38 MB)
```

## Verification

```bash
make data-check
```
