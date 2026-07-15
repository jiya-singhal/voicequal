# voicequal benchmark

A reproducible test set for measuring voicequal's classification
accuracy across a controlled range of noise levels.

## Datasets

The test set is generated from two public research datasets:

- **VocalSet** — clean vocal recordings from 20 professional
  singers. https://zenodo.org/records/1442513
- **MUSAN** — environmental noise recordings.
  https://www.openslr.org/17/

You do NOT need to download these to run the benchmark — the
generated `test_set/` and `manifest.csv` are committed. But if
you want to regenerate:

```bash
python benchmarks/generate_test_set.py \
    --vocalset /path/to/VocalSet \
    --musan /path/to/musan
```

## Structure

- `generate_test_set.py` — creates test_set/ from raw datasets
- `run_benchmark.py` — runs voicequal.assess() on each test file
- `test_set/` — 200 pre-generated test WAVs (checked in)
- `manifest.csv` — expected tier + metadata for each test file
- `results/` — historical benchmark results per voicequal version
