# Notes for COGS 202 Final Project

<!-- Installed openblas: `brew install openblas` (not sure if necessary now that using uv) -->

To install, did `uv init -p 3.10` then `uv add -r requirements.txt` and `uv sync`. Then `uv pip install -e .`. And `uv add tqdm`.

Then:

1. `uv run scripts/01_preprocess_data.py`
2. `uv run scripts/02_run_simulations.py rich_50` (first, update A_error_sd and B_error_sd in that script)
3. `uv run scripts/02_run_simulations.py lazy_50` (first, update A_error_sd and B_error_sd in that script)
4. `notebooks/figure2_transfer_interference.ipynb` / `notebooks/figure3_anns.ipynb` / `notebooks/figure4_individual_differences.ipynb`

TODO: Use all 305 subjects instead of just randomly sampled 20
And see any other TODOs in source
