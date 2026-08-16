# Data

This project uses the **S&P 500 stock data** dataset from Kaggle:

https://www.kaggle.com/datasets/camnugent/sandp500

It contains ~5 years of daily open, high, low, close, and volume data for all current S&P 500 companies, merged into a single file: `all_stocks_5yr.csv`, with columns `date, open, high, low, close, volume, Name`.

## Why is the dataset not committed?

The CSV is large and not owned by this repo, so it is left out.

## Setup

1. Download `all_stocks_5yr.csv` from the Kaggle link above (requires a free Kaggle account), or through the Kaggle CLI
```bash
kaggle datasets download -d camnugent/sandp500
unzip sandp500.zip -d .
```
2. The download includes several files but this project only needs `all_stocks_5yr.csv`. Copy just that file into this `data/` folder and discard the rest.
3. Run the analysis from `src/mp_analysis.py` (see main README).
