# S&P 500 Correlation Matrix Eigenpair Filtering via Marchenko-Pastur Law

This is an application of the Random Matrix Theory on the correlation matrix of S&P 500 stocks' return series. Through filtering using Marchenko-Pastur bounds, groups of correlated stocks can be found. 

## Overview

Given `N` stocks observed over `T` trading days, a correlation matrix estimated from this finite sample will inherently contain noise. The **Marchenko-Pastur (MP) law** describes the eigenvalue distribution expected from the covariance matrix of a random data matrix with aspect `q = N/T`. Any eigenvalue of the observed correlation matrix that falls **above** the MP upper bound `λ+` seems to be the result of correlation arising for a reason. More insight into why the eigenvalues below `λ-` are discarded, why the top loadings in eigenvectors should be correlated and economic interpretations of each eigenpair can be found at the associated paper: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7296838.

This project:
1. Builds a correlation matrix from ~5 years of daily log returns for 474 S&P 500 stocks.
2. Computes its eigenvalues/eigenvectors.
3. Compares the eigenvalues' bounds to the projected bounds under a random matrix determined by the Marchenko-Pastur law.
4. Checks the prediction that top eigenvector components are correlated by determining if the components are in the same industry.
5. Builds an "eigenportfolio" from a chosen eigenvector's weights and compares it against individual stocks 

## Setup

```bash
git clone https://github.com/zubinpatel07-lang/sp500-eigenvalue-analysis-via-marchenko-pastur-law.git
cd sp500-eigenvalue-analysis-via-marchenko-pastur-law
pip install -r requirements.txt
```

Download the dataset from Kaggle (see `data/README.md` for link) and place `all_stocks_5yr.csv` in the `data/` folder.

## Running the Code

```bash
cd src
python mp_analysis.py --data ../data/all_stocks_5yr.csv
```

Running this code with no extra arguments reproduces the default analysis (eigenvectors 10 and 12 for the industry charts; eigenvector 10 vs LEN, MHK, NWL for the eigenportfolio chart). This will regenerate:
- `figures/correlation_heatmap.png`
- `figures/eigenvalue_spectrum.png`
- `figures/eigenvector_10_chart.png`, `figures/eigenvector_12_chart.png`
- `figures/eigenportfolio_10_vs_LEN_MHK_NWL_performance.png`
- `output/eigenvector_loadings.txt`

### Command-line options

All arguments are optional. To see the full list of arguments: `python mp_analysis.py --help`.

| Argument | Default | Description |
| --- | --- | --- |
| `--data` | `../data/all_stocks_5yr.csv` | Path to the input CSV (see `data/README.md`) |
| `--figures-dir` | `../figures` | Where .png files are saved |
| `--output-dir` | `../output` | Where `eigenvector_loadings.txt` is saved |
| `--eigenvectors-index` | `10 12` | Eigenvector indicies to plot bar chart of eigenvector components |
| `--eigenportfolio-index` | `10` | Eigenvector index used to build the eigenportfolio |
| `--eigenportfolio-stocks` | `LEN MHK NWL` | One or more tickers to plot against the eigenportfolio |

For example, customizing these arguments would look like

```bash
python mp_analysis.py --eigenvectors-index 1 2 --eigenportfolio-index 1 --eigenportfolio-stocks BLK BRK.B PFG IVZ
```

## Results

See `figures/` for the correlation heatmap, eigenvalue spectrum, industry-colored loading charts and eigenportfolio performance chart.
See `output/eigenvector_loadings.txt` for the full top-loading stock list for each significant eigenvalue.  

## Data Source

[S&P 500 stock data](https://www.kaggle.com/datasets/camnugent/sandp500) (Kaggle, camnugent)
5 years of daily OHLCV data for S&P 500 companies.

["List of S&P 500 companies"](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)
Sector classifications in `src/ticker_industry.py` are sourced from GICS (Global Industry Classification Standard).
