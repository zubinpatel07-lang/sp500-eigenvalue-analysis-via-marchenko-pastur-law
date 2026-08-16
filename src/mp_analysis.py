"""
mp_analysis.py

Applies the Marchenko-Pastur law to the correaltion matrix of S&P 500 daily log returns in order to distinguish 
"meaningful" spectral components, or eigenvectors that represent significant correlation between a group of stocks, 
from "noise" spectral components, or eigenvectors that holds no significant correlation.

How to use:
  python mp_analysis.py --data ../data/all_stocks_5yr.csv

Outputs:
  ../figures/correlation_heatmap.png
  ../figures/eigenvalue_spectrum.png
  ../output/eigenvector_loadings.txt

"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
import matplotlib.dates as mdates
from ticker_industry import TICKER_INDUSTRY, build_ticker_industry_color_map

#Stocks used for an illustration of the correlation heatmap
#These stocks were chosen as recognizable companies that span many sectors of the market
CHOSEN_STOCKS = ['KO', 'PEP', "MDLZ", 'AAPL', 'MSFT', 'GOOGL', 'XOM', 'CVX']

def load_and_preprocess(csv_path: str) -> pd.DataFrame:
  """Loads raw price data and convertes to a matrix of daily log-returns.

  - Drop open, high, low, volume columns that we don't need, keeping only closing price
  - Pivots to a (date x ticker) symbol wide format
  - Drops tickers with more then 10 missing days, then linearly interpolates the remaining gaps
  - Converts price to log-returns
  """
  stock_file = pd.read_csv(csv_path, parse_dates=["date"])
  stock_file.drop(columns=["open", "high", "low", "volume"], inplace=True)

  prices = stock_file.pivot(index='date', columns='Name', values='close')

  missing = prices.isnull().sum()
  prices = prices.loc[:, missing <= 10]
  prices = prices.interpolate(method='linear', limit_direction='both')

  log_returns = np.log(prices/prices.shift(1)).dropna()
  return log_returns


def compute_correlation(returns: pd.DataFrame) -> pd.DataFrame:
  """Converts the return series into a Pearson correlation matrix"""
  return returns.corr()


def compute_eigendecomposition(correlation: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
  """Takes the eigenvalues of the correlation matrix
  Returns eigenvalues sorted decending and eigenvalues sorted identically. 
  """
  eigenvalues, eigenvectors = np.linalg.eigh(correlation)
  eigenvalues = eigenvalues[::-1]
  eigenvectors = eigenvectors[:, ::-1]
  return eigenvalues, eigenvectors


def compute_mp_bounds(n_stocks: int, n_days: int) -> tuple[float, float, float]:
  """Finds Marchenko-Pastur upper and lower bounds for aspect ratio q = N/T"""
  q = n_stocks / n_days
  lambda_upper = (1 + np.sqrt(q))**2
  lambda_lower = (1 - np.sqrt(q))**2
  return lambda_lower, lambda_upper, q


def plot_correlation_heatmap(returns: pd.DataFrame, tickers, save_path: str):
  """Creates a heatmap of the correlation matrix restricted to 'tickers'"""
  corr_chosen = returns[tickers].corr()
  plt.figure(figsize=(8, 6))
  sns.heatmap(corr_chosen, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
  plt.title("Correlation Matrix (Selected Stocks)")
  plt.tight_layout()
  plt.savefig(save_path, dpi=200)
  plt.close()


def plot_eigenvalue_spectrum(eigenvalues, lambda_lower, lambda_upper, save_path: str):
  """Show the full eigenvalue spectrum against the MP noise band"""
  plt.figure(figsize=(12, 5))
  plt.plot(eigenvalues, marker='o', markersize=3, linewidth=0.5)
  plt.axhline(y=lambda_upper, color='green', linestyle='--', label=f'MP bound λ+ = {lambda_upper:.2f}')
  plt.axhline(y=lambda_lower, color='red', linestyle='--', label=f'MP bound λ- = {lambda_lower:.2f}')
  plt.axhspan(lambda_lower, lambda_upper, color='grey', alpha=0.3, label='Noise Region')
  plt.xlabel('Index')
  plt.ylabel('Eigenvalue (asinh scale)')
  plt.title('All Eigenvalues')
  plt.yscale('asinh')
  plt.yticks([0, 0.5, 1, 2, 5, 10, 20, 100])
  plt.gca().yaxis.set_major_formatter(ScalarFormatter())
  plt.ylim(bottom=-0.01)
  plt.legend()
  plt.tight_layout()
  plt.savefig(save_path, dpi=200)
  plt.close()

def plot_top_eigenvector_loadings(eigenvalues, eigenvectors, columns, eigen_idx, save_path: str, top_n: int = 10):
  """Bar chart of the top-loading stocks for one eigenvector sorted by highest magnitude"""
  loadings = pd.Series(eigenvectors[:, eigen_idx], index=columns)
  
  top_idx = loadings.abs().sort_values(ascending=False).index
  if loadings[top_idx[0]] < 0:
    loadings = -loadings
 
  top_stocks = loadings.loc[top_idx[:top_n]]

  industry_color_map = build_ticker_industry_color_map()

  bar_colors = []
  for ticker in top_stocks.index:
    industry = TICKER_INDUSTRY.get(ticker, "Other")
    color = industry_color_map.get(industry, "grey")
    bar_colors.append(color)

  plt.figure(figsize=(12, 7))
  plt.barh(top_stocks.index, top_stocks.values, color=bar_colors)
  plt.axvline(0, color='black', linewidth=0.8)
  plt.xlabel('Eigenvector Loading Value')
  plt.title(f'Top 10 Stocks in Eigenvector {eigen_idx} (Eigenvalue {eigenvalues[eigen_idx]:.2f})', fontsize=14)
  plt.gca().invert_yaxis()

  present_industries = sorted(list(set(TICKER_INDUSTRY.get(ticker, 'Other') for ticker in top_stocks.index)))
  handles = [plt.Rectangle((0,0),1,1, color=industry_color_map.get(industry, "grey")) for industry in present_industries]
  plt.legend(handles, present_industries, title="Industry", bbox_to_anchor=(1.02, 0.95), loc='upper left')

  plt.tight_layout()
  plt.savefig(save_path, dpi=200, bbox_inches="tight")
  plt.close()

def plot_eigenportfolio_performance_vs_stocks(returns: pd.DataFrame, eigenvectors, eigenvector_index: int, stocks_to_plot, save_path: str, top_n: int = 15, intermediate_day_threshold: int = 8):
  """
  Plots eigenportfolio performance against individual stocks, noting dates of significant 3-day moves in these stocks. 
  """
  
  portfolio_weights = eigenvectors[:, eigenvector_index]
  eigenportfolio_returns = returns.dot(portfolio_weights)
  cumulative_eigenportfolio_returns = np.exp(eigenportfolio_returns.cumsum())
  
  rolling_3day_returns = returns[stocks_to_plot].rolling(3).sum()
  combined_3day_returns = rolling_3day_returns.sum(axis=1)

  top_drop_dates = combined_3day_returns.nsmallest(top_n).index.tolist()
  top_incline_dates = combined_3day_returns.nlargest(top_n).index.tolist()

  all_significant_dates_raw = top_drop_dates + top_incline_dates
  all_significant_dates = pd.to_datetime(all_significant_dates_raw).drop_duplicates().sort_values().tolist()

  grouped_significant_dates = []
  current_group = []

  for date in all_significant_dates:
    if not current_group:
      current_group.append(date)
    else:
      last_date_in_group = current_group[-1]
      if (date - last_date_in_group).days <= intermediate_day_threshold:
        current_group.append(date)
      else:
        grouped_significant_dates.append(current_group)
        current_group.append(date)

  if current_group:
    grouped_significant_dates.append(current_group)

  new_tick_positions = []
  new_tick_labels = []
  tick_colors = []

  for group in grouped_significant_dates:
    contains_drops = any(d in top_drop_dates for d in group)
    contains_incline = any(d in top_incline_dates for d in group)

    if contains_drops and contains_incline:
      continue
    if contains_drops:
      current_tick_color = "red" 
    else:
      current_tick_color = "green"

    new_tick_positions.append(group[0])
    new_tick_labels.append(f'{group[0].strftime("%m/%d/%Y")} ({len(group) * 3}d)')
    tick_colors.append(current_tick_color)


  plt.figure(figsize=(10, 6))

  for stock in stocks_to_plot:
    if stock in returns.columns:
      cumulative_returns = np.exp(returns[stock].cumsum())
      plt.plot(cumulative_returns.index, cumulative_returns, label=stock, linewidth=1)
    else:
      print(f"Stock {stock} not found in the returns DataFrame.")

  plt.plot(cumulative_eigenportfolio_returns.index, cumulative_eigenportfolio_returns, label=f'Eigenportfolio (Eigenvalue Index {eigenvector_index})', linewidth=2, color='purple')

  plt.title('Stock and Eigenportfolio Performance (Significant Declines and Inclines)', fontsize=16)
  plt.xlabel('Date', fontsize=12)
  plt.ylabel('Indexed Value (Base 1 at start)', fontsize=12)
  plt.legend(fontsize='large')
  plt.grid(True)

  plt.xticks(mdates.date2num(new_tick_positions), new_tick_labels, rotation=60, ha='right', fontsize=10)

  ax = plt.gca()
  for i, tick_label in enumerate(ax.get_xticklabels()):
    tick_label.set_color(tick_colors[i])

  plt.tight_layout()
  plt.savefig(save_path, dpi=200)
  plt.close()


def write_significant_eigenvectors(eigenvalues, eigenvectors, columns, lambda_upper: float, save_path: str, top_n: int=15):
  """Write the top loading tickers for each eigenvalue above the MP upper bound"""
  significant = np.where(eigenvalues > lambda_upper)[0]

  blocks = []
  for i in significant:
      loadings = pd.Series(eigenvectors[:, i], index=columns)

      #Eigenvectors hold the same information regardless if we * -1
      #Flip the largest-magnitude loading so it is always positive
      top_idx = loadings.abs().sort_values(ascending=False).index
      if loadings[top_idx[0]] < 0:
        loadings = -loadings
    
      top_stocks = loadings.loc[top_idx[:top_n]]
    
      header = (
        f"Eigenvalue {i} ({eigenvalues[i]:.2f}), "
        f"min={min(loadings):.4f}, max={max(loadings):.4f}"
      )
      blocks.append(header + "\n" + top_stocks.to_string())
  
  text = "\n\n".join(blocks)
  print(text)

  with open(save_path, "w") as f:
      f.write(text + "\n")

def main():
  parser = argparse.ArgumentParser(description="Marchenko-Pastur analysis of S&P 500 correlations.")
  parser.add_argument("--data", default="../data/all_stocks_5yr.csv", help="Path to all_stocks_5yr.csv (see data/README.md)")
  parser.add_argument("--figures-dir", default="../figures")
  parser.add_argument("--output-dir", default="../output")
  parser.add_argument("--industry-eigenvectors", nargs="+", type=int, default=[10, 12], help="Eigenvector indicies to plot bar chart of eigenvector components")
  parser.add_argument("--eigenportfolio-index", type=int, default=10, help="Eigenpair index to build the eigenportfolio from")
  parser.add_argument("--eigenportfolio-stocks", nargs="+", default=["LEN", "MHK", "NWL"], help="Individual stocks to plot the eigenportfolio against")
    
  args = parser.parse_args()

  os.makedirs(args.figures_dir, exist_ok=True)
  os.makedirs(args.output_dir, exist_ok=True)

  
  #1. Load and preprocess
  returns = load_and_preprocess(args.data)

  #2. Correlation Matrix
  correlation = compute_correlation(returns)

  #3. Eigendecomposition
  eigenvalues, eigenvectors = compute_eigendecomposition(correlation)

  #4. Marchenko_pastur bounds
  n_stocks = correlation.shape[0]
  n_days = returns.shape[0]
  lambda_lower, lambda_upper, q = compute_mp_bounds(n_stocks, n_days)
  print(f"N (stocks) = {n_stocks}, T (days) = {n_days}, q = N/T = {q:.4f}")
  print(f"MP noise band: [{lambda_lower:.3f}, {lambda_upper:.3f}]")

  #5. Figures
  plot_correlation_heatmap(returns, CHOSEN_STOCKS, os.path.join(args.figures_dir, "correlation_heatmap.png"))
  plot_eigenvalue_spectrum(eigenvalues, lambda_lower, lambda_upper, os.path.join(args.figures_dir, "eigenvalue_spectrum.png"))
  for index in args.industry_eigenvectors:
    plot_top_eigenvector_loadings(eigenvalues, eigenvectors, correlation.columns, index, os.path.join(args.figures_dir, f"industry_loadings_eig{index}.png"))
  plot_eigenportfolio_performance_vs_stocks(returns, eigenvectors, args.eigenportfolio_index, args.eigenportfolio_stocks, os.path.join(args.figures_dir, "eigenportfolio_performance.png"))

  #6. Significant eigenvectors -> output file
  write_significant_eigenvectors(eigenvalues, eigenvectors, correlation.columns, lambda_upper, os.path.join(args.output_dir, "eigenvector_loadings.txt"))

if __name__ == "__main__":
  main()
