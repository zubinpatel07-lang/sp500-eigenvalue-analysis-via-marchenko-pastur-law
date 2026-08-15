"""
ticker_industry.py

This file associates an industry with each stock ticker used in the dataset, and assigns a color to each industry. The file is used to color-code eigenvector loading plots by industry. 

The industries are classified by the Global Industry Classification Standard. The classifications used were from https://en.wikipedia.org/wiki/List_of_S%26P_500_companies.
GICS defines 11 sectors total and all 11 are respresented in this dictionary.

(The tickers only include a subset of stocks in the entire dataset.)
"""

TICKER_INDUSTRY = {
    "LUV": "Industrials", "AAL": "Industrials", "ALK": "Industrials",
    "UAL": "Industrials", "DAL": "Industrials",

    "RTN": "Industrials", "LMT": "Industrials", "NOC": "Industrials",
    "LLL": "Industrials", "HII": "Industrials", "GD": "Industrials",

    "NSC": "Industrials", "EFX": "Industrials", "FLR": "Industrials", "NLSN": "Industrials",
    "FBHS": "Industrials", "ITW": "Industrials", "HON": "Industrials", "TMK": "Industrials",

  
    "MPC": "Energy", "ANDV": "Energy", "VLO": "Energy", "PSX": "Energy",
    "MRO": "Energy", "DVN": "Energy", "EOG": "Energy", "APA": "Energy",
    "HES": "Energy", "COP": "Energy", "APC": "Energy", "NBL": "Energy",
    "CXO": "Energy", "NFX": "Energy", "OXY": "Energy", "RRC": "Energy",
    "XEC": "Energy", "HAL": "Energy", "EQT": "Energy", "HP": "Energy",

  
    "PVH": "Consumer Discretionary", "LEN": "Consumer Discretionary",
    "ORLY": "Consumer Discretionary", "AZO": "Consumer Discretionary",
    "MHK": "Consumer Discretionary", "WHR": "Consumer Discretionary",
    "M": "Consumer Discretionary", "JWN": "Consumer Discretionary", "KSS": "Consumer Discretionary",
    "GPS": "Consumer Discretionary", "LB": "Consumer Discretionary", "FL": "Consumer Discretionary",
    "TJX": "Consumer Discretionary", "ROST": "Consumer Discretionary",
    "NWL": "Consumer Discretionary",


    "ED": "Utilities", "XEL": "Utilities", "WEC": "Utilities", "CMS": "Utilities",
    "ES": "Utilities", "DUK": "Utilities", "SO": "Utilities", "DTE": "Utilities",
    "PNW": "Utilities", "AEP": "Utilities", "AEE": "Utilities", "NEE": "Utilities",
    "AWK": "Utilities", "ETR": "Utilities", "EIX": "Utilities", "PEG": "Utilities",


    "PBCT": "Financials", "MTB": "Financials", "HBAN": "Financials", "PNC": "Financials",
    "KEY": "Financials", "RF": "Financials", "USB": "Financials", "FITB": "Financials",
    "STI": "Financials", "WFC": "Financials", "BBT": "Financials", "AON": "Financials",
    "BLK": "Financials", "BRK.B": "Financials", "PFG": "Financials", "IVZ": "Financials",
    "AMP": "Financials", "BEN": "Financials", "L": "Financials",


    "AVGO": "Information Technology", "AMAT": "Information Technology",
    "LRCX": "Information Technology", "TXN": "Information Technology",
    "INTC": "Information Technology",


    "SPG": "Real Estate", "KIM": "Real Estate", "MAC": "Real Estate", "GGP": "Real Estate",
    "SLG": "Real Estate", "UDR": "Real Estate", "ESS": "Real Estate", "AVB": "Real Estate",
    "AIV": "Real Estate", "BXP": "Real Estate", "EQR": "Real Estate", "ARE": "Real Estate",


    "TGT": "Consumer Staples", "DG": "Consumer Staples", "GIS": "Consumer Staples",
    "K": "Consumer Staples", "MKC": "Consumer Staples", "HRL": "Consumer Staples",
    "KR": "Consumer Staples", "PG": "Consumer Staples", "CLX": "Consumer Staples",
    "CL": "Consumer Staples", "PEP": "Consumer Staples",


    "LH": "Health Care", "UNH": "Health Care", "ANTM": "Health Care", "CNC": "Health Care",
    "AET": "Health Care", "REGN": "Health Care", "CELG": "Health Care", "GILD": "Health Care",
    "HUM": "Health Care", "CI": "Health Care", "UHS": "Health Care", "HCA": "Health Care",
    "DGX": "Health Care", "ALXN": "Health Care", "BIIB": "Health Care", "INCY": "Health Care",
    "MRK": "Health Care", "MCK": "Health Care", "LLY": "Health Care", "AMGN": "Health Care",
    "PFE": "Health Care", "ABC": "Health Care", "CAH": "Health Care", "AGN": "Health Care",


    "TWX": "Communication Services", "DISCA": "Communication Services",
    "DISCK": "Communication Services", "VIAB": "Communication Services",
    "SNI": "Communication Services", "CBS": "Communication Services",
    "DIS": "Communication Services", "CMCSA": "Communication Services",
    "IPG": "Communication Services",


    "MLM": "Materials",
}

def build_ticker_industry_color_map(industries=None):
  if industries is None:
    industries = TICKER_INDUSTRY.values()

  unique_industries = sorted(set(industries))

  all_colors = (list(plt.get_cmap("tab20").colors))

  if len(unique_industries) > len(all_colors):
    raise ValueError(f"{len(all_colors)} colors for {len(unique_industries)} industries.")

  colors = all_colors[:len(unique_industries)]

  industry_color_map = {}
  for i, industry in enumerate(unique_industries):
    col = colors[i]
    industry_color_map[industry] = col

  return industry_color_map
  
    
    
