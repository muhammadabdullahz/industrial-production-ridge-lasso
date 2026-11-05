"""Project-wide paths and settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
FIGURE_DIR = ROOT / "figures"

DATA_URL = "https://raw.githubusercontent.com/muhammadabdullahz/industrial-production-ridge-lasso/refs/heads/main/2026-09-MD.csv"
RAW_FILE = RAW_DIR / "2026-09-MD.csv"

h = 3                      # default forecast horizon (months)
CUTOFF = "2009-12-01"      # last training month
TEST_START = "2010-01-01"  # first test month
HORIZONS = [1, 3, 6, 12]

# FRED-MD mnemonic -> group (from the FRED-MD appendix)
fred_group_map = {
    # Group 1: Output and Income
    "RPI": "Output & Income", "W875RX1": "Output & Income", "INDPRO": "Output & Income",
    "IPFPNSS": "Output & Income", "IPFINAL": "Output & Income", "IPCONGD": "Output & Income",
    "IPDCONGD": "Output & Income", "IPNCONGD": "Output & Income", "IPBUSEQ": "Output & Income",
    "IPMAT": "Output & Income", "IPDMAT": "Output & Income", "IPNMAT": "Output & Income",
    "IPMANSICS": "Output & Income", "IPB51222s": "Output & Income", "IPFUELS": "Output & Income",
    "NAPMPI": "Output & Income", "CUMFNS": "Output & Income",

    # Group 2: Labor Market
    "HWI": "Labor Market", "HWIURATIO": "Labor Market", "CLF160V": "Labor Market",
    "CE16OV": "Labor Market", "UNRATE": "Labor Market", "UEMPMEAN": "Labor Market",
    "UEMPLT5": "Labor Market", "UEMP5TO14": "Labor Market", "UEMP15OV": "Labor Market",
    "UEMP15T26": "Labor Market", "UEMP27OV": "Labor Market", "CLAIMSx": "Labor Market",
    "PAYEMS": "Labor Market", "USGOOD": "Labor Market", "CES1021000001": "Labor Market",
    "USCONS": "Labor Market", "MANEMP": "Labor Market", "DMANEMP": "Labor Market",
    "NDMANEMP": "Labor Market", "SRVPRD": "Labor Market", "USTPU": "Labor Market",
    "USWTRADE": "Labor Market", "USTRADE": "Labor Market", "USFIRE": "Labor Market",
    "USGOVT": "Labor Market", "CES0600000007": "Labor Market", "AWOTMAN": "Labor Market",
    "AWHMAN": "Labor Market", "NAPMEI": "Labor Market", "CES0600000008": "Labor Market",
    "CES2000000008": "Labor Market", "CES3000000008": "Labor Market",

    # Group 3: Housing
    "HOUST": "Housing", "HOUSTNE": "Housing", "HOUSTMW": "Housing", "HOUSTS": "Housing",
    "HOUSTW": "Housing", "PERMIT": "Housing", "PERMITNE": "Housing", "PERMITMW": "Housing",
    "PERMITS": "Housing", "PERMITW": "Housing",

    # Group 4: Consumption, Orders, & Inventories
    "DPCERA3M086SBEA": "Consumption & Orders", "CMRMTSPLX": "Consumption & Orders",
    "RETAILX": "Consumption & Orders", "NAPM": "Consumption & Orders",
    "NAPMNOI": "Consumption & Orders", "NAPMSDI": "Consumption & Orders",
    "NAPMII": "Consumption & Orders", "ACOGNO": "Consumption & Orders",
    "AMDMNOx": "Consumption & Orders", "ANDENOx": "Consumption & Orders",
    "AMDMUOx": "Consumption & Orders", "BUSINVx": "Consumption & Orders",
    "ISRATIOx": "Consumption & Orders", "UMCSENTx": "Consumption & Orders",

    # Group 5: Money & Credit
    "M1SL": "Money & Credit", "M2SL": "Money & Credit", "M2REAL": "Money & Credit",
    "BOGMBASE": "Money & Credit", "TOTRESNS": "Money & Credit", "NONBORRES": "Money & Credit",
    "BUSLOANS": "Money & Credit", "REALLN": "Money & Credit", "NONREVSL": "Money & Credit",
    "CONSPI": "Money & Credit", "MZMSL": "Money & Credit", "DTCOLNVHFNM": "Money & Credit",
    "DTCTHFNM": "Money & Credit", "INVEST": "Money & Credit",

    # Group 6: Interest & Exchange Rates
    "FEDFUNDS": "Interest & FX Rates", "CP3Mx": "Interest & FX Rates",
    "TB3MS": "Interest & FX Rates", "TB6MS": "Interest & FX Rates",
    "GS1": "Interest & FX Rates", "GS5": "Interest & FX Rates",
    "GS10": "Interest & FX Rates", "AAA": "Interest & FX Rates",
    "BAA": "Interest & FX Rates", "COMPAPFFx": "Interest & FX Rates",
    "TB3SMFFM": "Interest & FX Rates", "TB6SMFFM": "Interest & FX Rates",
    "T1YFFM": "Interest & FX Rates", "T5YFFM": "Interest & FX Rates",
    "T10YFFM": "Interest & FX Rates", "AAAFFM": "Interest & FX Rates",
    "BAAFFM": "Interest & FX Rates", "TWEXAFEGSMTHx": "Interest & FX Rates",
    "EXSZUSx": "Interest & FX Rates", "EXJPUSx": "Interest & FX Rates",
    "EXUSUKx": "Interest & FX Rates", "EXCAUSx": "Interest & FX Rates",

    # Group 7: Prices
    "WPSFD49207": "Prices", "WPSFD49502": "Prices", "WPSID61": "Prices",
    "WPSID62": "Prices", "OILPRICEx": "Prices", "PPICMM": "Prices",
    "NAPMPRI": "Prices", "CPIAUCSL": "Prices", "CPIAPPSL": "Prices",
    "CPITRNSL": "Prices", "CPIMEDSL": "Prices", "CUSR0000SAC": "Prices",
    "CUSR0000SAD": "Prices", "CUSR0000SAS": "Prices", "CPIULFSL": "Prices",
    "CUSR0000SA0L2": "Prices", "CUSR0000SA0L5": "Prices", "PCEPI": "Prices",
    "DDURRG3M086SBEA": "Prices", "DNDGRG3M086SBEA": "Prices", "DSERRG3M086SBEA": "Prices",

    # Group 8: Stock Market
    "S&P 500": "Stock Market", "S&P: indust": "Stock Market",
    "S&P div yield": "Stock Market", "S&P PE ratio": "Stock Market",
    "VIXCLSx": "Stock Market"
}

color_map = {
    "Output & Income": "#1f77b4", "Labor Market": "#ff7f0e", "Housing": "#2ca02c",
    "Consumption & Orders": "#d62728", "Money & Credit": "#9467bd",
    "Interest & FX Rates": "#8c564b", "Prices": "#e377c2", "Stock Market": "#7f7f7f"
}
