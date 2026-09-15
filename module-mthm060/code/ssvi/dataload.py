# import logging
# import pandas as pd

# from datetime import datetime
# from pathlib import Path

# logging.basicConfig()
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

# ------------------------------------------------------------------------------

# def find_processed_bhavs(basedir: Path|str) -> list:
#     """
#     Returns a list of all files in `basedir` that end with "*-allbhav-iv.csv".

#     Parameters:
#     `basedir`: Path|str:
#         The directory to scan for files.
#     """
#     _basedir = Path(basedir)
#     files = list(_basedir.rglob("*-allbhav-iv.csv"))
#     logger.info(f"{datetime.now()}: Found {len(files)} files.")
#     return files


# def load_processed_bhav(filepath: Path|str) -> pd.DataFrame:
#     """
#     Loads a single processed Bhavcopy, where "processed" means:
#     * Only Nifty options exist,
#     * Model-implied volatility has been obtained for each Nifty option, and
#     * Only OTM Nifty options and their corresponding IVs have been retained.

#     Expects at least the following columns in the CSV located at `filepath`:
#     `[date, expiry_date, strike, cp_flag, close, oi, nifty, div_yield,
#       years_to_expiry, iv, K]`

#     Returns a pd.DataFrame.
#     """
#     df = pd.read_csv(
#         filepath,
#         names     = [
#             "date",
#             "expiry_date",
#             "strike",
#             "cp_flag",
#             "close",
#             "oi",
#             "nifty",
#             "div_yield",
#             "years_to_expiry",
#             "iv",
#             "K"
#         ],
#         header    = 0,
#         parse_dates=[0, 1],
#         index_col = [0],
#     )

#     return df