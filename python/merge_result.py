# merge_dtw_results.py
# Scan F:\AFPD\dtw_csv for {station}_{year}_{signal}_{sat}_DTW.csv
# Merge by (year, signal, sat) across stations -> one wide CSV per group:
# F:\AFPD\dtw_merged_csv\merged_{year}_{signal}_{sat}.csv

import os
import re
import logging
import pandas as pd
from collections import defaultdict

# === Config ===
INPUT_DIR  = r"F:\AFPD\2025"
OUTPUT_DIR = r"F:\AFPD\2025\dtw_merged_csv"

# Regex for filenames like: ous2_2025_S1W_G25_DTW.csv
FNAME_RE = re.compile(
    r'^(?P<station>[A-Za-z0-9]+)_(?P<year>\d{4})_(?P<signal>[A-Za-z0-9]+)_(?P<sat>[A-Za-z]\d{2})_DTW\.csv$',
    re.IGNORECASE
)

def setup_logger():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(OUTPUT_DIR, "merge.log")
    logger = logging.getLogger("merge_dtw")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(fmt); fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setFormatter(fmt); ch.setLevel(logging.INFO)
        logger.addHandler(fh); logger.addHandler(ch)
    return logger

def list_candidate_files(input_dir):
    for name in os.listdir(input_dir):
        m = FNAME_RE.match(name)
        if m:
            yield name, m.groupdict()

def read_dtw_csv(path):
    # Read as string to preserve tokens like 'firstday', 'nodata'
    # Normalize 'Date' to datetime when merging/sorting (done outside).
    try:
        df = pd.read_csv(path, dtype=str)
    except Exception as e:
        raise RuntimeError(f"Failed to read {path}: {e}")
    if 'Date' not in df.columns or 'DTW Distance' not in df.columns:
        raise RuntimeError(f"Missing columns in {path}: expect ['Date','DTW Distance']")
    return df[['Date', 'DTW Distance']].copy()

def main():
    logger = setup_logger()
    logger.info("=== Merge job started ===")
    logger.info(f"Input dir:  {INPUT_DIR}")
    logger.info(f"Output dir: {OUTPUT_DIR}")

    groups = defaultdict(list)  # key=(year,signal,sat) -> list of (station, filepath)

    # Collect files
    for fname, meta in list_candidate_files(INPUT_DIR):
        station = meta['station']
        year    = meta['year']
        signal  = meta['signal'].upper()
        sat     = meta['sat'].upper()
        fpath   = os.path.join(INPUT_DIR, fname)
        groups[(year, signal, sat)].append((station, fpath))

    if not groups:
        logger.warning("No matching files found. Check file names and input directory.")
        return

    logger.info(f"Found {sum(len(v) for v in groups.values())} files across {len(groups)} (year,signal,sat) groups.")

    # Merge per group
    for (year, signal, sat), items in sorted(groups.items()):
        out_name = f"merged_{year}_{signal}_{sat}.csv"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        logger.info(f"Merging group: year={year} signal={signal} sat={sat} -> {out_name}")
        wide = None
        station_seen = set()

        for station, fpath in sorted(items):
            if station in station_seen:
                logger.warning(f"Duplicate station '{station}' found in group ({year},{signal},{sat}); keeping first occurrence.")
                continue
            try:
                df = read_dtw_csv(fpath)
            except Exception as e:
                logger.error(str(e))
                continue

            # Parse Date to datetime for proper sort/merge, but keep original string in output
            df['_Date_dt'] = pd.to_datetime(df['Date'], errors='coerce')
            # If Date parse fails completely, fallback: keep as-is and merge by string
            if df['_Date_dt'].notna().any():
                # Prefer datetime; but retain original Date for output
                merge_key = df['_Date_dt']
            else:
                merge_key = df['Date']

            # Prepare single-station frame
            col_name = station  # one column per station
            df_station = pd.DataFrame({
                '_Date_dt': pd.to_datetime(df['Date'], errors='coerce'),
                'Date': df['Date'],
                col_name: df['DTW Distance']
            })

            # For rows where _Date_dt is NaT, keep string Date as a fallback key
            # We'll merge on two keys progressively, but simplest is outer-merge on ['Date']
            # To ensure stable ordering later, we keep _Date_dt to sort if available.
            if wide is None:
                wide = df_station
            else:
                wide = pd.merge(
                    wide, df_station,
                    on=['Date', '_Date_dt'],
                    how='outer',
                    sort=False
                )

            station_seen.add(station)
            logger.info(f"  + {os.path.basename(fpath)} -> added column '{col_name}' ({len(df)} rows)")

        if wide is None:
            logger.warning(f"No valid files to merge for group ({year},{signal},{sat}). Skipped.")
            continue

        # Sort by datetime if present; otherwise by string Date
        if wide['_Date_dt'].notna().any():
            wide = wide.sort_values(['_Date_dt', 'Date'])
        else:
            wide = wide.sort_values(['Date'])

        # Drop helper column
        wide = wide.drop(columns=['_Date_dt'])

        # Write output
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        wide.to_csv(out_path, index=False)
        logger.info(f"[OK] Wrote {out_path} ({len(wide)} rows, {len(wide.columns)-1} stations)")

    logger.info("=== Merge job finished ===")

if __name__ == "__main__":
    main()
