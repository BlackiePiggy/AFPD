# sfpd_doy_files.py
# 按 DOY 单日文件：F:\obs_output\{year}\{signal}\{station}{DOY}.csv
# 列: time_utc,sat,signal_code,station,CN0_dBHz
# 严格用 前一日（DOY-1） 与 当日 做 DTW（仅在两日共同有效点上比较；不做任何插值）

import os
from typing import Dict, Iterable, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import logging

from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Windows/多平台：显式使用 spawn 以避免多进程递归启动问题 ---
try:
    import multiprocessing as _mp
    _mp.set_start_method("spawn", force=True)
except Exception:
    pass

SAMPLE_INT = 30
DAY_SAMPLES = 24 * 60 * 60 // SAMPLE_INT  # 2880

# 星座-频段白名单
ALLOWED_SIGNALS = {
    'G': {'S1C', 'S1W', 'S2W'},        # GPS
    'C': {'S2I', 'S6I', 'S7I'},        # BeiDou
}


def _setup_logger(log_path: str, logger_name: str | None = None):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if logger_name is None:
        base = os.path.splitext(os.path.basename(log_path))[0]
        logger_name = f"sfpd_{base}"
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def _day_grid(day: pd.Timestamp) -> pd.DatetimeIndex:
    start = pd.Timestamp(day.date())
    return pd.date_range(start, periods=DAY_SAMPLES, freq=f"{SAMPLE_INT}s")


def _read_day_csv(base_root: str, year: int, signal: str, station: str, doy: int) -> pd.DataFrame | None:
    path = os.path.join(base_root, f"{year}", signal, f"{station}{doy:03d}0.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    need = {'time_utc', 'sat', 'signal_code', 'station', 'CN0_dBHz'}
    if not need.issubset(df.columns):
        raise ValueError(f"{path} missing columns: {need - set(df.columns)}")
    # retain current signal & station (lenient)
    df = df[(df['signal_code'] == signal) & (df['station'].str.lower().str[:4] == station.lower()[:4])]
    df['time_utc'] = pd.to_datetime(df['time_utc'], errors='coerce')
    df = df.dropna(subset=['time_utc'])
    return df


def _align_to_grid(df: pd.DataFrame, day: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (raw_values_2880, valid_index_array); NaN where missing.
    Steps:
    - floor to 30s grid
    - aggregate duplicates by mean (unique index)
    - reindex to full-day 30s grid
    """
    grid = _day_grid(day)
    ts = pd.to_datetime(df['time_utc'], errors='coerce')
    ts = ts.dt.tz_localize(None)
    ts = ts.dt.floor('30s')
    s = pd.Series(df['CN0_dBHz'].values, index=ts).sort_index()
    s = s.groupby(level=0).mean()
    s = s[s.index.floor('D') == day.floor('D')]
    s_grid = s.reindex(grid)
    vals = s_grid.to_numpy(dtype=float)
    valid_idx = np.where(~np.isnan(vals))[0]
    return vals, valid_idx


def _dtw_residual(a: np.ndarray, b: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """DTW for a,b; return (dist, a_aligned, b_aligned)"""
    dist, path = fastdtw(a.reshape(-1, 1), b.reshape(-1, 1), dist=euclidean)
    ta = np.array([a[i] for i, _ in path])
    tb = np.array([b[j] for _, j in path])
    L = min(len(ta), len(tb))
    return float(dist), ta[:L], tb[:L]


def _plot(day_title: str, today_raw: np.ndarray, prev_raw: np.ndarray, residual: np.ndarray, dist: float, out_png: str):
    fig, ax = plt.subplots(figsize=(8, 4), dpi=140)
    ax.plot(prev_raw, label='Prev day (raw)')
    ax.plot(today_raw, label='Today (raw)')
    ax.plot(residual, label='Residual (DTW)')
    ax.legend()
    ax.set_title(day_title)
    ax.set_xlabel('30s samples within day')
    ax.set_ylabel('C/N0 (dB-Hz)')
    ax.text(0.98, 0.02, f"DTW: {dist:.2f}", transform=ax.transAxes, ha='right', va='bottom')
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)


def _date_iter(start_date: str, end_date: str) -> List[pd.Timestamp]:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    return list(pd.date_range(start, end, freq='D'))


# ======================
# 串行版本（无插值 + 星座-频段白名单）
# ======================
def calculate_sfpd_doy_files(
    input_root: str,
    output_img_root: str,
    output_csv_root: str,
    years: Iterable[int],
    signals: Iterable[str],
    stations: Iterable[str],
    constel_prns: Dict[str, Iterable[int]] | None,
    start_date: str,
    end_date: str,
    missing_ratio_threshold: float = 0.20,
    min_overlap_points: int = 120,
    residual_threshold: float = 5.0,
    save_plots: bool = False
):
    logger = _setup_logger(os.path.join(output_csv_root, "dtw_process.log"), logger_name="sfpd_serial")
    logger.info("=== Job started ===")
    logger.info(f"input_root={input_root}, output_img_root={output_img_root}, output_csv_root={output_csv_root}")
    logger.info(f"years={years}, signals={signals}, stations={stations}, constel_prns={constel_prns}")
    logger.info(
        f"date_range={start_date}..{end_date}, missing_ratio_threshold={missing_ratio_threshold}, "
        f"min_overlap_points={min_overlap_points}, residual_threshold={residual_threshold}, save_plots={save_plots}"
    )

    os.makedirs(output_csv_root, exist_ok=True)
    target_sats = sorted([f"{cs}{int(p):02d}" for cs, prns in constel_prns.items() for p in prns]) if constel_prns else None
    all_days = _date_iter(start_date, end_date)

    for year in years:
        days_year = [d for d in all_days if d.year == year]
        if not days_year:
            logger.warning(f"No days in range for year={year}; skip.")
            continue

        for signal in signals:
            for station in stations:
                logger.info(f"Start station-year-signal: station={station}, year={year}, signal={signal}")

                # preload all DOY files
                day_to_df = {}
                for d in days_year:
                    doy = d.timetuple().tm_yday
                    df = _read_day_csv(input_root, year, signal, station, doy)
                    if df is None:
                        logger.warning(
                            f"Missing file: {os.path.join(input_root, str(year), signal, f'{station}{doy:03d}0.csv')}"
                        )
                    else:
                        logger.info(f"Loaded: {station}{doy:03d}0.csv rows={len(df)}")
                        day_to_df[d] = df

                # build sat list
                sats = target_sats if target_sats is not None else sorted({
                    rec_sat for df in day_to_df.values() if df is not None
                    for rec_sat in df['sat'].unique().tolist()
                })
                logger.info(f"Sats to process: {sats}")

                for sat in sats:
                    # 星座-频段过滤：当前 signal 是否允许处理该 sat
                    cs = sat[0] if isinstance(sat, str) and len(sat) > 0 else None
                    allowed = ALLOWED_SIGNALS.get(cs, set())
                    if signal not in allowed:
                        logger.info(f"[SKIP] {station} {year} {signal} {sat} -> signal not allowed for constellation {cs}")
                        continue

                    name = f"{station}_{year}_{signal}_{sat}"
                    img_dir = os.path.join(output_img_root, name)
                    csv_path = os.path.join(output_csv_root, f"{name}_DTW.csv")
                    if os.path.exists(csv_path):
                        logger.info(f"[SKIP] existed CSV: {csv_path}")
                        continue

                    rows = []
                    prev_day = None
                    prev_raw = None
                    prev_valid = None

                    for d in tqdm(days_year, desc=f"{name}"):
                        df_today = day_to_df.get(d)
                        if df_today is None:
                            rows.append({"Date": d.date(), "DTW Distance": "file-not-found"})
                            logger.warning(f"{name} {d.date()} -> file-not-found")
                            prev_day, prev_raw, prev_valid = None, None, None
                            continue

                        df_today_sat = df_today[df_today['sat'] == sat]
                        if df_today_sat.empty:
                            rows.append({"Date": d.date(), "DTW Distance": "no-such-sat"})
                            logger.warning(f"{name} {d.date()} -> no-such-sat")
                            prev_day, prev_raw, prev_valid = d, None, None
                            continue

                        if prev_day is None:
                            rows.append({"Date": d.date(), "DTW Distance": "firstday"})
                            today_raw, today_valid = _align_to_grid(df_today_sat, d)
                            logger.info(f"{name} {d.date()} -> firstday (valid={len(today_valid)}/{DAY_SAMPLES})")
                            prev_day, prev_raw, prev_valid = d, today_raw, today_valid
                            continue

                        today_raw, today_valid = _align_to_grid(df_today_sat, d)

                        if (d - prev_day).days != 1:
                            rows.append({"Date": d.date(), "DTW Distance": "incontinuity"})
                            logger.warning(f"{name} {d.date()} -> incontinuity (prev_day={prev_day.date()})")
                            prev_day, prev_raw, prev_valid = d, today_raw, today_valid
                            continue

                        if prev_valid is None:
                            rows.append({"Date": d.date(), "DTW Distance": "prev-empty"})
                            logger.warning(f"{name} {d.date()} -> prev-empty")
                            prev_day, prev_raw, prev_valid = d, today_raw, today_valid
                            continue

                        overlap_idx = np.intersect1d(today_valid, prev_valid)
                        if len(overlap_idx) < min_overlap_points:
                            rows.append({"Date": d.date(), "DTW Distance": "overlap_empty"})
                            logger.warning(f"{name} {d.date()} -> overlap_empty (overlap={len(overlap_idx)})")
                            prev_day, prev_raw, prev_valid = d, today_raw, today_valid
                            continue

                        a = today_raw[overlap_idx]
                        b = prev_raw[overlap_idx]
                        dist, ta, tb = _dtw_residual(a, b)
                        residual = ta - tb
                        rows.append({"Date": d.date(), "DTW Distance": dist})
                        logger.info(
                            f"{name} {d.date()} -> DTW(overlap) dist={dist:.3f}, "
                            f"today_valid={len(today_valid)}, prev_valid={len(prev_valid)}, "
                            f"overlap={len(overlap_idx)}"
                        )

                        if save_plots:
                            os.makedirs(img_dir, exist_ok=True)
                            _plot(
                                day_title=f"{station} {signal} {sat} {d.date()}",
                                today_raw=a, prev_raw=b, residual=residual, dist=dist,
                                out_png=os.path.join(img_dir, f"Date_{d.strftime('%Y-%m-%d')}.png")
                            )

                        prev_day, prev_raw, prev_valid = d, today_raw, today_valid

                    out_df = pd.DataFrame(rows)
                    os.makedirs(output_csv_root, exist_ok=True)
                    out_df.to_csv(csv_path, index=False)
                    logger.info(f"[OK] Saved CSV: {csv_path}")

    logger.info("=== Job finished ===")


# ======================
# 并行版本（无插值 + 星座-频段白名单）
# ======================
def _process_one_sat(args) -> str:
    (input_root, output_img_root, output_csv_root,
     year, signal, station, sat, days_year,
     missing_ratio_threshold, min_overlap_points,
     residual_threshold, save_plots) = args

    # 星座-频段过滤
    cs = sat[0] if isinstance(sat, str) and len(sat) > 0 else None
    allowed = ALLOWED_SIGNALS.get(cs, set())
    if signal not in allowed:
        return f"[SKIP] signal {signal} not allowed for {sat}"

    name = f"{station}_{year}_{signal}_{sat}"
    img_dir = os.path.join(output_img_root, name)
    csv_path = os.path.join(output_csv_root, f"{name}_DTW.csv")

    if os.path.exists(csv_path):
        return f"[SKIP] {name}"

    log_dir = os.path.join(output_csv_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = _setup_logger(os.path.join(log_dir, f"{name}.log"), logger_name=f"sfpd_{name}")
    logger.info(f"=== Worker start: {name} ===")

    rows = []
    prev_day = None
    prev_raw = None
    prev_valid = None

    for d in days_year:
        doy = d.timetuple().tm_yday
        df_today = _read_day_csv(input_root, year, signal, station, doy)
        if df_today is None:
            rows.append({"Date": d.date(), "DTW Distance": "file-not-found"})
            logger.warning(f"{name} {d.date()} -> file-not-found")
            prev_day, prev_raw, prev_valid = None, None, None
            continue

        df_today_sat = df_today[df_today['sat'] == sat]
        if df_today_sat.empty:
            rows.append({"Date": d.date(), "DTW Distance": "no-such-sat"})
            logger.warning(f"{name} {d.date()} -> no-such-sat")
            prev_day, prev_raw, prev_valid = d, None, None
            continue

        if prev_day is None:
            rows.append({"Date": d.date(), "DTW Distance": "firstday"})
            today_raw, today_valid = _align_to_grid(df_today_sat, d)
            logger.info(f"{name} {d.date()} -> firstday (valid={len(today_valid)}/{DAY_SAMPLES})")
            prev_day, prev_raw, prev_valid = d, today_raw, today_valid
            continue

        today_raw, today_valid = _align_to_grid(df_today_sat, d)

        if (d - prev_day).days != 1:
            rows.append({"Date": d.date(), "DTW Distance": "incontinuity"})
            logger.warning(f"{name} {d.date()} -> incontinuity (prev_day={prev_day.date()})")
            prev_day, prev_raw, prev_valid = d, today_raw, today_valid
            continue

        if prev_valid is None:
            rows.append({"Date": d.date(), "DTW Distance": "prev-empty"})
            logger.warning(f"{name} {d.date()} -> prev-empty")
            prev_day, prev_raw, prev_valid = d, today_raw, today_valid
            continue

        overlap_idx = np.intersect1d(today_valid, prev_valid)
        if len(overlap_idx) < min_overlap_points:
            rows.append({"Date": d.date(), "DTW Distance": "overlap_empty"})
            logger.warning(f"{name} {d.date()} -> overlap_empty (overlap={len(overlap_idx)})")
            prev_day, prev_raw, prev_valid = d, today_raw, today_valid
            continue

        a = today_raw[overlap_idx]
        b = prev_raw[overlap_idx]
        dist, ta, tb = _dtw_residual(a, b)
        residual = ta - tb
        rows.append({"Date": d.date(), "DTW Distance": dist})
        logger.info(
            f"{name} {d.date()} -> DTW(overlap) dist={dist:.3f}, "
            f"today_valid={len(today_valid)}, prev_valid={len(prev_valid)}, "
            f"overlap={len(overlap_idx)}"
        )

        if save_plots:
            os.makedirs(img_dir, exist_ok=True)
            _plot(
                day_title=f"{station} {signal} {sat} {d.date()}",
                today_raw=a, prev_raw=b, residual=residual, dist=dist,
                out_png=os.path.join(img_dir, f"Date_{d.strftime('%Y-%m-%d')}.png")
            )

        prev_day, prev_raw, prev_valid = d, today_raw, today_valid

    out_df = pd.DataFrame(rows)
    os.makedirs(output_csv_root, exist_ok=True)
    out_df.to_csv(csv_path, index=False)
    logger.info(f"[OK] Saved CSV: {csv_path}")
    logger.info(f"=== Worker finish: {name} ===")
    return f"[OK] {name}"


def calculate_sfpd_doy_files_parallel(
    input_root: str,
    output_img_root: str,
    output_csv_root: str,
    years: Iterable[int],
    signals: Iterable[str],
    stations: Iterable[str],
    constel_prns: Dict[str, Iterable[int]] | None,
    start_date: str,
    end_date: str,
    missing_ratio_threshold: float = 0.20,
    min_overlap_points: int = 120,
    residual_threshold: float = 5.0,
    save_plots: bool = False,
    workers: int | None = None,
):
    logger = _setup_logger(os.path.join(output_csv_root, "dtw_process_master.log"), logger_name="sfpd_parallel_master")
    logger.info("=== Parallel Job started ===")
    logger.info(f"input_root={input_root}, output_img_root={output_img_root}, output_csv_root={output_csv_root}")
    logger.info(f"years={years}, signals={signals}, stations={stations}, constel_prns={constel_prns}")
    logger.info(
        f"date_range={start_date}..{end_date}, missing_ratio_threshold={missing_ratio_threshold}, "
        f"min_overlap_points={min_overlap_points}, residual_threshold={residual_threshold}, "
        f"save_plots={save_plots}, workers={workers}"
    )

    os.makedirs(output_csv_root, exist_ok=True)
    all_days = _date_iter(start_date, end_date)

    tasks = []
    for year in years:
        days_year = [d for d in all_days if d.year == year]
        if not days_year:
            logger.warning(f"No days in range for year={year}; skip.")
            continue

        for signal in signals:
            for station in stations:
                # 自动发现该组合的卫星（如果未显式指定 constel_prns）
                sats_from_files = set()
                if not constel_prns:
                    for d in days_year:
                        doy = d.timetuple().tm_yday
                        try:
                            df = _read_day_csv(input_root, year, signal, station, doy)
                        except ValueError as e:
                            logger.warning(f"{station} {year} {signal} {d.date()} -> {e}")
                            df = None
                        if df is None:
                            continue
                        sats_from_files.update(df['sat'].unique().tolist())

                target_sats = sorted([f"{cs}{int(p):02d}" for cs, prns in (constel_prns or {}).items() for p in prns])
                sats = target_sats if target_sats else sorted(sats_from_files)

                logger.info(f"Submit tasks: station={station}, year={year}, signal={signal}, n_sats={len(sats)}")
                for sat in sats:
                    # 星座-频段过滤：不符合则不派发任务
                    cs = sat[0] if isinstance(sat, str) and len(sat) > 0 else None
                    allowed = ALLOWED_SIGNALS.get(cs, set())
                    if signal not in allowed:
                        continue

                    tasks.append((
                        input_root, output_img_root, output_csv_root,
                        year, signal, station, sat, days_year,
                        missing_ratio_threshold, min_overlap_points,
                        residual_threshold, save_plots
                    ))

    if not tasks:
        logger.info("No tasks to run. Exit.")
        return

    max_workers = workers or max(1, (os.cpu_count() or 2) - 1)
    logger.info(f"Launching pool with {max_workers} workers, total tasks={len(tasks)}")

    done = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_process_one_sat, t): t for t in tasks}
        for fut in as_completed(futures):
            done += 1
            try:
                msg = fut.result()
                logger.info(f"[{done}/{len(tasks)}] {msg}")
            except Exception as e:
                (_input_root, _output_img_root, _output_csv_root,
                 year, signal, station, sat, *_rest) = futures[fut]
                logger.exception(f"[{done}/{len(tasks)}] [ERROR] {station}_{year}_{signal}_{sat}: {e}")

    logger.info("=== Parallel Job finished ===")


# ================
# 示例入口（可按需修改）
# ================
if __name__ == "__main__":
    # 示例：并行调用（signals 可以给全量，代码会按星座筛选）
    calculate_sfpd_doy_files_parallel(
        input_root=r"F:\obs_2_output",
        output_img_root=r"F:\AFPD\obs2\plots",
        output_csv_root=r"F:\AFPD\obs2\dtw_csv",
        years=[2024,2025],
        signals=["S1W","S1C","S2W","S2I","S6I","S7I"],
        stations=["aira", "bik0", "cas1", "hal1", "hlfx", "kat1", "savo", "stfu"],
        constel_prns={
            "G": [1, 3, 5, 6, 7, 8, 9, 10, 12, 15, 17, 24, 25, 26, 27, 29, 30, 31, 32],
            "C": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        },
        start_date="2025-01-01",
        end_date="2025-07-31",
        min_overlap_points=120,
        save_plots=True,
        workers=8,
    )

    pass
