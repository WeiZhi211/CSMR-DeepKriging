"""Configuration-driven loader for real spatiotemporal interpolation datasets.

The time window is inclusive in the CSV's original integer ``t`` coordinates.
Missing site-time pairs stay NaN in the target matrix and are never made into
training or evaluation labels.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def load_real_dataset(config: dict, data_path: str | Path | None = None):
    path = Path(data_path if data_path is not None else config["data_file"])
    frame = pd.read_csv(path)
    target_col = config["target_col"]
    covariate_cols = list(config.get("covariate_cols", []))
    coord_cols = list(config["coord_cols"])
    station_col = config.get("station_col")
    time_col = config.get("time_col", "t")
    required = [target_col, time_col, *coord_cols, *covariate_cols]
    if station_col:
        required.append(station_col)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    if len(coord_cols) != 2:
        raise ValueError("coord_cols must contain exactly two coordinate columns")
    if frame.empty:
        raise ValueError(f"{path}: empty CSV")

    times = pd.to_numeric(frame[time_col], errors="raise")
    if times.isna().any() or not np.all(times.to_numpy() == times.to_numpy().astype(np.int64)):
        raise ValueError(f"{path}: {time_col} must contain finite integer time indices")
    frame[time_col] = times.astype(np.int64)
    source_min, source_max = int(times.min()), int(times.max())
    start = int(config.get("time_start", source_min))
    end = int(config.get("time_end", source_max))
    if start < source_min or end > source_max or start > end:
        raise ValueError(f"time window [{start}, {end}] outside [{source_min}, {source_max}]")
    frame = frame.loc[frame[time_col].between(start, end)].copy()
    if frame.empty:
        raise ValueError(f"{path}: no rows in time window [{start}, {end}]")

    if station_col:
        if frame[station_col].isna().any():
            raise ValueError(f"{path}: missing station identifiers")
        frame["_station_key"] = frame[station_col]
    else:
        # GSOD has no station ID: its static coordinate pair is the site key.
        frame["_station_key"] = list(zip(frame[coord_cols[0]], frame[coord_cols[1]]))
    if frame[coord_cols].isna().any().any():
        raise ValueError(f"{path}: missing site coordinates")
    if frame.duplicated([time_col, "_station_key"]).any():
        raise ValueError(f"{path}: duplicate site-time rows")

    sites = sorted(frame["_station_key"].unique())
    station_to_index = {site: index for index, site in enumerate(sites)}
    static = frame.drop_duplicates("_station_key").set_index("_station_key")
    coords = static.loc[sites, coord_cols].to_numpy(dtype=np.float32)
    low = coords.min(axis=0)
    high = coords.max(axis=0)
    coords = (coords - low) / (high - low + 1e-8)

    n_times, n_sites = end - start + 1, len(sites)
    z = np.full((n_times, n_sites), np.nan, dtype=np.float32)
    x = np.full((n_times, n_sites, len(covariate_cols)), np.nan, dtype=np.float32)
    ti = frame[time_col].to_numpy(dtype=np.int64) - start
    si = frame["_station_key"].map(station_to_index).to_numpy(dtype=np.int64)
    z[ti, si] = pd.to_numeric(frame[target_col], errors="coerce").to_numpy(dtype=np.float32)
    for index, column in enumerate(covariate_cols):
        x[ti, si, index] = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float32)

    for index, column in enumerate(covariate_cols):
        finite = np.isfinite(x[:, :, index])
        if not finite.any():
            raise ValueError(f"{path}: covariate {column} has no finite values")
        mean = float(x[:, :, index][finite].mean())
        std = float(x[:, :, index][finite].std())
        if not np.isfinite(std) or std == 0:
            std = 1.0
        x[:, :, index] = np.where(finite, (x[:, :, index] - mean) / std, 0.0)

    scale = float(config.get("target_scale", 1.0)) if config.get("normalize_target", True) else 1.0
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("target_scale must be finite and positive")
    z /= scale
    if not np.isfinite(z).any():
        raise ValueError(f"{path}: no finite target values in selected time window")
    if len(covariate_cols) != int(config.get("p_covariates", len(covariate_cols))):
        raise ValueError("p_covariates must equal len(covariate_cols)")

    offset = start - int(config.get("time_index_base", source_min))
    metadata = {
        "scale_factor": scale,
        "time_start": start,
        "time_end": end,
        "time_offset_steps": offset,
        "n_times": n_times,
        "n_sites": n_sites,
        "n_finite_targets": int(np.isfinite(z).sum()),
        "target_col": target_col,
        "covariate_cols": covariate_cols,
    }
    print(f"Loaded {path}: T={n_times}, S={n_sites}, finite targets={metadata['n_finite_targets']}, "
          f"window={start}..{end}, target scale={scale:g}")
    return z, coords, x, metadata
