"""Shared CSV cache and download handling for the event study."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import yfinance as yf


def _validate_raw(frame, date_col, value_col, start, end):
    if not {date_col, value_col}.issubset(frame.columns):
        raise ValueError(f"Expected columns {date_col} and {value_col}")
    dates = pd.to_datetime(frame[date_col], errors="raise")
    values = pd.to_numeric(frame[value_col], errors="raise")
    observed = values.dropna()
    if (dates.isna().any() or dates.duplicated().any()
            or not dates.is_monotonic_increasing
            or not dates.between(pd.Timestamp(start), pd.Timestamp(end)).all()):
        raise ValueError("Dates must be unique, sorted, and within the requested period")
    if (len(observed) < 2 or not np.isfinite(observed).all()
            or not (observed > 0).all()
            or (value_col == "Close" and values.isna().any())):
        raise ValueError("Expected at least two finite, positive observations")
    frame[date_col], frame[value_col] = dates, values


def download_raw(data_dir, tickers, start, end):
    """Reuse valid CSVs; fetch and safely replace only missing or invalid inputs."""
    data_dir = Path(data_dir)
    for name, ticker in {**tickers, "fx": None}.items():
        date_col, value_col = ("Date", "Close") if ticker else ("observation_date", "DEXKOUS")
        path = data_dir / f"{name}.csv"
        try:
            _validate_raw(pd.read_csv(path), date_col, value_col, start, end)
            continue
        except (FileNotFoundError, ValueError, TypeError):
            pass

        print(f"{path.name}: cache missing or invalid; downloading.")
        temporary = None
        try:
            if ticker:
                frame = yf.download(ticker, start=start, end=end,
                                    auto_adjust=True, progress=False)
                frame.columns = frame.columns.get_level_values(0)
                frame = frame.rename_axis("Date").reset_index()
            else:
                frame = pd.read_csv(
                    "https://fred.stlouisfed.org/graph/fredgraph.csv"
                    f"?id=DEXKOUS&cosd={start}&coed={end}", na_values=["."],
                )
            _validate_raw(frame, date_col, value_col, start, end)
            # A failed write leaves the existing CSV intact.
            with NamedTemporaryFile(mode="w", dir=data_dir, prefix=f".{name}-",
                                    suffix=".csv", delete=False) as handle:
                temporary = Path(handle.name)
                frame.to_csv(handle, index=False)
            temporary.replace(path)
        except Exception as exc:
            raise RuntimeError(f"Could not download {name}; existing CSV left unchanged") from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
