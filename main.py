"""
FastAPI application for crypto candlestick analysis and probability brackets.

This file defines two API endpoints (`/api/klines` and `/api/stats`) that fetch
OHLCV data from Binance and compute statistical summaries.  The root URL `/`
serves a static HTML/JavaScript front-end from the `static/` directory.  The
front-end uses Chart.js to display histograms of high and low prices along with
their associated Gaussian curves and confidence bands (68%, 95% and 99.7%).

To deploy this application on a service like Render, place this file at the
repository root alongside a `requirements.txt` file and a `static/` folder.
The build command should install dependencies (`pip install -r requirements.txt`)
and the start command should launch Uvicorn (`uvicorn main:app --host 0.0.0.0 --port $PORT`).
"""

from __future__ import annotations

import math
from typing import List, Dict, Any, Optional

import numpy as np
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Endpoint for Binance Kline API
BINANCE_URL = "https://api.binance.com/api/v3/klines"

app = FastAPI(title="Crypto App with Brackets")

# Allow all cross-origin requests.  Render will host both the API and static
# front-end on the same domain so CORS isn't strictly necessary, but leaving
# this enabled simplifies local development and alternative deployments.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def gaussian_params(arr: List[float]) -> Dict[str, float]:
    """Compute mean and standard deviation for a list of floats.

    Ignores non-finite values.  Returns NaN values if there are fewer than
    two finite entries.
    """
    array = np.array(arr, dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 2:
        return {"mu": float("nan"), "sigma": float("nan")}
    return {"mu": float(np.mean(array)), "sigma": float(np.std(array, ddof=1))}


def probability_brackets(mu: float, sigma: float) -> Dict[str, List[Optional[float]]]:
    """Return 68%, 95% and 99.7% confidence intervals for a normal distribution.

    If either ``mu`` or ``sigma`` is non-finite or ``sigma`` is non-positive,
    returns ``None`` for all bounds.
    """
    if not (math.isfinite(mu) and math.isfinite(sigma) and sigma > 0):
        return {"p68": [None, None], "p95": [None, None], "p997": [None, None]}
    return {
        "p68": [mu - sigma, mu + sigma],
        "p95": [mu - 2 * sigma, mu + 2 * sigma],
        "p997": [mu - 3 * sigma, mu + 3 * sigma],
    }


async def get_klines(symbol: str, interval: str, limit: int) -> List[Dict[str, Any]]:
    """Fetch candlestick data from Binance.

    Parameters
    ----------
    symbol : str
        Trading pair, e.g. ``BTCUSDT``.  Will be uppercased.
    interval : str
        Candle interval (1m, 5m, 1h, etc.).
    limit : int
        Number of candlesticks to retrieve (1–1000).

    Returns
    -------
    list of dict
        Each dictionary contains ``open_time``, ``open``, ``high``, ``low``,
        ``close``, ``volume`` and ``close_time`` fields.
    """
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(BINANCE_URL, params=params)
    try:
        response.raise_for_status()
    except httpx.HTTPError as err:
        raise HTTPException(status_code=502, detail=f"Error contacting Binance: {err}")
    data = response.json()
    # Binance returns a dict with 'code' and 'msg' on error
    if isinstance(data, dict) and data.get("code"):
        raise HTTPException(status_code=400, detail=data.get("msg", "Binance error"))
    rows: List[Dict[str, Any]] = []
    for row in data:
        rows.append({
            "open_time": int(row[0]),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5]),
            "close_time": int(row[6]),
        })
    return rows


@app.get("/api/klines")
async def api_klines(
    symbol: str = Query("BTCUSDT", description="Trading pair, e.g. BTCUSDT"),
    interval: str = Query("1m", description="Candle interval (1m, 3m, 5m, 15m, etc.)"),
    limit: int = Query(200, ge=1, le=1000, description="Number of bars to return"),
) -> Dict[str, Any]:
    """Proxy endpoint that returns candlestick data from Binance."""
    rows = await get_klines(symbol, interval, limit)
    return {"symbol": symbol.upper(), "interval": interval, "limit": limit, "rows": rows}


@app.get("/api/stats")
async def api_stats(
    symbol: str = Query("BTCUSDT"),
    interval: str = Query("1m"),
    limit: int = Query(500, ge=10, le=1000),
) -> Dict[str, Any]:
    """Compute statistical summaries on candlestick data.

    Returns medians for high/low/close, covariance between close and high,
    Gaussian mean/standard deviation and probability brackets for the high
    and low distributions.
    """
    rows = await get_klines(symbol, interval, limit)
    highs = [r["high"] for r in rows]
    lows = [r["low"] for r in rows]
    closes = [r["close"] for r in rows]
    gh = gaussian_params(highs)
    gl = gaussian_params(lows)
    med_high = float(np.median(highs)) if highs else float("nan")
    med_low = float(np.median(lows)) if lows else float("nan")
    med_close = float(np.median(closes)) if closes else float("nan")
    cov_ch = float(np.cov(closes, highs, ddof=1)[0, 1]) if len(closes) >= 2 else float("nan")
    return {
        "symbol": symbol.upper(),
        "interval": interval,
        "n": len(rows),
        "medians": {"high": med_high, "low": med_low, "close": med_close},
        "cov_close_high": cov_ch,
        "gauss_high": {
            "mu": gh["mu"],
            "sigma": gh["sigma"],
            "brackets": probability_brackets(gh["mu"], gh["sigma"]),
        },
        "gauss_low": {
            "mu": gl["mu"],
            "sigma": gl["sigma"],
            "brackets": probability_brackets(gl["mu"], gl["sigma"]),
        },
    }


# Set up static file serving
from pathlib import Path
static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def index() -> FileResponse:
    """Serve the front-end HTML file from the static directory."""
    index_path = static_dir / "index.html"
    return FileResponse(str(index_path))