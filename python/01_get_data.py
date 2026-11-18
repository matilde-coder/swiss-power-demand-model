"""Build the daily modelling dataset: Swiss electricity demand + weather + calendar.

Sources
- Demand: Swissgrid 15-minute data ("energy consumption of end users in the
  Swiss control block"), read from the DuckDB file built by the sibling project
  ../swiss-power-balance. The daily aggregate is cached in data/daily_demand.csv,
  so the project still runs if that database is not present.
- Weather: Open-Meteo historical archive (ERA5 reanalysis, free, no API key),
  daily mean temperature and sunshine for six Swiss cities, population-weighted.

Output: data/daily.csv (one row per day, 2020-01-01 to 2025-12-31).
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from dateutil.easter import easter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DB_PATH = Path(os.environ.get(
    "SWISS_POWER_DB", ROOT.parent / "swiss-power-balance" / "data" / "swiss_power.duckdb"))
START, END = "2020-01-01", "2025-12-31"

# City, latitude, longitude, approximate agglomeration population (thousands).
CITIES = [
    ("Zurich", 47.37, 8.54, 1450),
    ("Geneva", 46.20, 6.15, 640),
    ("Basel", 47.56, 7.59, 550),
    ("Bern", 46.95, 7.45, 440),
    ("Lausanne", 46.52, 6.63, 430),
    ("Lucerne", 47.05, 8.31, 240),
]


def load_demand() -> pd.DataFrame:
    cache = DATA / "daily_demand.csv"
    if DB_PATH.exists():
        import duckdb

        con = duckdb.connect(str(DB_PATH), read_only=True)
        df = con.sql("""
            SELECT CAST(ts_local AS DATE)              AS date,
                   COUNT(*)                            AS n_quarter_hours,
                   SUM(consumption_end_users_kwh) / 1e6 AS demand_gwh_raw
            FROM grid_15min
            GROUP BY 1 ORDER BY 1
        """).df()
        # Daylight-saving days have 92 or 100 quarter-hours; rescale to 24 h so
        # every day is comparable.
        df["demand_gwh"] = df["demand_gwh_raw"] * 96 / df["n_quarter_hours"]
        df = df[["date", "demand_gwh"]]
        df.to_csv(cache, index=False)
        print(f"demand: {len(df)} days from {DB_PATH.name}")
    else:
        df = pd.read_csv(cache)
        print(f"demand: {len(df)} days from cache ({cache.name})")
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_weather() -> pd.DataFrame:
    cache = DATA / "weather_daily.csv"
    if cache.exists():
        print(f"weather: from cache ({cache.name})")
        return pd.read_csv(cache, parse_dates=["date"])

    resp = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": ",".join(str(c[1]) for c in CITIES),
            "longitude": ",".join(str(c[2]) for c in CITIES),
            "start_date": START,
            "end_date": END,
            "daily": "temperature_2m_mean,sunshine_duration",
            "timezone": "Europe/Zurich",
        },
        timeout=60,
    )
    resp.raise_for_status()
    payload = resp.json()  # one entry per location, in request order

    weights = pd.Series({c[0]: c[3] for c in CITIES}, dtype=float)
    weights /= weights.sum()
    temp, sun = {}, {}
    for (name, *_), loc in zip(CITIES, payload):
        idx = pd.to_datetime(loc["daily"]["time"])
        temp[name] = pd.Series(loc["daily"]["temperature_2m_mean"], index=idx)
        sun[name] = pd.Series(loc["daily"]["sunshine_duration"], index=idx) / 3600
    temp, sun = pd.DataFrame(temp), pd.DataFrame(sun)

    df = pd.DataFrame({
        "temp_c": (temp * weights).sum(axis=1).round(2),
        "sunshine_h": (sun * weights).sum(axis=1).round(2),
    }).rename_axis("date").reset_index()
    df.to_csv(cache, index=False)
    print(f"weather: {len(df)} days downloaded from Open-Meteo")
    return df


def national_holidays(year: int) -> dict[date, str]:
    """Holidays observed in (almost) all cantons, plus 2 January."""
    e = easter(year)
    return {
        date(year, 1, 1): "New Year",
        date(year, 1, 2): "Berchtold's Day",
        e - timedelta(days=2): "Good Friday",
        e + timedelta(days=1): "Easter Monday",
        e + timedelta(days=39): "Ascension",
        e + timedelta(days=50): "Whit Monday",
        date(year, 8, 1): "National Day",
        date(year, 12, 25): "Christmas",
        date(year, 12, 26): "St Stephen's Day",
    }


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    hol = {}
    for y in df["date"].dt.year.unique():
        hol.update(national_holidays(int(y)))
    d = df["date"]
    df["year"] = d.dt.year
    df["month"] = d.dt.month
    df["day_of_year"] = d.dt.day_of_year
    df["weekday"] = d.dt.day_name()
    df["is_holiday"] = d.dt.date.map(lambda x: x in hol).astype(int)
    # Industry and offices largely pause between Christmas and early January.
    df["is_xmas_break"] = (((d.dt.month == 12) & (d.dt.day >= 24))
                           | ((d.dt.month == 1) & (d.dt.day <= 6))).astype(int)
    df["season"] = d.dt.month.map({12: "winter", 1: "winter", 2: "winter",
                                   3: "spring", 4: "spring", 5: "spring",
                                   6: "summer", 7: "summer", 8: "summer",
                                   9: "autumn", 10: "autumn", 11: "autumn"})
    # Heating / cooling degree days (Swiss convention: heating below 12 C mean,
    # base 20 C; cooling above 18.3 C).
    df["hdd"] = (20 - df["temp_c"]).where(df["temp_c"] < 12, 0).round(2)
    df["cdd"] = (df["temp_c"] - 18.3).clip(lower=0).round(2)
    return df


def main() -> None:
    DATA.mkdir(exist_ok=True)
    df = load_demand().merge(load_weather(), on="date", how="inner")
    df = add_calendar(df)
    assert df["date"].is_unique and df.notna().all().all(), "gaps in merged data"
    df.to_csv(DATA / "daily.csv", index=False)
    print(f"daily.csv: {len(df)} rows, {df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}")


if __name__ == "__main__":
    main()
