"""Where and when do the models get it wrong?

- Error by group (season, day type, holidays): is the error systematic for
  some kinds of days?
- Monitoring KPI: 30-day rolling bias with a +/-2 % alert band, the kind of
  check that would trigger a model review in production.

Outputs: results/error_by_group.csv, results/drift_alerts.csv, figures/01-04.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
ALERT_PCT = 2.0
ROLL_DAYS = 30

# Reference categorical palette (validated slots 1-3); actuals in neutral ink.
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"rule_last_week": "#eb6834", "linear": "#1baf7a", "gradient_boosting": "#2a78d6"}
LABELS = {"rule_last_week": "Rule: same day last week", "linear": "Linear model",
          "gradient_boosting": "Gradient boosting"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left",
})


def load() -> pd.DataFrame:
    p = pd.read_csv(RESULTS / "predictions.csv", parse_dates=["date"])
    d = pd.read_csv(ROOT / "data" / "daily.csv", parse_dates=["date"])
    p = p.merge(d[["date", "season", "weekday", "is_holiday", "is_xmas_break"]], on="date")
    p["error_pct"] = p["error_gwh"] / p["demand_gwh"] * 100
    p["day_type"] = p["weekday"].map(
        lambda w: "Sunday" if w == "Sunday" else "Saturday" if w == "Saturday"
        else "Monday" if w == "Monday" else "Tue-Fri")
    p["holiday"] = p["is_holiday"].map({0: "normal day", 1: "public holiday"})
    return p


def error_by_group(p: pd.DataFrame) -> pd.DataFrame:
    out = []
    for dim in ["season", "day_type", "holiday"]:
        g = (p.groupby(["model", dim])["error_pct"]
             .agg(bias_pct="mean", MAPE_pct=lambda e: e.abs().mean(), days="count")
             .reset_index().rename(columns={dim: "group"}))
        out.append(g.assign(dimension=dim))
    res = pd.concat(out)[["dimension", "group", "model", "days", "MAPE_pct", "bias_pct"]].round(2)
    res.to_csv(RESULTS / "error_by_group.csv", index=False)
    return res


def fig_actual_vs_pred(p: pd.DataFrame) -> None:
    # Zoom into eight weeks around Christmas: daily detail is unreadable over two years.
    p = p[(p["date"] >= "2024-12-01") & (p["date"] <= "2025-01-26")]
    wide = p.pivot_table(index="date", columns="model", values="pred_gwh")
    actual = p.drop_duplicates("date").set_index("date")["demand_gwh"]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.axvspan(pd.Timestamp("2024-12-23 12:00"), pd.Timestamp("2025-01-06 12:00"),
               color=GRID, alpha=0.6, lw=0)
    ax.text(pd.Timestamp("2024-12-24"), 197, "Christmas break", color=MUTED, fontsize=9)
    ax.plot(actual.index, actual, color=INK, lw=2, marker="o", ms=3, label="Actual demand")
    for m in ["rule_last_week", "gradient_boosting"]:
        ax.plot(wide.index, wide[m], color=COLORS[m], lw=2, label=LABELS[m])
    ax.set_title("Daily Swiss electricity demand vs forecasts (test data, Dec 2024 - Jan 2025)")
    ax.set_ylabel("GWh per day")
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.12))
    fig.savefig(FIGURES / "01_actual_vs_forecast.png")
    plt.close(fig)


def fig_model_comparison() -> None:
    m = pd.read_csv(RESULTS / "metrics.csv").set_index("model")
    order = ["rule_last_week", "linear", "gradient_boosting"]
    fig, ax = plt.subplots(figsize=(6.5, 2.4))
    y = range(len(order))
    ax.barh(y, m.loc[order, "MAPE_pct"], color=[COLORS[o] for o in order], height=0.55)
    for i, o in enumerate(order):
        ax.text(m.loc[o, "MAPE_pct"] + 0.08, i, f"{m.loc[o, 'MAPE_pct']:.1f} %",
                va="center", color=INK)
    ax.set_yticks(list(y), [LABELS[o] for o in order])
    ax.invert_yaxis()
    ax.set_xlabel("Mean absolute percentage error on 2024-2025 (lower is better)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Forecast error on unseen years")
    fig.savefig(FIGURES / "02_model_comparison.png")
    plt.close(fig)


def drift_monitor(p: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.axhspan(-ALERT_PCT, ALERT_PCT, color=GRID, alpha=0.6, lw=0)
    ax.axhline(0, color=MUTED, lw=0.8)
    alerts = []
    for m in ["linear", "gradient_boosting"]:
        s = (p[p["model"] == m].set_index("date")["error_pct"]
             .rolling(ROLL_DAYS, min_periods=ROLL_DAYS).mean())
        ax.plot(s.index, s, color=COLORS[m], lw=2, label=LABELS[m])
        flagged = s[s.abs() > ALERT_PCT]
        alerts.append(pd.DataFrame({"model": m, "date": flagged.index,
                                    "rolling_bias_pct": flagged.round(2).values}))
    a = pd.concat(alerts)
    a.to_csv(RESULTS / "drift_alerts.csv", index=False)
    ax.text(p["date"].min(), ALERT_PCT + 0.2, f"alert band ±{ALERT_PCT:.0f} %",
            color=MUTED, fontsize=9)
    ax.set_title(f"Monitoring: {ROLL_DAYS}-day rolling bias (actual − forecast)")
    ax.set_ylabel("% of demand\n(< 0: model too high)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(frameon=False, loc="lower left")
    fig.savefig(FIGURES / "03_drift_monitor.png")
    plt.close(fig)
    share = a.groupby("model").size() / p.groupby("model")["date"].nunique()
    print("share of days with drift alert:\n" + (share.dropna() * 100).round(1).to_string())


def fig_bias_by_group(g: pd.DataFrame) -> None:
    sub = g[g["model"] == "gradient_boosting"].copy()
    sub["label"] = sub["group"]
    order = ["winter", "spring", "summer", "autumn", "Tue-Fri", "Monday", "Saturday",
             "Sunday", "normal day", "public holiday"]
    sub = sub.set_index("label").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    y = range(len(sub))
    ax.barh(y, sub["bias_pct"], color=COLORS["gradient_boosting"], height=0.55)
    ax.axvline(0, color=MUTED, lw=0.8)
    for i, (b, n) in enumerate(zip(sub["bias_pct"], sub["days"])):
        ax.text(b - 0.1 if b < 0 else b + 0.1, i, f"{b:+.1f} %  (n={n})",
                va="center", ha="right" if b < 0 else "left", color=INK, fontsize=9)
    ax.set_yticks(list(y), sub["label"])
    ax.invert_yaxis()
    ax.set_xlim(min(sub["bias_pct"].min() * 1.9, -1), max(sub["bias_pct"].max() * 1.9, 1))
    for yy in (3.5, 7.5):
        ax.axhline(yy, color=GRID, lw=1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean bias, % of demand (< 0: model forecasts too high)")
    ax.set_title("Gradient boosting: systematic error by type of day")
    fig.savefig(FIGURES / "04_bias_by_group.png")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    p = load()
    g = error_by_group(p)
    print(g[g["model"] == "gradient_boosting"].to_string(index=False))
    fig_actual_vs_pred(p)
    fig_model_comparison()
    drift_monitor(p)
    fig_bias_by_group(g)


if __name__ == "__main__":
    main()
