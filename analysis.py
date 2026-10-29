"""
Regression discontinuity estimates, tables and figures for the paper.

Design (as in Berger & Pope 2011): one row per regular-season game, seen from the home team.
  running variable  m      = home halftime margin (home points - away points at the half)
  treatment         behind = 1 if m < 0
  outcome           win    = 100 if the home team wins, else 0, so jumps are in percentage points
Games tied at halftime and neutral-site games are dropped.

Main estimator, used throughout: OLS on games with |m| <= 10, separate linear slopes on each
side of 0, heteroskedasticity-robust (HC1) standard errors. The coefficient on `behind` is the
jump in P(win) from just ahead to just behind. Two jumps from separate samples are compared
with a z-test on their difference.

Reads data/nba_games.csv (from get_data.py), writes CSVs and figures to output/ and builds
output/tables.docx (make_tables.py). Runs in about 20 seconds.
"""

import contextlib
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from rdrobust import rdrobust
from scipy.stats import norm

from make_tables import write_tables

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "output"

# ESPN season labels: 2015 = 2014/15
ERAS = {
    "bp": ("B&P sample 1993/94-2008/09", 1994, 2009),
    "post": ("Post-B&P 2009/10-2013/14", 2010, 2014),
    "3pt": ("Three-point era 2014/15-2024/25", 2015, 2025),
}
SHORT_LINE = (1995, 1997)  # three-point line shortened to 22 ft, 1994/95-1996/97
COVID = (2020, 2021)  # 2019/20 (bubble) and 2020/21 (empty arenas)
BW = 10  # window: |halftime margin| <= 10

# The cubic is only for comparison with Berger & Pope (Table 6); high-order polynomials
# are not recommended in RD designs (Gelman & Imbens 2019).
SPECS = {
    "linear": "{y} ~ behind * m",
    "cubic": "{y} ~ behind * (m + I(m**2) + I(m**3))",
}


def load_games() -> pd.DataFrame:
    g = pd.read_csv(DATA_DIR / "nba_games.csv")
    g = g[(g["season_type"] == "regular") & ~g["neutral_site"] & (g["half_margin_home"] != 0)].copy()
    g["m"] = g["half_margin_home"].astype(int)
    g["behind"] = (g["m"] < 0).astype(int)
    g["win"] = 100 * g["home_win"]
    g["second_half_margin"] = g["final_margin_home"] - g["m"]  # includes overtime
    g["tpa"] = g["home_tpa"] + g["away_tpa"]
    g["fga"] = g["home_fga"] + g["away_fga"]
    g["short_line"] = g["season"].between(*SHORT_LINE).astype(int)
    for key, (_, first, last) in ERAS.items():
        g.loc[g["season"].between(first, last), "era"] = key
    return g


def by_era(g: pd.DataFrame) -> dict:
    return {label: g[g["era"] == key] for key, (label, _, _) in ERAS.items()}


def save(df: pd.DataFrame, name: str) -> None:
    # 6 decimals in the CSV, so the tables round only once
    print(f"=== {name} ===\n{df.round(3).to_string()}\n")
    df.round(6).to_csv(OUT_DIR / f"{name}.csv")


# Estimators

def rd(df: pd.DataFrame, y: str = "win", bw: int = BW, spec: str = "linear") -> dict:
    """Main estimator: the jump at m = 0 is the coefficient on `behind`."""
    d = df[df["m"].abs() <= bw]
    fit = smf.ols(SPECS[spec].format(y=y), data=d).fit(cov_type="HC1")
    b, se = fit.params["behind"], fit.bse["behind"]
    return {"coef": b, "se": se, "ci_lo": b - 1.96 * se, "ci_hi": b + 1.96 * se,
            "p": fit.pvalues["behind"], "n": int(fit.nobs)}


def rd_robust(df: pd.DataFrame, y: str = "win") -> dict:
    """
    Local linear RD with a triangular kernel, MSE-optimal bandwidth and robust bias-corrected
    CI and p-value (Calonico, Cattaneo & Titiunik 2014). The running variable is -m, so the
    estimate is the jump from just ahead to just behind, as in rd(). The margin is in whole
    points, hence the mass-point adjustment.
    """
    with contextlib.redirect_stdout(io.StringIO()):  # rdrobust prints a mass-points notice
        r = rdrobust(df[y].to_numpy(float), -df["m"].to_numpy(float), c=0, p=1, masspoints="adjust")
    return {"coef": r.coef.iloc[0, 0], "se": r.se.iloc[0, 0], "ci_lo": r.ci.iloc[2, 0],
            "ci_hi": r.ci.iloc[2, 1], "p": r.pv.iloc[2, 0], "n": int(sum(r.N_h)), "bw": r.bws.iloc[0, 0]}


def difference(a: dict, b: dict) -> dict:
    """z-test for a minus b, two estimates from separate samples: SE = sqrt(se_a² + se_b²)."""
    coef, se = a["coef"] - b["coef"], np.sqrt(a["se"] ** 2 + b["se"] ** 2)
    return {"coef": coef, "se": se, "p": 2 * norm.sf(abs(coef / se)), "n": a["n"] + b["n"]}


# Tables

def environment(g: pd.DataFrame, by: str) -> pd.DataFrame:
    """Games, home win %, three-point attempts as % of field-goal attempts (games with shot
    data), and the SD of the second-half margin in close games (|m| <= BW)."""
    close = g[g["m"].abs() <= BW]
    grouped = g.groupby(by)
    return pd.DataFrame({
        "games": grouped.size(),
        "home_win_pct": grouped["win"].mean(),
        "3pa_share_pct": 100 * grouped["tpa"].sum() / grouped["fga"].sum(),
        "sd_second_half_margin": close.groupby(by)["second_half_margin"].std(),
    })


def table1_eras(g: pd.DataFrame) -> pd.DataFrame:
    out = environment(g, "era").loc[list(ERAS)]
    out.index = [label for label, _, _ in ERAS.values()]
    return out.rename_axis("era")


def table2_down1_up1(g: pd.DataFrame) -> pd.DataFrame:
    """Raw home win % when down 1 vs up 1 at halftime. share_at_minus_1 checks that games
    are not sorted to one side of the cutoff."""
    rows = []
    for label, e in by_era(g).items():
        down1, up1 = e.loc[e["m"] == -1, "win"], e.loc[e["m"] == 1, "win"]
        rows.append({"era": label, "win_pct_down_1": down1.mean(), "win_pct_up_1": up1.mean(),
                     "difference_pp": down1.mean() - up1.mean(),
                     "n_down_1": len(down1), "n_up_1": len(up1),
                     "share_at_minus_1": len(down1) / (len(down1) + len(up1))})
    return pd.DataFrame(rows).set_index("era")


def table3a_rd_by_era(g: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame([{"era": label, **rd(e)} for label, e in by_era(g).items()]).set_index("era")
    out["mde"] = 2.8 * out["se"]  # minimum detectable effect: 80% power, 5% level
    return out


def table3b_era_differences(g: pd.DataFrame) -> pd.DataFrame:
    jump = {key: rd(g[g["era"] == key]) for key in ERAS}
    tests = {"post minus B&P": difference(jump["post"], jump["bp"]),
             "3pt minus B&P": difference(jump["3pt"], jump["bp"]),
             "3pt minus post": difference(jump["3pt"], jump["post"])}
    return pd.DataFrame([{"comparison": k, "coef": r["coef"], "se": r["se"], "p": r["p"]}
                         for k, r in tests.items()]).set_index("comparison")


def table4_mechanism(g: pd.DataFrame) -> pd.DataFrame:
    """
    Effort: the jump in the second-half points margin (B&P's mechanism).
    Noise:  the value of one halftime point, i.e. how much one more point of halftime margin
            raises P(win): the average of the two slopes in the main regression, m + 0.5 * behind:m.
            Noisier second halves make each point worth less.
    """
    rows = []
    for label, e in by_era(g).items():
        effort = rd(e, y="second_half_margin")
        fit = smf.ols(SPECS["linear"].format(y="win"), data=e[e["m"].abs() <= BW]).fit(cov_type="HC1")
        value = fit.t_test("m + 0.5 * behind:m = 0")
        rows.append({"era": label,
                     "points_jump": effort["coef"], "points_se": effort["se"], "points_p": effort["p"],
                     "point_value": float(value.effect[0]), "point_value_se": float(value.sd[0][0]),
                     "point_value_p": float(value.pvalue), "n": effort["n"]})
    out = pd.DataFrame(rows).set_index("era")
    out["point_value_vs_bp_pct"] = 100 * out["point_value"] / out["point_value"].iloc[0]
    return out


def table5_short_line(g: pd.DataFrame) -> pd.DataFrame:
    """B&P era only: short-line vs regular-line seasons, for winning and the second-half margin."""
    bp = g[g["era"] == "bp"]
    rows = []
    for y in ("win", "second_half_margin"):
        regular, short = rd(bp[bp["short_line"] == 0], y=y), rd(bp[bp["short_line"] == 1], y=y)
        tests = {"regular-line seasons": regular, "short-line seasons": short,
                 "short minus regular": difference(short, regular)}
        rows += [{"outcome": y, "estimate": k, "coef": r["coef"], "se": r["se"], "p": r["p"], "n": r["n"]}
                 for k, r in tests.items()]
    return pd.DataFrame(rows).set_index(["outcome", "estimate"])


def table6_robustness(g: pd.DataFrame) -> pd.DataFrame:
    checks = {"linear ±5": {"bw": 5}, "linear ±10 (main)": {}, "linear ±15": {"bw": 15},
              "cubic ±10": {"spec": "cubic"}}
    rows = []
    for label, e in by_era(g).items():
        for check, kw in checks.items():
            rows.append({"era": label, "check": check, **rd(e, **kw), "bw": kw.get("bw", BW)})
        rows.append({"era": label, "check": "rdrobust", **rd_robust(e)})
    three = g[g["era"] == "3pt"]
    for check, s in {"without COVID seasons": three[~three["season"].isin(COVID)],
                     "2014/15-2018/19 only": three[three["season"] <= 2019],
                     "2019/20-2024/25 only": three[three["season"] >= 2020]}.items():
        rows.append({"era": ERAS["3pt"][0], "check": check, **rd(s), "bw": BW})
    return pd.DataFrame(rows).set_index(["era", "check"])


def tableA1_seasons(g: pd.DataFrame) -> pd.DataFrame:
    """Table 1 columns and the main RD estimate, season by season."""
    jumps = pd.DataFrame([{"season": s, **rd(e)} for s, e in g.groupby("season")]).set_index("season")
    return environment(g, "season").join(g.groupby("season")["era"].first()).join(jumps)


# Figures: 16 cm wide (A4 text width) so fonts print at their true size; no titles, the
# captions are in the paper. Each era keeps its colour and marker in every figure.

INK, MUTED = "#0b0b0b", "#52514e"
ERA_STYLE = {"bp": ("#2a78d6", "o"), "post": ("#eb6834", "s"), "3pt": ("#1baf7a", "^")}
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
    "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
    "figure.facecolor": "white", "savefig.dpi": 300, "savefig.bbox": "tight",
})


def era_name(label: str) -> tuple[str, str]:
    """'B&P sample 1993/94-2008/09' -> ('B&P sample', '1993/94–2008/09')."""
    name, seasons = label.rsplit(" ", 1)
    return name, seasons.replace("-", "–")


def figure2_rd_by_era(g: pd.DataFrame):
    """Figure 2. Dots are home win rates by halftime margin; lines are the linear fits on each side
    of the cutoff (main specification, Table 3)."""
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.5), sharey=True)
    for ax, (key, (label, _, _)) in zip(axes, ERAS.items()):
        color, marker = ERA_STYLE[key]
        e = g[(g["era"] == key) & (g["m"].abs() <= BW)]
        means = e.groupby("m")["win"].mean()
        ax.plot(means.index, means, marker, color=color, ms=4, ls="none")
        fit = smf.ols(SPECS["linear"].format(y="win"), data=e).fit(cov_type="HC1")
        for xs in (np.arange(-BW, 0.01, 0.5), np.arange(0, BW + 0.01, 0.5)):
            ys = fit.predict(pd.DataFrame({"m": xs, "behind": int(xs[0] < 0)}))
            ax.plot(xs, ys, color=color, lw=1.2)
        ax.axvline(0, color=MUTED, lw=0.6, ls="--")
        name, seasons = era_name(label)
        ax.set_title(f"{name}\n{seasons}", color=INK)
        jump = f"{fit.params['behind']:.1f}".replace("-", "−")
        ax.text(0.04, 0.96, f"Jump: {jump} pp\n(SE {fit.bse['behind']:.1f})", transform=ax.transAxes,
                va="top", fontsize=8, color=INK)
        ax.set_xticks([-10, -5, 0, 5, 10])
    axes[0].set_ylabel("Home team wins (%)")
    fig.supxlabel("Home halftime margin (points)", fontsize=9, y=0.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure2_rd_by_era.png")
    plt.close(fig)


def figure3_jump_vs_3pa_share(seasons: pd.DataFrame):
    """Figure 3. One point per season: the jump in P(win) at a halftime margin of 0 (main
    specification, Table A1), with 95% confidence intervals. Circled points are the short-line
    seasons (1994/95–1996/97). The dashed line is an unweighted linear fit of the season
    estimates on the three-point share."""
    x = seasons["3pa_share_pct"]
    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    for key, (label, first, last) in ERAS.items():
        color, marker = ERA_STYLE[key]
        s = seasons.loc[first:last]
        ax.errorbar(s["3pa_share_pct"], s["coef"], yerr=1.96 * s["se"], fmt=marker, color=color,
                    ms=4, elinewidth=0.8, capsize=0, alpha=0.9, label=" ".join(era_name(label)))
    short = seasons.loc[SHORT_LINE[0]:SHORT_LINE[1]]
    ax.plot(short["3pa_share_pct"], short["coef"], "o", ms=11, mfc="none", mec=INK, mew=0.8,
            label="Short three-point line, 1994/95–1996/97")
    slope = np.polyfit(x, seasons["coef"], 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, np.polyval(slope, xs), color=MUTED, ls="--", lw=1, label="Linear fit")
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xlabel("Three-point attempts (% of field-goal attempts), season")
    ax.set_ylabel("Jump in P(win) at margin 0 (pp)")
    handles, labels = ax.get_legend_handles_labels()
    order = [2, 3, 4, 0, 1]  # eras in the first column, then short line and fit
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure3_jump_vs_3pa_share.png")
    plt.close(fig)


def figureA1_seasons(seasons: pd.DataFrame, era_jumps: pd.DataFrame):
    """Figure A1. (a) Each season's jump with the era estimates, (b) three-point share, (c) SD of
    the second-half margin. Shaded: short-line seasons; dashed: era boundaries."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6.3, 6.2), sharex=True, height_ratios=[2.2, 1, 1])
    for key, (label, first, last) in ERAS.items():
        color, marker = ERA_STYLE[key]
        s = seasons.loc[first:last]
        ax1.errorbar(s.index, s["coef"], yerr=1.96 * s["se"], fmt=marker, color=color, ms=4,
                     elinewidth=0.8, capsize=0, label=" ".join(era_name(label)))
        ax1.hlines(era_jumps.loc[label, "coef"], first - 0.4, last + 0.4, color=color, lw=2)
    ax1.axhline(0, color=INK, lw=0.6)
    ax1.text(SHORT_LINE[0] - 0.4, 0.97, "Short\nline", transform=ax1.get_xaxis_transform(),
             va="top", fontsize=7, color=MUTED)
    ax1.set_ylabel("Jump in P(win) at\nmargin 0 (pp)")
    for ax, col, ylabel in ((ax2, "3pa_share_pct", "Three-point attempts\n(% of FGA)"),
                            (ax3, "sd_second_half_margin", "SD of second-half\nmargin (points)")):
        ax.plot(seasons.index, seasons[col], color=MUTED, lw=1.2, marker="o", ms=2.5)
        ax.set_ylabel(ylabel)
    for ax, letter in zip((ax1, ax2, ax3), "abc"):
        for _, first, _ in list(ERAS.values())[1:]:
            ax.axvline(first - 0.5, color=MUTED, lw=0.6, ls="--")
        ax.axvspan(SHORT_LINE[0] - 0.5, SHORT_LINE[1] + 0.5, color="#000000", alpha=0.07, lw=0)
        ax.set_title(f"({letter})", loc="left", color=INK)
    ticks = list(range(1995, 2026, 5))
    ax3.set_xticks(ticks, [f"{t - 1}/{str(t)[-2:]}" for t in ticks])
    ax3.set_xlabel("Season")
    fig.tight_layout()
    fig.legend(*ax1.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=3, handletextpad=0.3)
    fig.savefig(OUT_DIR / "figureA1_seasons.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    g = load_games()
    print(f"{len(g)} games in the analysis sample\n")

    era_jumps = table3a_rd_by_era(g)
    seasons = tableA1_seasons(g)

    save(table1_eras(g), "table1_eras")
    save(table2_down1_up1(g), "table2_down1_up1")
    save(era_jumps, "table3a_rd_by_era")
    save(table3b_era_differences(g), "table3b_era_differences")
    save(table4_mechanism(g), "table4_mechanism")
    save(table5_short_line(g), "table5_short_line")
    save(table6_robustness(g), "table6_robustness")
    save(seasons, "tableA1_seasons")

    figure2_rd_by_era(g)
    figure3_jump_vs_3pa_share(seasons)
    figureA1_seasons(seasons, era_jumps)
    write_tables(OUT_DIR)
    print(f"Tables and figures saved to {OUT_DIR}")
