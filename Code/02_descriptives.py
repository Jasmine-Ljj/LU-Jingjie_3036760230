"""
02_descriptives.py — Card & Krueger (1994) replication: descriptive statistics.

Reproduces:
  * Table 2  (store-type distribution + wave-1/wave-2 means, by state)
  * Table 3  (mean FTE before/after the minimum-wage increase + DiD)
  * Figure 1 (distribution of the change in FTE employment, NJ vs PA)

Reads Data/processed/ck_wide.csv (produced by 01_clean.py) and writes tables to
Output/table/ and the figure to Output/figure/.

Run from the project root:    python Code/02_descriptives.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WIDE = ROOT / "Data" / "processed" / "ck_wide.csv"
TBL_DIR = ROOT / "Output" / "table"
FIG_DIR = ROOT / "Output" / "figure"

STATE = {1: "NJ", 0: "PA"}

# Palette (from the data-viz reference instance; CVD-safe categorical slots)
C_NJ = "#2a78d6"   # slot 1 blue  -> New Jersey (treatment)
C_PA = "#eb6834"   # slot 2 orange -> Pennsylvania (control)
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"


def load() -> pd.DataFrame:
    df = pd.read_csv(WIDE)
    # "price of a full meal" = soda + fries + entree (paper's definition)
    df["meal1"] = df["PSODA"] + df["PFRY"] + df["PENTREE"]
    df["meal2"] = df["PSODA2"] + df["PFRY2"] + df["PENTREE2"]
    # "% full-time employees" = store-level FT / FTE (paper's definition)
    df["pct_ft1"] = df["EMPFT"] / df["fte1"] * 100
    df["pct_ft2"] = df["EMPFT2"] / df["fte2"] * 100
    return df


def mean_by_state(df, col):
    """Mean of `col` for NJ and PA, using all available observations."""
    return {st: df.loc[df["STATE"] == s, col].mean() for s, st in STATE.items()}


def share_by_state(df, mask):
    """Share of stores satisfying `mask` (%), for NJ and PA."""
    return {
        st: df.loc[(df["STATE"] == s) & mask, "SHEET"].size
        / df.loc[df["STATE"] == s, "SHEET"].size * 100
        for s, st in STATE.items()
    }


def table2_panelA(df) -> pd.DataFrame:
    """Distribution of store types, by state (percentages)."""
    rows = {
        "Burger King": share_by_state(df, df["CHAIN"] == 1),
        "KFC": share_by_state(df, df["CHAIN"] == 2),
        "Roy Rogers": share_by_state(df, df["CHAIN"] == 3),
        "Wendy's": share_by_state(df, df["CHAIN"] == 4),
        "Company-owned": share_by_state(df, df["CO_OWNED"] == 1),
    }
    out = pd.DataFrame(rows, index=["NJ", "PA"]).T.round(1)
    out.index.name = "Store type"
    return out


def table2_means(df) -> pd.DataFrame:
    """Wave-1 and wave-2 means of key variables, by state."""
    rows = {}
    # Wave 1
    rows["FTE employment (wave 1)"] = mean_by_state(df, "fte1")
    rows["% full-time employees (wave 1)"] = mean_by_state(df, "pct_ft1")
    rows["Starting wage (wave 1)"] = mean_by_state(df, "WAGE_ST")
    rows["% wage = $4.25 (wave 1)"] = share_by_state(df, df["WAGE_ST"] == 4.25)
    rows["Price of full meal (wave 1)"] = mean_by_state(df, "meal1")
    rows["Hours open, weekday (wave 1)"] = mean_by_state(df, "HRSOPEN")
    rows["Recruiting bonus % (wave 1)"] = share_by_state(df, df["BONUS"] == 1)
    # Wave 2
    rows["FTE employment (wave 2)"] = mean_by_state(df, "fte2")
    rows["% full-time employees (wave 2)"] = mean_by_state(df, "pct_ft2")
    rows["Starting wage (wave 2)"] = mean_by_state(df, "WAGE_ST2")
    rows["% wage = $4.25 (wave 2)"] = share_by_state(df, df["WAGE_ST2"] == 4.25)

    out = pd.DataFrame(rows, index=["NJ", "PA"]).T.round(2)
    out.index.name = "Variable"
    return out


def table3_did(df) -> pd.DataFrame:
    """Mean FTE before/after the increase, by state, and the DiD estimate."""
    before = mean_by_state(df, "fte1")
    after = mean_by_state(df, "fte2")
    change = {st: after[st] - before[st] for st in STATE.values()}

    out = pd.DataFrame(
        {
            "PA": [before["PA"], after["PA"], change["PA"]],
            "NJ": [before["NJ"], after["NJ"], change["NJ"]],
            "NJ - PA": [before["NJ"] - before["PA"],
                        after["NJ"] - after["PA"],
                        change["NJ"] - change["PA"]],
        },
        index=["FTE before", "FTE after", "Change in FTE"],
    ).round(2)
    out.index.name = "FTE employment"
    return out


def figure1(df) -> Path:
    """Distribution of the change in FTE employment (wave2 - wave1), NJ vs PA."""
    bal = df[df["in_main_sample"]].copy()
    nj = bal.loc[bal["STATE"] == 1, "d_fte"]
    pa = bal.loc[bal["STATE"] == 0, "d_fte"]

    bins = np.arange(-42.5, 47.5, 5)  # 5-FTE bins

    fig, ax = plt.subplots(figsize=(8, 4.8))
    # Share of stores per bin (normalised so NJ and PA are comparable)
    ax.hist(pa, bins=bins, weights=np.ones_like(pa) / len(pa) * 100,
            alpha=0.55, color=C_PA, label="Pennsylvania (control)", edgecolor="white")
    ax.hist(nj, bins=bins, weights=np.ones_like(nj) / len(nj) * 100,
            alpha=0.55, color=C_NJ, label="New Jersey (treatment)", edgecolor="white")
    ax.axvline(0, color=INK, lw=1, ls="--", alpha=0.6)

    ax.set_xlabel("Change in full-time-equivalent employment (wave 2 − wave 1)")
    ax.set_ylabel("Percent of stores")
    ax.set_title("Distribution of Employment Change, NJ vs PA")
    ax.legend(frameon=False, loc="upper right")

    # Recessive chrome
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, lw=0.8)

    fig.tight_layout()
    out = FIG_DIR / "figure1.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def save(df: pd.DataFrame, name: str) -> None:
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TBL_DIR / name)
    print(f"\n=== {name} ===")
    print(df.to_string(float_format=lambda x: f"{x:.2f}"))


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load()

    save(table2_panelA(df), "table2_panelA_store_types.csv")
    save(table2_means(df), "table2_means.csv")
    save(table3_did(df), "table3_did.csv")

    fig = figure1(df)
    print(f"\nWrote figure: {fig}")


if __name__ == "__main__":
    main()
