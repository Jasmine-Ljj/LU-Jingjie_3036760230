"""
run_all.py — one-command reproducibility.

The single entry point that regenerates EVERY table and figure of the project
from the raw survey data.  Run it with:

    python Code/run_all.py

from the project root.  It performs the whole pipeline in one go:

    1. Clean        — runs Code/01_clean.py (constructs FTE, sample -> Data/processed/)
    2. Descriptives — Table 2 (store types, wave-1 & wave-2 means), Table 3, Figure 1
    3. DiD          — Table 4 (five models) + the wage first stage
    4. Extension    — ownership-heterogeneity DDD + wage first stage + border robustness

and then verifies that every expected output file was written.  No manual
modification of data or code is needed — just place Data/raw/public.csv and run
this one command.  The only helper is Code/01_clean.py (data cleaning).

Figure 1 reproduces the paper's "Distribution of Starting Wage Rates" (two
panels, February and November 1992, New Jersey vs Pennsylvania).
"""

from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

CODE_DIR = Path(__file__).resolve().parent    # Code/
PROJECT = CODE_DIR.parent                     # project root
WIDE = PROJECT / "Data" / "processed" / "ck_wide.csv"
TBL_DIR = PROJECT / "Output" / "table"
FIG_DIR = PROJECT / "Output" / "figure"

STATE = {1: "NJ", 0: "PA"}
NEW_MINIMUM = 5.05
# STATUS2 codes for stores temporarily closed / under renovation in wave 2
# (treated as missing, or set to 0 in Table 3 row 5), per the paper.
TEMP_CLOSED = {2, 4, 5}

# CVD-safe categorical palette (slot 1 blue = NJ, slot 2 orange = PA)
C_NJ = "#2a78d6"
C_PA = "#eb6834"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

# Table 4 models: (label, formula, treatment variable, control variable names)
MODELS = [
    ("(i) NJ dummy", "d_fte ~ STATE", "STATE", []),
    ("(ii) NJ dummy + chain & ownership",
     "d_fte ~ STATE + C(CHAIN) + CO_OWNED", "STATE", ["C(CHAIN)", "CO_OWNED"]),
    ("(iii) wage GAP", "d_fte ~ GAP", "GAP", []),
    ("(iv) wage GAP + chain & ownership",
     "d_fte ~ GAP + C(CHAIN) + CO_OWNED", "GAP", ["C(CHAIN)", "CO_OWNED"]),
    ("(v) wage GAP + chain & ownership + region",
     "d_fte ~ GAP + C(CHAIN) + CO_OWNED + SOUTHJ + CENTRALJ + PA1 + PA2",
     "GAP", ["C(CHAIN)", "CO_OWNED", "SOUTHJ", "CENTRALJ", "PA1", "PA2"]),
]

# Every table and figure the workflow is expected to produce.
EXPECTED_OUTPUTS = [
    "Data/processed/ck_wide.csv",
    "Data/processed/ck_long.csv",
    "Output/table/table2.csv",
    "Output/table/table3_did.csv",
    "Output/figure/figure1.png",
    "Output/table/table4_did.csv",
    "Output/table/first_stage.csv",
    "Output/table/extension_ownership_ddd.csv",
    "Output/table/extension_ownership_wage.csv",
    "Output/table/extension_border_robustness.csv",
    "Output/figure/extension_ownership_did.png",
]


# --------------------------------------------------------------------------
# Step 1 — clean (delegated to Code/01_clean.py)
# --------------------------------------------------------------------------
def step_clean() -> None:
    result = subprocess.run(
        [sys.executable, str(CODE_DIR / "01_clean.py")], cwd=str(PROJECT)
    )
    if result.returncode != 0:
        sys.exit(result.returncode)


def load() -> pd.DataFrame:
    """Read the cleaned wide data and add derived columns (meal price, %FT, GAP)."""
    df = pd.read_csv(WIDE)
    df["meal1"] = df["PSODA"] + df["PFRY"] + df["PENTREE"]
    df["meal2"] = df["PSODA2"] + df["PFRY2"] + df["PENTREE2"]
    df["pct_ft1"] = df["EMPFT"] / df["fte1"] * 100
    df["pct_ft2"] = df["EMPFT2"] / df["fte2"] * 100
    df["GAP"] = np.where(
        df["STATE"] == 1,
        np.maximum(0.0, (NEW_MINIMUM - df["WAGE_ST"]) / df["WAGE_ST"]),
        0.0,
    )
    return df


# --------------------------------------------------------------------------
# Step 2 — descriptives (Table 2, Table 3, Figure 1)
# --------------------------------------------------------------------------
def _t_stat(nj: pd.Series, pa: pd.Series) -> float:
    """Welch t-statistic for equality of means between the NJ and PA groups."""
    nj = nj.dropna()
    pa = pa.dropna()
    m1, m2 = nj.mean(), pa.mean()
    v1, v2 = nj.var(ddof=1), pa.var(ddof=1)
    n1, n2 = len(nj), len(pa)
    se = np.sqrt(v1 / n1 + v2 / n2)
    return (m1 - m2) / se if se > 0 else np.nan


def _by_state(df: pd.DataFrame, series: pd.Series, pct: bool = False):
    """(NJ mean, PA mean, t-stat) for a column or indicator.

    `pct=True` scales a 0/1 indicator by 100 (a percentage-of-stores row).
    """
    s = series.astype(float)
    nj = s[df["STATE"] == 1]
    pa = s[df["STATE"] == 0]
    m1, m2 = nj.mean(), pa.mean()
    t = _t_stat(nj, pa)
    if pct:
        m1, m2 = m1 * 100, m2 * 100
    return m1, m2, t


def table2(df: pd.DataFrame) -> pd.DataFrame:
    """Table 2 — one combined table: store types + wave-1 + wave-2 means.

    Columns are [NJ, PA, t]; the three sections are separated by header rows
    (empty cells) to mirror the paper's single-table layout.
    """
    sections = [
        ("1. Distribution of store types (%)", [
            ("Burger King", df["CHAIN"] == 1, True),
            ("KFC", df["CHAIN"] == 2, True),
            ("Roy Rogers", df["CHAIN"] == 3, True),
            ("Wendy's", df["CHAIN"] == 4, True),
            ("Company-owned", df["CO_OWNED"] == 1, True),
        ]),
        ("2. Means in wave 1 (Feb-Mar 1992)", [
            ("FTE employment", df["fte1"], False),
            ("% full-time employees", df["pct_ft1"], False),
            ("Starting wage", df["WAGE_ST"], False),
            ("% wage = $4.25", df["WAGE_ST"] == 4.25, True),
            ("Price of full meal", df["meal1"], False),
            ("Hours open (weekday)", df["HRSOPEN"], False),
            ("Recruiting bonus %", df["BONUS"], True),
        ]),
        ("3. Means in wave 2 (Nov-Dec 1992)", [
            ("FTE employment", df["fte2"], False),
            ("% full-time employees", df["pct_ft2"], False),
            ("Starting wage", df["WAGE_ST2"], False),
            ("% wage = $4.25", df["WAGE_ST2"] == 4.25, True),
            ("% wage = $5.05", df["WAGE_ST2"] == 5.05, True),
            ("Price of full meal", df["meal2"], False),
            ("Hours open (weekday)", df["HRSOPEN2"], False),
            ("Recruiting bonus %", df["SPECIAL2"], True),
        ]),
    ]
    index, data = [], []
    for section, items in sections:
        index.append(section)
        data.append([np.nan] * 3)
        for label, series, pct in items:
            m1, m2, t = _by_state(df, series, pct=pct)
            index.append(label)
            data.append([m1, m2, t])
    out = pd.DataFrame(data, index=index, columns=["NJ", "PA", "t"])
    out.index.name = "Variable"
    return out


def _group_stats(df: pd.DataFrame, mask: pd.Series) -> list:
    """[before, after, change, balanced-change, temp-closed=0 change] FTE."""
    sub = df[mask]
    before = sub["fte1"].mean()
    after = sub["fte2"].mean()

    bal = sub.dropna(subset=["fte1", "fte2"])
    bal_change = (bal["fte2"] - bal["fte1"]).mean()

    # row 5: wave-2 employment at temporarily-closed stores set to 0
    sub2 = sub.copy()
    sub2.loc[sub2["STATUS2"].isin(TEMP_CLOSED), "fte2"] = 0.0
    bal2 = sub2.dropna(subset=["fte1", "fte2"])
    temp_change = (bal2["fte2"] - bal2["fte1"]).mean()

    return [before, after, after - before, bal_change, temp_change]


def table3(df: pd.DataFrame) -> pd.DataFrame:
    """Table 3 — average employment per store before/after, by state and by
    the wave-1 starting wage within NJ, with the five change rows."""
    g_pa = _group_stats(df, df["STATE"] == 0)
    g_nj = _group_stats(df, df["STATE"] == 1)
    g_425 = _group_stats(df, (df["STATE"] == 1) & (df["WAGE_ST"] == 4.25))
    g_mid = _group_stats(df, (df["STATE"] == 1)
                         & (df["WAGE_ST"] > 4.25) & (df["WAGE_ST"] < 5.00))
    g_high = _group_stats(df, (df["STATE"] == 1) & (df["WAGE_ST"] >= 5.00))

    rows = [
        "FTE before (all obs)",
        "FTE after (all obs)",
        "Change in mean FTE",
        "Change, balanced sample",
        "Change, temp-closed = 0",
    ]
    data = {
        "PA": g_pa,
        "NJ": g_nj,
        "NJ - PA": [a - b for a, b in zip(g_nj, g_pa)],
        "NJ wage = $4.25": g_425,
        "NJ wage $4.26-$4.99": g_mid,
        "NJ wage >= $5.00": g_high,
        "Low - high": [a - b for a, b in zip(g_425, g_high)],
        "Midrange - high": [a - b for a, b in zip(g_mid, g_high)],
    }
    out = pd.DataFrame(data, index=rows).round(2)
    out.index.name = "FTE employment"
    return out


def figure1(df: pd.DataFrame) -> Path:
    """Figure 1 — distribution of starting wage rates (Feb and Nov 1992)."""
    bins = np.arange(4.25, 6.26, 0.25)          # $0.25 wage bins, $4.25-$6.25
    labels = [f"${b:.2f}" for b in bins[:-1]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    panels = [("WAGE_ST", "February 1992 (wave 1)"),
              ("WAGE_ST2", "November 1992 (wave 2)")]

    for ax, (col, title) in zip(axes, panels):
        nj = df.loc[df["STATE"] == 1, col].dropna()
        pa = df.loc[df["STATE"] == 0, col].dropna()
        nj_share = np.histogram(nj, bins=bins)[0] / len(nj) * 100
        pa_share = np.histogram(pa, bins=bins)[0] / len(pa) * 100

        x = np.arange(len(nj_share))
        w = 0.4
        ax.bar(x - w / 2, nj_share, w, color=C_NJ, label="New Jersey")
        ax.bar(x + w / 2, pa_share, w, color=C_PA, label="Pennsylvania")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("Starting wage range ($)")
        ax.set_title(title, color=INK)
        ax.legend(frameon=False, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        ax.tick_params(colors=MUTED, length=0)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID, lw=0.8)

    axes[0].set_ylabel("Percent of stores")
    fig.suptitle("Distribution of Starting Wage Rates", color=INK, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = FIG_DIR / "figure1.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def save(df: pd.DataFrame, name: str) -> None:
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TBL_DIR / name)
    print(f"\n=== {name} ===")
    print(df.round(3).to_string())


# --------------------------------------------------------------------------
# Step 3 — DiD (Table 4 + wage first stage)
# --------------------------------------------------------------------------
def _p_controls(res, controls) -> float:
    """Joint F-test p-value for exclusion of the control variables."""
    if not controls:
        return np.nan
    names = []
    for c in controls:
        for p in res.params.index:
            if p == c or p.startswith(c + "["):
                names.append(p)
    if not names:
        return np.nan
    idx = list(res.params.index)
    R = np.zeros((len(names), len(idx)))
    for i, n in enumerate(names):
        R[i, idx.index(n)] = 1.0
    return float(res.f_test(R).pvalue)


def fit(model: tuple, data: pd.DataFrame) -> dict:
    label, formula, treat, controls = model
    res = smf.ols(formula, data=data).fit()  # ordinary OLS SEs, as in the paper
    return {
        "model": label,
        "coef": res.params[treat],
        "se": res.bse[treat],
        "tstat": res.params[treat] / res.bse[treat],
        "nobs": int(res.nobs),
        "se_regression": float(np.sqrt(res.mse_resid)),
        "p_controls": _p_controls(res, controls),
    }


def table4(df: pd.DataFrame):
    """Restrict to the Table-4 sample and estimate models (i)-(v)."""
    s = df.dropna(subset=["fte1", "fte2", "WAGE_ST"]).copy()
    rows = [fit(m, s) for m in MODELS]
    return pd.DataFrame(rows), s


def first_stage(df: pd.DataFrame) -> pd.DataFrame:
    """d_wage on NJ: did the minimum wage raise NJ starting wages?"""
    s = df.dropna(subset=["d_wage"]).copy()
    res = smf.ols("d_wage ~ STATE", data=s).fit()
    return pd.DataFrame([{
        "model": "d_wage ~ NJ dummy",
        "coef": res.params["STATE"],
        "se": res.bse["STATE"],
        "tstat": res.params["STATE"] / res.bse["STATE"],
        "nobs": int(res.nobs),
    }])


# --------------------------------------------------------------------------
# Step 4 — extension (ownership-heterogeneity DDD + robustness)
# --------------------------------------------------------------------------
def ddd(df: pd.DataFrame, y: str, chain_controls: bool) -> dict:
    """Estimate the ownership DDD for outcome `y`, returning the three DiD terms."""
    formula = f"{y} ~ STATE + CO_OWNED + STATE:CO_OWNED"
    if chain_controls:
        formula += " + C(CHAIN)"
    res = smf.ols(formula, data=df).fit()  # OLS SEs, consistent with the paper

    b_state = res.params["STATE"]
    b_inter = res.params["STATE:CO_OWNED"]
    V = res.cov_params()

    def se(*keys):
        var = sum(V.loc[a, b] for a in keys for b in keys)
        return np.sqrt(var)

    co_owned_did = b_state + b_inter
    return {
        "franchised_coef": float(b_state),
        "franchised_se": float(se("STATE")),
        "co_owned_coef": float(co_owned_did),
        "co_owned_se": float(se("STATE", "STATE:CO_OWNED")),
        "heterogeneity_coef": float(b_inter),
        "heterogeneity_se": float(se("STATE:CO_OWNED")),
        "nobs": int(res.nobs),
    }


def ownership_table(df: pd.DataFrame, y: str, ylabel: str) -> pd.DataFrame:
    rows = []
    for chain in (False, True):
        r = ddd(df, y, chain)
        r.update(spec="DDD" if not chain else "DDD + chain controls", outcome=ylabel)
        rows.append(r)
    return pd.DataFrame(rows)


def print_ddd(df: pd.DataFrame, title: str) -> None:
    print(f"\n=== {title} ===")
    for _, r in df.iterrows():
        fc, fs = r["franchised_coef"], r["franchised_se"]
        cc, cs = r["co_owned_coef"], r["co_owned_se"]
        hc, hs = r["heterogeneity_coef"], r["heterogeneity_se"]
        print(f"  {r['spec']:22s} ({r['outcome']})")
        print(f"    franchised DiD    = {fc:6.3f}  (SE {fs:.3f}, t = {fc/fs:5.2f})")
        print(f"    company-owned DiD = {cc:6.3f}  (SE {cs:.3f}, t = {cc/cs:5.2f})")
        print(f"    heterogeneity (b) = {hc:6.3f}  (SE {hs:.3f}, t = {hc/hs:5.2f})")


def coefficient_plot(full: pd.DataFrame, border: pd.DataFrame, out: Path) -> Path:
    """Coefficient plot: ownership DiD estimates with 95% CIs, two panels."""
    def estimates(tbl):
        row = tbl[tbl["spec"] == "DDD"].iloc[0]
        return {
            "Franchised": (row["franchised_coef"], row["franchised_se"]),
            "Company-owned": (row["co_owned_coef"], row["co_owned_se"]),
        }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
    panels = [("Full sample (NJ vs PA)", estimates(full)),
              ("Border sample (NJ vs PA1)", estimates(border))]

    for ax, (title, est) in zip(axes, panels):
        ypos = {"Franchised": 1, "Company-owned": 0}
        for name, (coef, se) in est.items():
            color = C_NJ if name == "Franchised" else C_PA
            ax.errorbar(coef, ypos[name], xerr=1.96 * se,
                        fmt="o", ms=8, color=color, capsize=5, lw=2)
        ax.axvline(0, color=INK, lw=1, ls="--", alpha=0.6)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Company-owned", "Franchised"])
        ax.set_title(title, color=INK)
        ax.set_xlabel("DiD estimate of minimum wage on FTE employment")
        ax.set_xlim(-8, 14)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        ax.tick_params(colors=MUTED, length=0)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def verify_outputs() -> None:
    """Check that every expected table and figure exists after the run."""
    print("\n" + "=" * 64)
    print("Output check")
    print("=" * 64)
    missing = []
    for rel in EXPECTED_OUTPUTS:
        exists = (PROJECT / rel).exists()
        if not exists:
            missing.append(rel)
        print(f"  [{'OK' if exists else 'MISS'}] {rel}")
    if missing:
        print("\n[run_all] MISSING outputs — the pipeline did not reproduce "
              "everything. Check the step logs above.")
        sys.exit(1)
    print(f"\n[run_all] All {len(EXPECTED_OUTPUTS)} outputs regenerated "
          "successfully.")


def main() -> None:
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 64}\nStep 1/4 — clean\n{'=' * 64}")
    step_clean()
    df = load()

    print(f"\n{'=' * 64}\nStep 2/4 — descriptives (Table 2, Table 3, Figure 1)\n{'=' * 64}")
    save(table2(df).round(2), "table2.csv")
    save(table3(df), "table3_did.csv")
    print(f"\nWrote figure: {figure1(df)}")

    print(f"\n{'=' * 64}\nStep 3/4 — DiD (Table 4 + first stage)\n{'=' * 64}")
    t4, s = table4(df)
    t4 = t4.round(3)
    t4.to_csv(TBL_DIR / "table4_did.csv", index=False)
    print("=== Table 4: DiD estimates of the minimum wage on FTE employment ===")
    print(f"Sample: {len(s)} stores;  dFTE mean = {s['d_fte'].mean():.3f}, "
          f"SD = {s['d_fte'].std():.3f}  (paper: n=357, mean=-0.237, SD=8.825)")
    print(f"Mean GAP among NJ stores = {s.loc[s['STATE'] == 1, 'GAP'].mean():.3f}  "
          f"(paper: 0.11)")
    print(t4.to_string(index=False))

    fs = first_stage(df).round(3)
    fs.to_csv(TBL_DIR / "first_stage.csv", index=False)
    print("\n=== First stage: effect on starting wage (validation) ===")
    print(fs.to_string(index=False))

    print(f"\n{'=' * 64}\nStep 4/4 — extension (ownership heterogeneity + robustness)\n{'=' * 64}")
    emp = df[df["in_main_sample"]].copy()
    wage = df.dropna(subset=["d_wage"]).copy()

    t_emp = ownership_table(emp, "d_fte", "dFTE")
    t_emp.to_csv(TBL_DIR / "extension_ownership_ddd.csv", index=False)
    print_ddd(t_emp, "Extension 1: employment response by ownership type (DDD)")

    t_wage = ownership_table(wage, "d_wage", "dStarting wage")
    t_wage.to_csv(TBL_DIR / "extension_ownership_wage.csv", index=False)
    print_ddd(t_wage, "Validation: did the minimum wage bite differently by ownership type?")

    border = emp[(emp["STATE"] == 1) | (emp["PA1"] == 1)].copy()
    print(f"\n=== Extension 2: border-subsample robustness (NJ vs PA1, n={len(border)}) ===")
    base = smf.ols("d_fte ~ STATE", data=border).fit()
    print(f"  main DiD (ΔFTE ~ NJ): {base.params['STATE']:.3f}  "
          f"(SE {base.bse['STATE']:.3f}, t = {base.params['STATE'] / base.bse['STATE']:.2f})")
    t_border = ownership_table(border, "d_fte", "dFTE")
    t_border.to_csv(TBL_DIR / "extension_border_robustness.csv", index=False)
    print_ddd(t_border, "DDD on the border sample")

    print(f"\nWrote figure: "
          f"{coefficient_plot(t_emp, t_border, FIG_DIR / 'extension_ownership_did.png')}")

    verify_outputs()


if __name__ == "__main__":
    main()
