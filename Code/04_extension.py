"""
04_extension.py — independent extension: heterogeneity by ownership type.

Economic question: does the employment response to the New Jersey minimum-wage
increase differ between company-owned and franchised restaurants?  Franchisees
are independent small-business owners with thinner margins, so they may respond
more strongly to a cost shock than company-owned outlets of large chains.

Design (difference-in-difference-in-differences, DDD):
    dFTE_i = a + d*NJ_i + t*CO_OWNED_i + b*(NJ_i x CO_OWNED_i) + controls + e_i
where
    d      = DiD effect for franchised stores (CO_OWNED = 0)
    d + b  = DiD effect for company-owned stores (CO_OWNED = 1)
    b      = heterogeneity (company-owned minus franchised)

We also run a wage "first stage" DDD to check that the minimum wage bit
similarly across ownership types, and a border-subsample robustness check that
restricts the control group to eastern-PA stores (PA1, the Philadelphia suburbs).

Reads Data/processed/ck_wide.csv; writes Output/table/extension_*.csv.
Run from the project root:    python Code/04_extension.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
WIDE = ROOT / "Data" / "processed" / "ck_wide.csv"
TBL_DIR = ROOT / "Output" / "table"
FIG_DIR = ROOT / "Output" / "figure"

# Palette (CVD-safe categorical slots from the data-viz reference instance)
C_FRANCHISE = "#2a78d6"   # slot 1 blue   -> franchised
C_CO_OWNED = "#eb6834"    # slot 2 orange -> company-owned
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"


def load() -> pd.DataFrame:
    return pd.read_csv(WIDE)


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
        "franchised_did": (b_state, se("STATE")),
        "co_owned_did": (co_owned_did, se("STATE", "STATE:CO_OWNED")),
        "heterogeneity": (b_inter, se("STATE:CO_OWNED")),
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
        fd, cd, h = r["franchised_did"], r["co_owned_did"], r["heterogeneity"]
        print(f"  {r['spec']:22s} ({r['outcome']})")
        print(f"    franchised DiD    = {fd[0]:6.3f}  (SE {fd[1]:.3f}, t = {fd[0]/fd[1]:5.2f})")
        print(f"    company-owned DiD = {cd[0]:6.3f}  (SE {cd[1]:.3f}, t = {cd[0]/cd[1]:5.2f})")
        print(f"    heterogeneity (b) = {h[0]:6.3f}  (SE {h[1]:.3f}, t = {h[0]/h[1]:5.2f})")


def coefficient_plot(full, border, out: Path) -> Path:
    """Coefficient plot: ownership DiD estimates with 95% CIs, two panels."""
    def estimates(tbl):
        row = tbl[tbl["spec"] == "DDD"].iloc[0]
        return {
            "Franchised": row["franchised_did"],
            "Company-owned": row["co_owned_did"],
        }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
    panels = [("Full sample (NJ vs PA)", estimates(full)),
              ("Border sample (NJ vs PA1)", estimates(border))]

    for ax, (title, est) in zip(axes, panels):
        ypos = {"Franchised": 1, "Company-owned": 0}
        for name, (coef, se) in est.items():
            color = C_FRANCHISE if name == "Franchised" else C_CO_OWNED
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


def main() -> None:
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    df = load()

    # Employment sample: balanced stores with valid FTE in both waves.
    emp = df[df["in_main_sample"]].copy()
    # Wage sample: stores with valid starting wage in both waves.
    wage = df.dropna(subset=["d_wage"]).copy()

    # --- Part 1: ownership heterogeneity in the employment response ---
    t_emp = ownership_table(emp, "d_fte", "dFTE")
    t_emp.to_csv(TBL_DIR / "extension_ownership_ddd.csv", index=False)
    print_ddd(t_emp, "扩展1：直营 vs 加盟 的就业反应差异 (DDD)")

    # --- Part 2: compliance (wage first stage) by ownership ---
    t_wage = ownership_table(wage, "d_wage", "dStarting wage")
    t_wage.to_csv(TBL_DIR / "extension_ownership_wage.csv", index=False)
    print_ddd(t_wage, "验证：最低工资对起薪的冲击是否因企业类型而异")

    # --- Part 3: border-subsample robustness (NJ vs PA1 only) ---
    border = emp[(emp["STATE"] == 1) | (emp["PA1"] == 1)].copy()
    print(f"\n=== 扩展2：边界子样本稳健性 (NJ vs PA1, n={len(border)}) ===")
    base = smf.ols("d_fte ~ STATE", data=border).fit()
    print(f"  主 DiD (ΔFTE ~ NJ): {base.params['STATE']:.3f}  (SE {base.bse['STATE']:.3f}, "
          f"t = {base.params['STATE']/base.bse['STATE']:.2f})")
    t_border = ownership_table(border, "d_fte", "dFTE")
    t_border.to_csv(TBL_DIR / "extension_border_robustness.csv", index=False)
    print_ddd(t_border, "边界样本下的 DDD")

    fig = coefficient_plot(t_emp, t_border, FIG_DIR / "extension_ownership_did.png")
    print(f"\nWrote figure: {fig}")


if __name__ == "__main__":
    main()
