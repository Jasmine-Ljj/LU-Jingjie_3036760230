"""
03_did.py — Card & Krueger (1994) replication: Table 4 (DiD regressions).

Reproduces the paper's Table 4, which regresses the change in full-time-equivalent
(FTE) employment on the treatment.  The paper estimates five models:

    (i)   dFTE ~ NJ dummy
    (ii)  dFTE ~ NJ dummy + chain & ownership controls
    (iii) dFTE ~ wage GAP
    (iv)  dFTE ~ wage GAP + chain & ownership controls
    (v)   dFTE ~ wage GAP + chain & ownership + region controls

where:
  * dFTE = FTE(wave 2) - FTE(wave 1);  FTE = full-time + managers + 0.5*part-time
  * NJ dummy = 1 for New Jersey stores
  * GAP = proportional wage increase needed to reach the new $5.05 minimum:
          GAP = max(0, (5.05 - WAGE_ST)/WAGE_ST) for NJ stores, 0 for PA stores
  * chain & ownership = three chain dummies + company-owned dummy
  * region = dummies for two NJ regions and two eastern-PA regions

Standard errors are ordinary (homoskedastic) OLS standard errors, matching the
convention used in the original paper (Card-Krueger report OLS SEs).  Robust SEs
are ~20% larger but leave the conclusions unchanged; they are reported as a
robustness check in the extension (04_extension.py).  Sample: stores with valid
FTE in both waves and a valid wave-1 starting wage (needed for GAP).  This gives
365 stores; the paper reports 357 — a minor difference attributable to the public
replication file vs. the original survey data (documented in the report).

Also outputs a "first stage" check that the minimum wage raised NJ starting wages.

Run from the project root:    python Code/03_did.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
WIDE = ROOT / "Data" / "processed" / "ck_wide.csv"
TBL_DIR = ROOT / "Output" / "table"

NEW_MINIMUM = 5.05

# Models for Table 4: (label, formula, treatment variable name)
MODELS = [
    ("(i) NJ dummy", "d_fte ~ STATE", "STATE"),
    ("(ii) NJ dummy + chain & ownership", "d_fte ~ STATE + C(CHAIN) + CO_OWNED", "STATE"),
    ("(iii) wage GAP", "d_fte ~ GAP", "GAP"),
    ("(iv) wage GAP + chain & ownership", "d_fte ~ GAP + C(CHAIN) + CO_OWNED", "GAP"),
    ("(v) wage GAP + chain & ownership + region",
     "d_fte ~ GAP + C(CHAIN) + CO_OWNED + SOUTHJ + CENTRALJ + PA1 + PA2", "GAP"),
]


def load() -> pd.DataFrame:
    df = pd.read_csv(WIDE)
    # Wage-gap treatment intensity: proportional raise needed to reach $5.05.
    df["GAP"] = np.where(
        df["STATE"] == 1,
        np.maximum(0.0, (NEW_MINIMUM - df["WAGE_ST"]) / df["WAGE_ST"]),
        0.0,
    )
    return df


def fit(model: tuple, data: pd.DataFrame) -> dict:
    label, formula, treat = model
    res = smf.ols(formula, data=data).fit()  # ordinary OLS SEs, as in the paper
    return {
        "model": label,
        "treatment": treat,
        "coef": res.params[treat],
        "se": res.bse[treat],
        "tstat": res.params[treat] / res.bse[treat],
        "nobs": int(res.nobs),
    }


def table4(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict to the Table-4 sample and estimate models (i)-(v)."""
    s = df.dropna(subset=["fte1", "fte2", "WAGE_ST"]).copy()
    rows = [fit(m, s) for m in MODELS]
    out = pd.DataFrame(rows)
    return out, s


def first_stage(df: pd.DataFrame) -> pd.DataFrame:
    """d_wage on NJ: did the minimum wage raise NJ starting wages?"""
    s = df.dropna(subset=["d_wage"]).copy()
    res = smf.ols("d_wage ~ STATE", data=s).fit()  # ordinary OLS SEs
    return pd.DataFrame([{
        "model": "d_wage ~ NJ dummy",
        "coef": res.params["STATE"],
        "se": res.bse["STATE"],
        "tstat": res.params["STATE"] / res.bse["STATE"],
        "nobs": int(res.nobs),
    }])


def main() -> None:
    TBL_DIR.mkdir(parents=True, exist_ok=True)
    df = load()

    t4, s = table4(df)
    t4 = t4.round(3)
    t4.to_csv(TBL_DIR / "table4_did.csv", index=False)
    print("=== Table 4: DiD estimates of the minimum wage on FTE employment ===")
    print(f"Sample: {len(s)} stores;  dFTE mean = {s['d_fte'].mean():.3f}, "
          f"SD = {s['d_fte'].std():.3f}  (paper: n=357, mean=-0.237, SD=8.825)")
    print(f"Mean GAP among NJ stores = {s.loc[s['STATE']==1, 'GAP'].mean():.3f}  (paper: 0.11)")
    print(t4.to_string(index=False))

    fs = first_stage(df).round(3)
    fs.to_csv(TBL_DIR / "first_stage.csv", index=False)
    print("\n=== First stage: effect on starting wage (validation) ===")
    print(fs.to_string(index=False))


if __name__ == "__main__":
    main()
