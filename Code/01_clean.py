"""
01_clean.py — Card & Krueger (1994) replication: data cleaning.

Reads the raw public survey data (Data/raw/public.csv), constructs the key
outcome and covariate variables, defines the analysis samples, and writes
tidy wide and long tables to Data/processed/.

Run from the project root:    python Code/01_clean.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "Data" / "raw" / "public.csv"
PROCESSED = ROOT / "Data" / "processed"

# Card-Krueger codebook labels
CHAIN_LABELS = {1: "Burger King", 2: "KFC", 3: "Roy Rogers", 4: "Wendy's"}
STATE_LABELS = {1: "New Jersey", 0: "Pennsylvania"}

# STATUS2: status of the second (post) interview.
#   1 = interview completed (restaurant still in business)
#   3 = closed permanently (out of business; wave-2 employment recorded as 0)
#   0, 2, 4, 5 = no usable wave-2 data (refused / not located / other)
CLOSED_CODE = 3
IN_BUSINESS_CODE = 1


def load_raw() -> pd.DataFrame:
    """Load the raw survey data. Empty cells are read as NaN."""
    df = pd.read_csv(RAW)
    # Defensive: treat any empty string left over as missing.
    df = df.replace({"": np.nan})
    return df


def add_employment(df: pd.DataFrame) -> pd.DataFrame:
    """Construct employment outcomes for both waves.

    FTE (full-time-equivalent) = full-time + managers + 0.5 * part-time.
    This is the paper's primary employment measure in Table 4.
    """
    for wave, sfx in (("1", ""), ("2", "2")):
        ft = df[f"EMPFT{sfx}"]
        pt = df[f"EMPPT{sfx}"]
        mgr = df[f"NMGRS{sfx}"]
        df[f"fte{wave}"] = ft + mgr + 0.5 * pt           # full-time equivalent
        df[f"tot{wave}"] = ft + mgr + pt                 # headcount (all workers)
    return df


def add_changes(df: pd.DataFrame) -> pd.DataFrame:
    """Wave-2 minus wave-1 differences (used for first-difference DiD)."""
    diff_map = {
        "d_fte": ("fte2", "fte1"),
        "d_tot": ("tot2", "tot1"),
        "d_empft": ("EMPFT2", "EMPFT"),
        "d_emppt": ("EMPPT2", "EMPPT"),
        "d_nmgrs": ("NMGRS2", "NMGRS"),
        "d_wage": ("WAGE_ST2", "WAGE_ST"),
    }
    for out, (post, pre) in diff_map.items():
        df[out] = df[post] - df[pre]
    return df


def add_sample_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Define the main balanced sample.

    Main sample = stores that were either still open (STATUS2==1) or closed
    (STATUS2==3, employment = 0) AND have non-missing FTE in both waves.
    This reproduces the paper's Table 4 FTE estimate of ~2.75.
    """
    df["closed"] = (df["STATUS2"] == CLOSED_CODE).astype(int)
    df["in_main_sample"] = (
        df["STATUS2"].isin([IN_BUSINESS_CODE, CLOSED_CODE])
        & df["fte1"].notna()
        & df["fte2"].notna()
    )
    return df


def build_long_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Stack the two waves into a long panel (one row per store-wave).

    Used by 03_did.py for the level regression with store fixed effects.
    """
    wave1_vars = {
        "EMPFT": "empft", "EMPPT": "emppt", "NMGRS": "nmgrs",
        "fte1": "fte", "tot1": "tot", "WAGE_ST": "wage_st",
        "PSODA": "psoda", "PFRY": "pfry", "PENTREE": "pentree",
        "HRSOPEN": "hrsopen", "NREGS": "nregs",
    }
    wave2_vars = {
        "EMPFT2": "empft", "EMPPT2": "emppt", "NMGRS2": "nmgrs",
        "fte2": "fte", "tot2": "tot", "WAGE_ST2": "wage_st",
        "PSODA2": "psoda", "PFRY2": "pfry", "PENTREE2": "pentree",
        "HRSOPEN2": "hrsopen", "NREGS2": "nregs",
    }
    id_cols = ["SHEET", "CHAIN", "CO_OWNED", "STATE", "SOUTHJ", "CENTRALJ",
               "NORTHJ", "PA1", "PA2", "SHORE", "closed", "in_main_sample"]

    frames = []
    for wave, var_map in ((1, wave1_vars), (2, wave2_vars)):
        sub = df[id_cols].copy()
        for src, dst in var_map.items():
            sub[dst] = df[src]
        sub["wave"] = wave
        sub["post"] = int(wave == 2)
        frames.append(sub)

    long = pd.concat(frames, ignore_index=True).sort_values(["SHEET", "wave"])
    return long


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)

    df = load_raw()
    df = add_employment(df)
    df = add_changes(df)
    df = add_sample_flags(df)

    # Human-readable labels (kept alongside codes for the report)
    df["chain_name"] = df["CHAIN"].map(CHAIN_LABELS)
    df["state_name"] = df["STATE"].map(STATE_LABELS)

    long = build_long_panel(df)

    # Write processed data
    df.to_csv(PROCESSED / "ck_wide.csv", index=False)
    long.to_csv(PROCESSED / "ck_long.csv", index=False)

    # Console summary for a quick sanity check
    main = df[df["in_main_sample"]]
    nj, pa = main[main["STATE"] == 1], main[main["STATE"] == 0]
    did = (nj["fte2"].mean() - nj["fte1"].mean()) - (pa["fte2"].mean() - pa["fte1"].mean())
    print(f"Raw rows: {len(df)}")
    print(f"Main balanced sample: {len(main)} stores  (NJ={len(nj)}, PA={len(pa)})")
    print(f"Naive FTE DiD: {did:.4f}  (paper reports 2.75)")
    print(f"Wrote: {PROCESSED / 'ck_wide.csv'} and {PROCESSED / 'ck_long.csv'}")


if __name__ == "__main__":
    main()
