# Reusable workflow: Card–Krueger (1994) DiD replication

A reusable, single-command workflow that regenerates the project's main tables and
figures from the raw survey data.

## Input contract

Place the raw survey file at `Data/raw/public.csv` (410 rows × 46 columns, the
Princeton Card–Krueger public replication data). Nothing else is needed.

## Run

```bash
pip install -r requirements.txt
python Code/run_all.py
```

## Pipeline

| Step | Where | Produces |
|---|---|---|
| 1. Clean | `Code/01_clean.py` | `Data/processed/ck_wide.csv`, `ck_long.csv` |
| 2. Descriptives | `Code/run_all.py` | Table 2, Table 3, Figure 1 |
| 3. DiD | `Code/run_all.py` | Table 4 (5 models), wage first stage |
| 4. Extension | `Code/run_all.py` | ownership DDD, border robustness, figure |

## Key definitions (do not change without updating the report)

- **FTE** = full-time + managers + 0.5 × part-time employees.
- **Treatment** = `STATE == 1` (New Jersey).
- **Main sample** = valid FTE in both waves, incl. stores that closed (wave-2
  employment = 0).
- **GAP** = `max(0, (5.05 − WAGE_ST)/WAGE_ST)` for NJ stores, else 0.

## Reuse

To adapt this workflow to another two-period DiD dataset, keep the same pipeline
shape: (1) clean → construct outcome + treatment + sample; (2) describe → means and
distribution; (3) estimate → first-difference DiD with a dose/continuous treatment;
(4) extend → heterogeneity + robustness. Replace the variable names in
`01_clean.py` and `run_all.py`.
