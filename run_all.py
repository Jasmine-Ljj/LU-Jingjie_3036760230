"""
run_all.py — one-command reproducible workflow.

Given the raw survey data at Data/raw/public.csv, regenerates every table and
figure of the project in the correct order:

    01_clean.py          -> Data/processed/ck_wide.csv, ck_long.csv
    02_descriptives.py   -> Output/table/table2_*, table3_*, Output/figure/figure1.png
    03_did.py            -> Output/table/table4_did.csv, first_stage.csv
    04_extension.py      -> Output/table/extension_*, Output/figure/extension_*.png

Usage (from the project root):
    python run_all.py

No manual modification of data or code is needed — just place the input file and
run this one command.
"""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
STEPS = [
    "01_clean.py",
    "02_descriptives.py",
    "03_did.py",
    "04_extension.py",
]


def main() -> None:
    for step in STEPS:
        print(f"\n{'=' * 64}\nRunning Code/{step}\n{'=' * 64}")
        result = subprocess.run(
            [sys.executable, str(ROOT / "Code" / step)], cwd=str(ROOT)
        )
        if result.returncode != 0:
            print(f"\n[run_all] Step {step} FAILED with exit code {result.returncode}.")
            sys.exit(result.returncode)

    print("\n[run_all] All steps completed. Tables are in Output/table/, "
          "figures in Output/figure/.")


if __name__ == "__main__":
    main()
