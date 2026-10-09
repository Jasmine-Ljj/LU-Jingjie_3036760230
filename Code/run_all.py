"""
run_all.py — one-command reproducibility.

The single entry point that regenerates EVERY table and figure of the project
from the raw survey data.  Run it with:

    python Code/run_all.py

from the project root.  It executes the four pipeline steps in order:

    01_clean.py          -> Data/processed/ck_wide.csv, ck_long.csv
    02_descriptives.py   -> Output/table/table2_*, table3_*; Output/figure/figure1.png
    03_did.py            -> Output/table/table4_did.csv, first_stage.csv
    04_extension.py      -> Output/table/extension_*; Output/figure/extension_*.png

and then verifies that every expected output file was written.  No manual
modification of data or code is needed — just place Data/raw/public.csv and run
this one command.
"""

from pathlib import Path
import subprocess
import sys

CODE_DIR = Path(__file__).resolve().parent   # Code/
PROJECT = CODE_DIR.parent                    # project root

STEPS = [
    "01_clean.py",
    "02_descriptives.py",
    "03_did.py",
    "04_extension.py",
]

# Every table and figure the workflow is expected to produce.
EXPECTED_OUTPUTS = [
    "Data/processed/ck_wide.csv",
    "Data/processed/ck_long.csv",
    "Output/table/table2_panelA_store_types.csv",
    "Output/table/table2_means.csv",
    "Output/table/table3_did.csv",
    "Output/figure/figure1.png",
    "Output/table/table4_did.csv",
    "Output/table/first_stage.csv",
    "Output/table/extension_ownership_ddd.csv",
    "Output/table/extension_ownership_wage.csv",
    "Output/table/extension_border_robustness.csv",
    "Output/figure/extension_ownership_did.png",
]


def run_step(step: str) -> None:
    print(f"\n{'=' * 64}\nRunning Code/{step}\n{'=' * 64}")
    result = subprocess.run(
        [sys.executable, str(CODE_DIR / step)], cwd=str(PROJECT)
    )
    if result.returncode != 0:
        print(f"\n[run_all] Step {step} FAILED with exit code {result.returncode}.")
        sys.exit(result.returncode)


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
    for step in STEPS:
        run_step(step)
    verify_outputs()


if __name__ == "__main__":
    main()
