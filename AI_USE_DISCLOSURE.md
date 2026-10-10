# AI-Use Disclosure

**Course:** ECO6067 — Individual Project
**Student:** LU Jingjie (3036760230)
**Paper replicated:** Card, D., & Krueger, A. B. (1994). "Minimum Wages and Employment: A Case Study of the Fast-Food Industry in New Jersey and Pennsylvania." *American Economic Review*, 84(4), 772–793.

---

## What AI was used for

AI assistance (Claude Code) was used throughout the project as a coding and writing assistant. Specifically, AI was used to:

- scaffold the Python analysis scripts (`01_clean.py` and `run_all.py`);
- help interpret the survey codebook and reconstruct variable definitions (e.g., FTE, the full-meal price, and the wage-gap variable);
- diagnose and fix implementation issues; and
- draft the empirical report and the README.

## How AI-generated code, analysis, and writing were checked

Every AI-suggested data choice was verified against the original paper and the data itself. In particular: the FTE definition, the treatment of closed stores (which is what yields the exact +2.75 DiD), the sample restrictions, and the choice of ordinary vs. robust standard errors were each traced back to the paper and validated against the published numbers. All regression output was independently re-derived by hand-inspecting the intermediate data (e.g., means by state and wave), and the final workflow was executed end-to-end to confirm reproducibility.

## Important AI errors or suggestions that were corrected or rejected

1. **Heteroskedasticity-robust standard errors.** AI initially used robust (HC1) standard errors, which produced a standard error of 1.37 rather than the paper's 1.19; upon checking, I found the paper uses ordinary OLS standard errors, and switched accordingly.
2. **Collinear region dummies.** AI initially included region dummies together with the NJ treatment dummy in a DiD regression, which is perfectly collinear and caused the coefficient to collapse from 2.75 to 1.04; I corrected this by following the paper's design, which uses the continuous wage-gap variable (not the NJ dummy) in the region-controlled specification.
3. **Table 4 outcome.** AI initially treated Table 4's employment measure as a full set of outcomes (full-time, part-time, managers); I corrected this after confirming the paper's Table 4 uses only FTE as the dependent variable.
4. **Extension topic.** The initial "treatment-intensity" extension idea overlapped with the paper's own wage-gap variable, so I replaced it with the ownership-heterogeneity extension.
5. **Figure 1 content.** AI initially produced "Figure 1" as a histogram of the change in FTE employment. When I compared it against the original paper, I found that the paper's Figure 1 is the *distribution of starting wage rates*, so I corrected it to the two-panel wage distribution (February and November 1992).
6. **Table 2 completeness and the recruiting-bonus denominator.** AI's Table 2 was split into three fragments, omitted the wave-2 "% wage = $5.05" row and the wave-2 meal/hours/bonus rows, and computed the wave-2 recruiting-bonus share over all stores (19.3%) rather than over non-missing responses (20.3%, matching the paper). I corrected these and merged the fragments into the paper's single four-column table.
7. **Table 3 layout.** AI's Table 3 had only three columns (PA, NJ, NJ − PA); the paper reports eight columns, adding the three New Jersey starting-wage groups and their differences. I expanded it to the full 8-column × 5-row table.
8. **Word-conversion math bug.** When converting the merged Table 2 to Word, pandoc interpreted the literal dollar signs ("Starting wage ($/hr)", "% wage = $4.25") as math delimiters and dropped table rows; I diagnosed this and escaped the dollar signs so the amounts render literally while the equations remain typeset math.

All final data choices, sample definitions, and interpretations are my responsibility.
