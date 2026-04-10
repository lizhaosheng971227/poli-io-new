# pablo/ — ESS-MPD age-cohort regressions (AI-validated data)

Ported from `ess-mpd-paper/` to run with the Zhaosheng AI-validated matched dataset.

## Data

- **Main input**: `Dropbox/ess-mpd-paper/0_data/Zhaosheng_proc/ess_mpd_matched_validated_plus_ai_rigorous_v6_2.dta` (12 GB)
- **Raw MPD**: `Dropbox/ess-mpd-paper/0_data/raw/MPD/MPDataset_MPDS2024a_stata14.dta` (polarization) and `MPDataset_MPDS2025a_stata14.dta` (coverage)
- **Country list**: `Dropbox/ess-mpd-paper/0_data/proc/ess_mpd_matched_clean_reg_TR.dta` (only for extracting ESS country codes in polarization script)

All data paths are absolute to Dropbox; nothing is copied into the repo.

## Execution order

```
1. stata-mp -b do analysis/cohort_regs_main/a3_reg_age_cohorts_idx_TR.do
   → output/tables/tabfinal_idx_{raw,std,pca}_TR.tex

2. stata-mp -b do analysis/polarization/mpd_prepare.do
   → output/tables/MPD_overtime_charts_pol.dta

3. Rscript analysis/polarization/mpd_chart.r
   → output/figures/mpd_charts_overttime/mpd_p82_combined.png

4. stata-mp -b do analysis/coverage_mpd/a_coverage_check.do
   → output/tables/mpd_code_availability.csv

5. Rscript analysis/coverage_mpd/b_coverage_viz.r
   → output/figures/mpd_coverage_{economic,cultural}.png

6. cd slides && latexmk -pdf slides.tex
   → slides/slides.pdf
```

Steps 1, 2, 4 are independent. Step 3 depends on 2. Step 5 depends on 4. Step 6 depends on all.

## Required packages

**Stata**: reghdfe, estout, gtools, isocodes, ebrdify, distinct, unique
**R**: tidyverse, haven, ggtext, ggrepel, scales, writexl, RColorBrewer, data.table
**LaTeX**: metropolis beamer theme, librefranklin, natbib, booktabs, threeparttable
