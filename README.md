# PyBLP pipeline README

This folder contains a multi-stage pipeline that turns a validated ESS–MPD merge into:

1. a respondent-level PyBLP backbone,
2. party-election leader characteristics collected with the OpenAI API,
3. a finalized respondent file with candidate characteristics merged back in, and
4. the final PyBLP demand/supply analysis outputs.

## Which file should you actually run?

For normal use, use one of these two files:

- **`run_full_pyblp_pipeline_spyder.py`**
  - best if you want to open one file in Spyder, edit a few settings at the top, and press Run.
- **`run_full_pyblp_pipeline.py`**
  - best if you want to run from Terminal with command-line arguments.

These two wrappers run the original stage scripts in the correct order. They are the safest entry points because they:

- avoid manual flipping of several `RUN_* = True/False` switches,
- skip outputs that already exist by default,
- write a preflight report before running,
- write a run log after running, and
- keep using **`build_pyblp_backbone_spyder_v2.py`** rather than the older backbone script.

## File-by-file guide

### Original stage scripts

#### `build_pyblp_backbone_spyder_v2.py`
This is the **main backbone builder** and the one you should keep using.

It builds:
- `auto_regression_pyblp_backbone.dta`
- `demo_file_pyblp_backbone.dta`
- `party_election_backbone.csv/.dta`
- `party_leader_targets_seed.csv/.dta`
- `pyblp_backbone_build_report.csv/.json`

It can also do the later **finalize** step that merges party characteristics back into the backbone and writes:
- `auto_regression_pyblp_ready_with_char.dta`

#### `build_pyblp_backbone_spyder.py`
This is the **older backbone script**.

Recommendation: **do not use it** unless you have a very specific reason. The wrappers ignore it on purpose.

#### `party_char_ai_from_backbone_spyder.py`
This is the **main leader-characteristics AI pipeline**.

It runs in three logical stages:
- prep
- collect
- merge

It builds:
- `party_leader_targets_from_backbone.csv/.dta`
- `openai_party_leader_requests_from_backbone.jsonl`
- `ai_results_campaignavg_gpt54_from_backbone.csv/.jsonl`
- `party_char_v3_campaignavg.csv/.dta`

#### `party_char_ai_repair_from_backbone_spyder.py`
This is the **repair pass** for AI rows that are still unresolved, missing age/gender, or otherwise problematic.

It builds:
- `party_char_repair_queue_from_backbone.csv`
- `ai_results_campaignavg_gpt54_from_backbone_repairs.csv/.jsonl`
- `ai_results_campaignavg_gpt54_from_backbone_repaired.csv`
- `party_char_v3_campaignavg_repaired.csv/.dta`
- repair summaries

#### `patch_iceland_2017_leaders_spyder.py`
This is an **optional patch** for a specific Iceland 2017 issue.

By default it writes patched copies rather than overwriting originals.

#### `pyblp_multicost_supply_spyder_v5.py`
This is the **final analysis script**.

It uses the finalized backbone, demo file, MPD dataset, and sign spreadsheet to build the final PyBLP outputs.

Main outputs go to:
- `pyblp_backbone/pyblp_multicost_supply_out_v4/`

### New wrapper files

#### `run_full_pyblp_pipeline.py`
Terminal/CLI wrapper.

Useful when you want command-line control, for example:

```bash
python run_full_pyblp_pipeline.py \
  --base-dir "/Users/zl279/Downloads/Poli-IO/raw_data" \
  --api-key "$OPENAI_API_KEY" \
  --patch-iceland
```

#### `run_full_pyblp_pipeline_spyder.py`
Spyder-friendly wrapper.

Open this file, edit the **SETTINGS** block at the top, and press Run.

This is the easiest file to use if you do not want to think about the order.

## The actual pipeline order

The wrappers run the stages in this order:

1. backbone build
2. AI prep
3. AI collect
4. AI merge
5. repair queue
6. repair collect
7. repair apply
8. finalize backbone with party characteristics
9. optional Iceland patch
10. final PyBLP analysis

## Inputs you need

At minimum, the pipeline expects these inputs to exist somewhere under your `BASE_DIR` or to be passed explicitly:

### Needed for the backbone build
- validated ESS–MPD merge:
  - preferred: `0_data/proc/ess_mpd_matched_validated.dta`
  - fallback: `ess_mpd_matched_validated.dta`

### Needed for the AI stages
- OpenAI API key
- the `openai` Python package

### Needed for the final PyBLP stage
- one of the MPD data files, for example:
  - `0_data/MPD/MPDataset_MPDS2025a_stata14.dta`
  - `0_data/MPD/MPDataset_MPDS2024a_stata14.dta`
- the MPD items spreadsheet, for example:
  - `MPD_items_codebook_v1a.xlsx`
  - `0_data/MPD_items_codebook_v1a.xlsx`
- the `pyblp` and `statsmodels` Python packages

## Recommended package install

A typical environment will need at least:

```bash
pip install pandas numpy openai statsmodels pyblp
```

## Quick start in Spyder

Open **`run_full_pyblp_pipeline_spyder.py`** and edit only the top block.

### Full run from scratch

Set:

```python
BASE_DIR = Path("/Users/zl279/Downloads/Poli-IO/raw_data")
OPENAI_API_KEY = ""   # or leave blank and use env var
RUN_BACKBONE_BUILD = True
RUN_MAIN_AI = True
RUN_REPAIR = True
RUN_FINALIZE = True
RUN_PATCH_ICELAND_2017 = False
RUN_FINAL_PYBLP = True
SKIP_EXISTING_OUTPUTS = True
DRY_RUN_ONLY = False
PREFLIGHT_ONLY = False
```

Then press Run.

### Check everything without running

Set:

```python
PREFLIGHT_ONLY = True
```

This writes a preflight report and exits before any stage runs.

### Just see the planned stages

Set:

```python
DRY_RUN_ONLY = True
```

### Skip AI and use an existing party-characteristics file

Set:

```python
RUN_MAIN_AI = False
RUN_REPAIR = False
PARTY_CHAR_PATH = Path("/Users/zl279/Downloads/Poli-IO/raw_data/pyblp_backbone/party_char_v3_campaignavg_repaired.dta")
```

`PARTY_CHAR_PATH` may be either **`.dta`** or **`.csv`** in the new wrappers.

### Run only finalize + PyBLP using existing outputs

Set:

```python
RUN_BACKBONE_BUILD = False
RUN_MAIN_AI = False
RUN_REPAIR = False
RUN_FINALIZE = True
RUN_FINAL_PYBLP = True
PARTY_CHAR_PATH = Path("/Users/zl279/Downloads/Poli-IO/raw_data/pyblp_backbone/party_char_v3_campaignavg_repaired.dta")
```

## What the wrappers now check before running

The updated wrappers do a preflight pass that checks:

- whether the required source scripts are next to the runner,
- whether `BASE_DIR` exists,
- whether the validated ESS–MPD input exists when the backbone build is needed,
- whether `openai` is installed and an API key is present when AI collection is actually planned,
- whether `pyblp` and `statsmodels` are installed when the final PyBLP stage is planned,
- whether an MPD dataset and sign spreadsheet can be found when the final PyBLP stage is planned,
- whether a user-supplied party-char path exists and has a valid suffix.

It also writes:

- `pyblp_backbone/master_pipeline_preflight.json`
- `pyblp_backbone/master_pipeline_run_log.json`

## Important behavior changes in the updated wrapper

The updated wrapper is safer than the first version in a few ways:

- it no longer fails just because the validated input is missing **when the backbone build is not actually needed**,
- it no longer tries to infer a party-char file **when finalize is skipped**,
- it now accepts **party-char CSV files** and converts them to a temporary Stata file for finalize,
- it only requires the OpenAI key right before a collect stage that really needs it.

## Outputs to expect

### After backbone build
Under `pyblp_backbone/`:
- `auto_regression_pyblp_backbone.dta`
- `demo_file_pyblp_backbone.dta`
- `party_election_backbone.csv/.dta`
- `party_leader_targets_seed.csv/.dta`

### After main AI
Under `pyblp_backbone/`:
- `party_leader_targets_from_backbone.csv/.dta`
- `ai_results_campaignavg_gpt54_from_backbone.csv/.jsonl`
- `party_char_v3_campaignavg.csv/.dta`

### After repair
Under `pyblp_backbone/`:
- `party_char_v3_campaignavg_repaired.csv/.dta`
- repair summary files

### After finalize
Under `pyblp_backbone/`:
- `auto_regression_pyblp_ready_with_char.dta`

### After the final PyBLP run
Under `pyblp_backbone/pyblp_multicost_supply_out_v4/`:
- `demand_common_support_complete_case_multicost.csv`
- `supply_multicost_specs.csv`
- `supply_multicost_sample_summary.csv`
- `party_election_supply_panel.csv`
- plus other diagnostics/products written by the PyBLP script

## Common problems

### “No OpenAI API key”
Either:
- paste the key into `OPENAI_API_KEY` in the Spyder wrapper, or
- set the environment variable before launching Spyder/Terminal.

### “The openai package is required”
Install it:

```bash
pip install openai
```

### “The final PyBLP stage could not start because a required package is missing”
Install the missing packages:

```bash
pip install pyblp statsmodels
```

### Local file named `pyblp.py`
If you have a file named `pyblp.py` in your current working directory, it can shadow the real package and break the final stage. Rename it and remove `__pycache__`.

### Existing outputs make it look like nothing happened
That is usually because:

```python
SKIP_EXISTING_OUTPUTS = True
```

If you want to force a rerun, set:

```python
SKIP_EXISTING_OUTPUTS = False
```

Be careful with that setting for AI stages because it can trigger new API usage and costs.

## Recommended practice

For a first attempt, use this sequence:

1. `PREFLIGHT_ONLY = True`
2. fix any reported path/package problems
3. set `PREFLIGHT_ONLY = False`
4. keep `SKIP_EXISTING_OUTPUTS = True`
5. run the full pipeline

## One honest caveat

No wrapper can make the pipeline literally guaranteed on every machine, because success still depends on:

- your local file paths,
- the actual contents/readability of your `.dta` and `.xlsx` files,
- installed package versions,
- API access,
- and the runtime behavior of `pyblp` on your data.

What the new wrappers do is make the pipeline **much safer and more predictable** by checking the most common failure points up front and by enforcing the right stage order.
