#!/usr/bin/env python3
from __future__ import annotations

"""
Spyder-friendly one-click runner for the full PyBLP pipeline.

How to use
----------
1) Put this file in the same folder as:
   - run_full_pyblp_pipeline.py
   - build_pyblp_backbone_spyder_v2.py
   - party_char_ai_from_backbone_spyder.py
   - party_char_ai_repair_from_backbone_spyder.py
   - patch_iceland_2017_leaders_spyder.py
   - pyblp_multicost_supply_spyder_v5.py
2) Edit only the SETTINGS block below.
3) Press Run in Spyder.

This wrapper does not ask you to flip the original scripts' RUN_* flags. It loads
those scripts, overrides paths/settings in memory, and runs them in the correct order.
The older file `build_pyblp_backbone_spyder.py` is intentionally not used.
"""

import argparse
import importlib.util
import sys
from pathlib import Path


# =============================================================================
# SETTINGS — edit only this block
# =============================================================================
BASE_DIR = Path("/Users/zl279/Downloads/Poli-IO/raw_data")

# Leave as None to auto-detect the default validated input.
VALIDATED_PATH: Path | None = None

# Leave blank to use the OPENAI_API_KEY environment variable.
OPENAI_API_KEY = ""
MODEL = "gpt-5.4"

# If you already have a party-characteristics file and want to use it directly,
# set RUN_MAIN_AI = False and optionally RUN_REPAIR = False, then point this to
# a .dta or .csv file.
PARTY_CHAR_PATH: Path | None = None

# Optional explicit inputs for the final PyBLP stage. Leave as None to auto-detect.
MPD_PATH: Path | None = None
SPREADSHEET_PATH: Path | None = None

# Stage toggles.
RUN_BACKBONE_BUILD = True
RUN_MAIN_AI = True
RUN_REPAIR = True
RUN_FINALIZE = True
RUN_PATCH_ICELAND_2017 = False
RUN_FINAL_PYBLP = True

# Existing-output behavior.
# True  = skip stages whose standard outputs already exist.
# False = force the stage to run again.
SKIP_EXISTING_OUTPUTS = True

# Safe test modes.
DRY_RUN_ONLY = False
PREFLIGHT_ONLY = False

# Optional partial runs for testing. Leave as None for the full queue.
PILOT_LIMIT: int | None = None
REPAIR_LIMIT: int | None = None

# Main AI options.
INCLUDE_LEGACY_HINTS = False
USE_LEGACY_QUEUES = True

# Repair options.
INCLUDE_LOW_CONFIDENCE_REPAIR = False
CONFIDENCE_THRESHOLD = 0.85


# =============================================================================
# Internal wrapper logic
# =============================================================================
def load_cli_runner():
    here = Path(__file__).resolve().parent
    runner_path = here / "run_full_pyblp_pipeline.py"
    if not runner_path.exists():
        raise FileNotFoundError(
            f"Could not find run_full_pyblp_pipeline.py next to this file: {runner_path}"
        )
    spec = importlib.util.spec_from_file_location("run_full_pyblp_pipeline_runtime", runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load runner module from: {runner_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_namespace() -> argparse.Namespace:
    return argparse.Namespace(
        base_dir=BASE_DIR,
        validated_path=VALIDATED_PATH,
        party_char_path=PARTY_CHAR_PATH,
        mpd_path=MPD_PATH,
        spreadsheet_path=SPREADSHEET_PATH,
        api_key=OPENAI_API_KEY,
        model=MODEL,
        skip_ai=not RUN_MAIN_AI,
        skip_repair=not RUN_REPAIR,
        skip_finalize=not RUN_FINALIZE,
        skip_pyblp=not RUN_FINAL_PYBLP,
        patch_iceland=RUN_PATCH_ICELAND_2017,
        no_skip_existing=not SKIP_EXISTING_OUTPUTS,
        dry_run=DRY_RUN_ONLY,
        preflight_only=PREFLIGHT_ONLY,
        pilot_limit=PILOT_LIMIT,
        repair_limit=REPAIR_LIMIT,
        include_legacy_hints=INCLUDE_LEGACY_HINTS,
        no_legacy_queues=not USE_LEGACY_QUEUES,
        include_low_confidence_repair=INCLUDE_LOW_CONFIDENCE_REPAIR,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )


def main() -> int:
    runner = load_cli_runner()
    args = build_namespace()
    return runner.run_pipeline(args)


if __name__ == "__main__":
    raise SystemExit(main())
