#!/usr/bin/env python3
from __future__ import annotations

"""
Master runner for the PyBLP backbone -> leader-AI -> repair -> finalize -> PyBLP pipeline.

This wrapper is meant to sit next to these original scripts and orchestrate them in the
recommended order without you flipping True/False flags by hand:

- build_pyblp_backbone_spyder_v2.py
- party_char_ai_from_backbone_spyder.py
- party_char_ai_repair_from_backbone_spyder.py
- patch_iceland_2017_leaders_spyder.py
- pyblp_multicost_supply_spyder_v5.py

It intentionally ignores the older `build_pyblp_backbone_spyder.py` file.

Typical full run
----------------
python run_full_pyblp_pipeline.py \
    --base-dir "/Users/zl279/Downloads/Poli-IO/raw_data" \
    --api-key "$OPENAI_API_KEY" \
    --patch-iceland

Notes
-----
- By default, this runs the full pipeline, including the final PyBLP analysis.
- API-using stages are skipped automatically when their output files already exist,
  unless you pass --no-skip-existing.
- If you pass --pilot-limit or --repair-limit, you are asking for a partial AI run.
  That is useful for testing, but the merged/finalized outputs will also be partial.
"""

import argparse
import json
import os
import re
import sys
import types
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_BASE_DIR = Path("/Users/zl279/Downloads/Poli-IO/raw_data")
DEFAULT_MODEL = "gpt-5.4"


@dataclass
class StepRecord:
    name: str
    status: str
    started_at: str
    finished_at: str | None = None
    outputs: list[str] = field(default_factory=list)
    note: str = ""


def ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def say(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the full PyBLP backbone/AI/PyBLP pipeline in one file.")
    p.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    p.add_argument("--validated-path", type=Path, default=None)
    p.add_argument("--party-char-path", type=Path, default=None,
                   help="Use an existing party-char .dta/.csv instead of building one through the AI steps.")
    p.add_argument("--mpd-path", type=Path, default=None)
    p.add_argument("--spreadsheet-path", type=Path, default=None)
    p.add_argument("--api-key", default="", help="OpenAI API key. If omitted, OPENAI_API_KEY from the environment is used.")
    p.add_argument("--model", default=DEFAULT_MODEL)

    p.add_argument("--skip-ai", action="store_true", help="Skip AI prep/collect/merge and use an existing party-char file.")
    p.add_argument("--skip-repair", action="store_true", help="Skip the repair pass.")
    p.add_argument("--skip-finalize", action="store_true", help="Skip merging party characteristics back into auto_regression.")
    p.add_argument("--skip-pyblp", action="store_true", help="Skip the final PyBLP estimation stage.")
    p.add_argument("--patch-iceland", action="store_true", help="Run the Iceland 2017 patch step.")
    p.add_argument("--no-skip-existing", action="store_true", help="Re-run stages even if their usual outputs already exist.")
    p.add_argument("--dry-run", action="store_true", help="Print the planned stages and exit.")

    p.add_argument("--pilot-limit", type=int, default=None, help="Collect only the first N main AI rows.")
    p.add_argument("--repair-limit", type=int, default=None, help="Collect only the first N repair rows.")
    p.add_argument("--include-legacy-hints", action="store_true", help="Expose legacy manual hints to the model during main collection.")
    p.add_argument("--no-legacy-queues", action="store_true", help="Do not use the legacy handwritten CSV for queues/audit hints.")
    p.add_argument("--include-low-confidence-repair", action="store_true")
    p.add_argument("--confidence-threshold", type=float, default=0.85)

    return p.parse_args()


def require_api_key(api_key: str) -> None:
    if api_key:
        return
    raise RuntimeError(
        "An OpenAI API key is required for AI collection stages. "
        "Pass --api-key or set OPENAI_API_KEY in your environment."
    )


def patch_base_dir_in_source(source: str, base_dir: Path) -> str:
    replacement = f"BASE_DIR = Path({repr(str(base_dir))})"
    patched, n = re.subn(
        r"(?m)^BASE_DIR\s*=\s*Path\((?:r)?['\"].*?['\"]\)",
        replacement,
        source,
        count=1,
    )
    if n == 0:
        raise RuntimeError("Could not find BASE_DIR assignment to patch.")
    return patched


def load_script_module(script_path: Path, module_name: str, base_dir: Path | None = None) -> types.ModuleType:
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")
    source = script_path.read_text(encoding="utf-8")
    if base_dir is not None:
        source = patch_base_dir_in_source(source, base_dir)
    module = types.ModuleType(module_name)
    module.__file__ = str(script_path)
    module.__name__ = module_name
    exec(compile(source, str(script_path), "exec"), module.__dict__)
    return module


def all_exist(paths: list[Path]) -> bool:
    return bool(paths) and all(p.exists() for p in paths)


def record_step(steps: list[StepRecord], name: str, status: str, outputs: list[Path] | None = None, note: str = "") -> None:
    steps.append(
        StepRecord(
            name=name,
            status=status,
            started_at=ts(),
            finished_at=ts(),
            outputs=[str(p) for p in (outputs or [])],
            note=note,
        )
    )


def write_run_log(log_path: Path, args: argparse.Namespace, steps: list[StepRecord], failed: bool = False, error: str = "") -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": ts(),
        "failed": failed,
        "error": error,
        "settings": {
            "base_dir": str(args.base_dir),
            "validated_path": str(args.validated_path) if args.validated_path else None,
            "party_char_path": str(args.party_char_path) if args.party_char_path else None,
            "mpd_path": str(args.mpd_path) if args.mpd_path else None,
            "spreadsheet_path": str(args.spreadsheet_path) if args.spreadsheet_path else None,
            "skip_ai": bool(args.skip_ai),
            "skip_repair": bool(args.skip_repair),
            "skip_finalize": bool(args.skip_finalize),
            "skip_pyblp": bool(args.skip_pyblp),
            "patch_iceland": bool(args.patch_iceland),
            "skip_existing": not bool(args.no_skip_existing),
            "pilot_limit": args.pilot_limit,
            "repair_limit": args.repair_limit,
            "include_legacy_hints": bool(args.include_legacy_hints),
            "use_legacy_queues": not bool(args.no_legacy_queues),
            "include_low_confidence_repair": bool(args.include_low_confidence_repair),
            "confidence_threshold": float(args.confidence_threshold),
            "model": args.model,
        },
        "steps": [s.__dict__ for s in steps],
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def default_validated_path(base_dir: Path) -> Path:
    p = base_dir / "0_data" / "proc" / "ess_mpd_matched_validated.dta"
    return p if p.exists() else (base_dir / "ess_mpd_matched_validated.dta")


def backbone_paths(base_dir: Path) -> dict[str, Path]:
    out = base_dir / "pyblp_backbone"
    return {
        "dir": out,
        "auto": out / "auto_regression_pyblp_backbone.dta",
        "demo": out / "demo_file_pyblp_backbone.dta",
        "pe_csv": out / "party_election_backbone.csv",
        "pe_dta": out / "party_election_backbone.dta",
        "seed_csv": out / "party_leader_targets_seed.csv",
        "seed_dta": out / "party_leader_targets_seed.dta",
        "report_csv": out / "pyblp_backbone_build_report.csv",
        "report_json": out / "pyblp_backbone_build_report.json",
        "final_auto": out / "auto_regression_pyblp_ready_with_char.dta",
        "targets_csv": out / "party_leader_targets_from_backbone.csv",
        "targets_dta": out / "party_leader_targets_from_backbone.dta",
        "batch_jsonl": out / "openai_party_leader_requests_from_backbone.jsonl",
        "ai_results_csv": out / "ai_results_campaignavg_gpt54_from_backbone.csv",
        "ai_results_jsonl": out / "ai_results_campaignavg_gpt54_from_backbone.jsonl",
        "final_party_csv": out / "party_char_v3_campaignavg.csv",
        "final_party_dta": out / "party_char_v3_campaignavg.dta",
        "repair_queue_csv": out / "party_char_repair_queue_from_backbone.csv",
        "repair_jsonl": out / "ai_results_campaignavg_gpt54_from_backbone_repairs.jsonl",
        "repair_csv": out / "ai_results_campaignavg_gpt54_from_backbone_repairs.csv",
        "patched_ai_results_csv": out / "ai_results_campaignavg_gpt54_from_backbone_repaired.csv",
        "final_repaired_csv": out / "party_char_v3_campaignavg_repaired.csv",
        "final_repaired_dta": out / "party_char_v3_campaignavg_repaired.dta",
        "patch_summary_csv": out / "party_char_repair_patch_summary.csv",
        "remaining_after_repair_csv": out / "party_char_repair_remaining_after_pass.csv",
        "iceland_patch_report_csv": out / "iceland_2017_patch_report.csv",
        "iceland_patch_report_json": out / "iceland_2017_patch_report.json",
        "pyblp_out_dir": out / "pyblp_multicost_supply_out_v4",
        "master_log": out / "master_pipeline_run_log.json",
    }


def configure_backbone_module(mod: types.ModuleType, base_dir: Path, validated_path: Path, party_char_path: Path | None = None) -> None:
    paths = backbone_paths(base_dir)
    mod.BASE_DIR = base_dir
    mod.DATA0_DIR = base_dir / "0_data"
    mod.PROC_DIR = mod.DATA0_DIR / "proc"
    mod.OUTPUT_DIR = paths["dir"]
    mod.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mod.VALIDATED_PATH = validated_path
    mod.AUTO_OUT = paths["auto"]
    mod.DEMO_OUT = paths["demo"]
    mod.PARTY_ELECTION_CSV = paths["pe_csv"]
    mod.PARTY_ELECTION_DTA = paths["pe_dta"]
    mod.AI_SEED_CSV = paths["seed_csv"]
    mod.AI_SEED_DTA = paths["seed_dta"]
    mod.REPORT_CSV = paths["report_csv"]
    mod.REPORT_JSON = paths["report_json"]
    mod.FINAL_AUTO_WITH_CHAR = paths["final_auto"]
    if party_char_path is not None:
        mod.PARTY_CHAR_PATH = party_char_path


def configure_ai_module(mod: types.ModuleType, base_dir: Path, api_key: str, model: str, args: argparse.Namespace) -> None:
    paths = backbone_paths(base_dir)
    mod.BASE_DIR = base_dir
    mod.BACKBONE_DIR = paths["dir"]
    mod.BACKBONE_DIR.mkdir(parents=True, exist_ok=True)
    mod.SEED_DTA = paths["seed_dta"]
    mod.SEED_CSV = paths["seed_csv"]
    mod.PARTY_ELECTION_DTA = paths["pe_dta"]
    mod.PARTY_ELECTION_CSV = paths["pe_csv"]
    mod.LEGACY_CSV_PATH = base_dir / "elections_with_wiki_TEST50_important.csv"
    mod.OUT_DIR = paths["dir"]
    mod.TARGETS_CSV = paths["targets_csv"]
    mod.TARGETS_DTA = paths["targets_dta"]
    mod.BATCH_JSONL = paths["batch_jsonl"]
    mod.AI_RESULTS_JSONL = paths["ai_results_jsonl"]
    mod.AI_RESULTS_CSV = paths["ai_results_csv"]
    mod.FINAL_CSV = paths["final_party_csv"]
    mod.FINAL_DTA = paths["final_party_dta"]
    mod.OPENAI_API_KEY = api_key
    mod.MODEL = model
    mod.USE_LEGACY_CSV_FOR_HINTS_AND_QUEUES = not bool(args.no_legacy_queues)
    mod.INCLUDE_LEGACY_HINTS = bool(args.include_legacy_hints)
    mod.PILOT_LIMIT = args.pilot_limit


def configure_repair_module(mod: types.ModuleType, base_dir: Path, api_key: str, model: str, args: argparse.Namespace) -> None:
    paths = backbone_paths(base_dir)
    mod.BASE_DIR = base_dir
    mod.BACKBONE_DIR = paths["dir"]
    mod.BACKBONE_DIR.mkdir(parents=True, exist_ok=True)
    mod.TARGETS_CSV = paths["targets_csv"]
    mod.TARGETS_DTA = paths["targets_dta"]
    mod.AI_RESULTS_CSV = paths["ai_results_csv"]
    mod.AI_RESULTS_JSONL = paths["ai_results_jsonl"]
    mod.REPAIR_QUEUE_CSV = paths["repair_queue_csv"]
    mod.REPAIR_JSONL = paths["repair_jsonl"]
    mod.REPAIR_CSV = paths["repair_csv"]
    mod.PATCHED_AI_RESULTS_CSV = paths["patched_ai_results_csv"]
    mod.FINAL_REPAIRED_CSV = paths["final_repaired_csv"]
    mod.FINAL_REPAIRED_DTA = paths["final_repaired_dta"]
    mod.PATCH_SUMMARY_CSV = paths["patch_summary_csv"]
    mod.REMAINING_AFTER_REPAIR_CSV = paths["remaining_after_repair_csv"]
    mod.OPENAI_API_KEY = api_key
    mod.MODEL = model
    mod.REPAIR_LIMIT = args.repair_limit
    mod.INCLUDE_LOW_CONFIDENCE = bool(args.include_low_confidence_repair)
    mod.CONFIDENCE_THRESHOLD = float(args.confidence_threshold)


def configure_patch_module(mod: types.ModuleType, base_dir: Path) -> None:
    paths = backbone_paths(base_dir)
    mod.BASE_DIR = base_dir
    mod.BACKBONE_DIR = paths["dir"]
    mod.REPORT_CSV = paths["iceland_patch_report_csv"]
    mod.REPORT_JSON = paths["iceland_patch_report_json"]
    mod.TARGET_FILES = [
        paths["ai_results_csv"],
        paths["patched_ai_results_csv"],
        paths["final_party_csv"],
        paths["final_party_dta"],
        paths["final_repaired_csv"],
        paths["final_repaired_dta"],
        paths["final_auto"],
    ]


def configure_pyblp_module(mod: types.ModuleType, base_dir: Path, args: argparse.Namespace) -> None:
    paths = backbone_paths(base_dir)
    mod.BASE_DIR = base_dir
    mod.DATA0_DIR = base_dir / "0_data"
    mod.BACKBONE_DIR = paths["dir"]
    mod.OUT_DIR = paths["pyblp_out_dir"]
    mod.OUT_DIR.mkdir(parents=True, exist_ok=True)
    mod.AUTO_CANDIDATES = [
        paths["dir"] / "auto_regression_pyblp_ready_with_char_patched.dta",
        paths["final_auto"],
        paths["auto"],
    ]
    mod.DEMO_CANDIDATES = [paths["demo"]]
    mod.MPD_CANDIDATES = [
        args.mpd_path if args.mpd_path else (mod.DATA0_DIR / "MPD" / "MPDataset_MPDS2025a_stata14.dta"),
        mod.DATA0_DIR / "MPD" / "MPDataset_MPDS2024a_stata14.dta",
        Path("/Users/zl279/Downloads/OneDrive_1_7-31-2025/MPDataset_MPDS2024a_stata14.dta"),
    ]
    mod.SPREADSHEET_CANDIDATES = [
        args.spreadsheet_path if args.spreadsheet_path else (base_dir / "MPD_items_codebook_v1a.xlsx"),
        mod.DATA0_DIR / "MPD_items_codebook_v1a.xlsx",
        Path("/Users/zl279/Downloads/MPD_items_codebook_v1a.xlsx"),
    ]
    mod.AUTO_REG_PATH = None
    mod.DEMO_PATH = None
    mod.MPD_PATH = None
    mod.SPREADSHEET_PATH = None


def infer_party_char_path(base_dir: Path, args: argparse.Namespace, prefer_repaired: bool, ai_ran_this_time: bool) -> Path:
    paths = backbone_paths(base_dir)
    if args.party_char_path is not None:
        return args.party_char_path
    if prefer_repaired and paths["final_repaired_dta"].exists():
        return paths["final_repaired_dta"]
    if ai_ran_this_time:
        return paths["final_party_dta"]
    if paths["final_repaired_dta"].exists():
        return paths["final_repaired_dta"]
    if paths["final_party_dta"].exists():
        return paths["final_party_dta"]
    raise FileNotFoundError(
        "Could not determine a party-characteristics file for finalization. "
        "Either run the AI stages or pass --party-char-path."
    )


def main() -> int:
    args = parse_args()
    base_dir = args.base_dir.expanduser().resolve()
    if args.validated_path is not None:
        args.validated_path = args.validated_path.expanduser().resolve()
    if args.party_char_path is not None:
        args.party_char_path = args.party_char_path.expanduser().resolve()
    if args.mpd_path is not None:
        args.mpd_path = args.mpd_path.expanduser().resolve()
    if args.spreadsheet_path is not None:
        args.spreadsheet_path = args.spreadsheet_path.expanduser().resolve()

    scripts_dir = Path(__file__).resolve().parent
    paths = backbone_paths(base_dir)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    skip_existing = not bool(args.no_skip_existing)
    api_key = (args.api_key or os.environ.get("OPENAI_API_KEY", "")).strip()

    if args.pilot_limit is not None:
        say("Pilot limit is active. The AI outputs will be partial.")
    if args.repair_limit is not None:
        say("Repair limit is active. The repair outputs will be partial.")

    steps: list[StepRecord] = []
    ai_ran_this_time = False
    prefer_repaired = not bool(args.skip_repair)

    if args.dry_run:
        say(f"Base directory: {base_dir}")
        say("Planned stages:")
        print("  1. build backbone")
        if not args.skip_ai:
            print("  2. AI prep")
            print("  3. AI collect")
            print("  4. AI merge")
        if not args.skip_repair:
            print("  5. repair queue")
            print("  6. repair collect")
            print("  7. repair apply")
        if not args.skip_finalize:
            print("  8. finalize backbone with party characteristics")
        if args.patch_iceland:
            print("  9. Iceland 2017 patch")
        if not args.skip_pyblp:
            print("  10. run PyBLP analysis")
        return 0

    try:
        validated_path = args.validated_path if args.validated_path is not None else default_validated_path(base_dir)
        if not validated_path.exists():
            raise FileNotFoundError(f"Validated ESS-MPD input not found: {validated_path}")

        # ------------------------------------------------------------------
        # 1) Backbone build
        # ------------------------------------------------------------------
        backbone_outputs = [paths["auto"], paths["demo"], paths["pe_csv"], paths["pe_dta"], paths["seed_csv"], paths["seed_dta"]]
        if skip_existing and all_exist(backbone_outputs):
            say("Skipping backbone build because the backbone outputs already exist.")
            record_step(steps, "build_backbone", "skipped", backbone_outputs, note="outputs already exist")
        else:
            say("Running backbone build...")
            bb = load_script_module(scripts_dir / "build_pyblp_backbone_spyder_v2.py", "build_pyblp_backbone_spyder_v2_runtime", base_dir)
            configure_backbone_module(bb, base_dir, validated_path)
            bb.RUN_BUILD_BACKBONE = True
            bb.RUN_FINALIZE_WITH_PARTY_CHAR = False
            bb.main()
            record_step(steps, "build_backbone", "completed", backbone_outputs)

        # ------------------------------------------------------------------
        # 2-4) Main AI pipeline
        # ------------------------------------------------------------------
        if not args.skip_ai:
            collect_outputs = [paths["ai_results_csv"], paths["ai_results_jsonl"]]
            repair_collect_needed = not args.skip_repair
            if (not (skip_existing and all_exist(collect_outputs))) or repair_collect_needed:
                require_api_key(api_key)

            ai = load_script_module(scripts_dir / "party_char_ai_from_backbone_spyder.py", "party_char_ai_from_backbone_runtime", base_dir)
            configure_ai_module(ai, base_dir, api_key, args.model, args)

            prep_outputs = [paths["targets_csv"], paths["targets_dta"], paths["batch_jsonl"]]
            if skip_existing and all_exist(prep_outputs):
                say("Skipping AI prep because its outputs already exist.")
                record_step(steps, "ai_prep", "skipped", prep_outputs, note="outputs already exist")
            else:
                say("Running AI prep...")
                ai.run_prep_step()
                record_step(steps, "ai_prep", "completed", prep_outputs)

            collect_skip_allowed = skip_existing and args.pilot_limit is None
            if collect_skip_allowed and all_exist(collect_outputs):
                say("Skipping AI collect because the collection outputs already exist.")
                record_step(steps, "ai_collect", "skipped", collect_outputs, note="outputs already exist")
            else:
                say("Running AI collect...")
                ai.run_collect_step()
                record_step(steps, "ai_collect", "completed", collect_outputs)

            merge_outputs = [paths["final_party_csv"], paths["final_party_dta"]]
            merge_skip_allowed = skip_existing and args.pilot_limit is None
            if merge_skip_allowed and all_exist(merge_outputs):
                say("Skipping AI merge because the merged party-char outputs already exist.")
                record_step(steps, "ai_merge", "skipped", merge_outputs, note="outputs already exist")
            else:
                say("Running AI merge...")
                ai.run_merge_step()
                record_step(steps, "ai_merge", "completed", merge_outputs)
            ai_ran_this_time = True
        else:
            say("Skipping the main AI pipeline because --skip-ai was provided.")
            record_step(steps, "ai_pipeline", "skipped", note="--skip-ai")

        # ------------------------------------------------------------------
        # 5-7) Repair pipeline
        # ------------------------------------------------------------------
        repair_queue_empty = False
        if not args.skip_repair:
            require_api_key(api_key)
            repair = load_script_module(scripts_dir / "party_char_ai_repair_from_backbone_spyder.py", "party_char_ai_repair_from_backbone_runtime", base_dir)
            configure_repair_module(repair, base_dir, api_key, args.model, args)

            repair_queue_outputs = [paths["repair_queue_csv"]]
            if skip_existing and all_exist(repair_queue_outputs):
                say("Skipping repair-queue build because the queue already exists.")
                record_step(steps, "repair_queue", "skipped", repair_queue_outputs, note="outputs already exist")
            else:
                say("Building repair queue...")
                targets = repair.read_targets()
                ai_df = repair.read_ai_results()
                q = repair.build_repair_queue(targets, ai_df)
                q.to_csv(repair.REPAIR_QUEUE_CSV, index=False)
                say(f"Saved repair queue: {repair.REPAIR_QUEUE_CSV}")
                record_step(steps, "repair_queue", "completed", repair_queue_outputs)

            if not paths["repair_queue_csv"].exists():
                raise FileNotFoundError(f"Repair queue not found: {paths['repair_queue_csv']}")
            queue_df = pd.read_csv(paths["repair_queue_csv"])
            repair_queue_empty = queue_df.empty
            if repair_queue_empty:
                say("Repair queue is empty. Skipping repair collection and repair apply.")
                record_step(steps, "repair_collect", "skipped", note="repair queue is empty")
                record_step(steps, "repair_apply", "skipped", note="repair queue is empty")
                prefer_repaired = False
            else:
                repair_collect_outputs = [paths["repair_csv"], paths["repair_jsonl"]]
                repair_collect_skip_allowed = skip_existing and args.repair_limit is None
                if repair_collect_skip_allowed and all_exist(repair_collect_outputs):
                    say("Skipping repair collect because repair outputs already exist.")
                    record_step(steps, "repair_collect", "skipped", repair_collect_outputs, note="outputs already exist")
                else:
                    say("Running repair collect...")
                    repair._set_openai_key_for_session()
                    parse_cols = ["edate_mpd"] if "edate_mpd" in pd.read_csv(repair.REPAIR_QUEUE_CSV, nrows=0).columns else None
                    queue_df = pd.read_csv(repair.REPAIR_QUEUE_CSV, parse_dates=parse_cols)
                    if repair.REPAIR_LIMIT is not None:
                        queue_df = queue_df.head(int(repair.REPAIR_LIMIT)).copy()
                    say(f"Rows to repair in this run: {len(queue_df)}")
                    if len(queue_df) == 0:
                        say("Nothing to repair after applying REPAIR_LIMIT.")
                    else:
                        repair.collect_repairs(queue_df, repair.REPAIR_JSONL, repair.REPAIR_CSV, repair.MODEL)
                    record_step(steps, "repair_collect", "completed", repair_collect_outputs)

                repair_apply_outputs = [
                    paths["patched_ai_results_csv"], paths["final_repaired_csv"], paths["final_repaired_dta"],
                    paths["patch_summary_csv"], paths["remaining_after_repair_csv"],
                ]
                repair_apply_skip_allowed = skip_existing and all_exist(repair_apply_outputs)
                if repair_apply_skip_allowed:
                    say("Skipping repair apply because the repaired outputs already exist.")
                    record_step(steps, "repair_apply", "skipped", repair_apply_outputs, note="outputs already exist")
                else:
                    say("Applying repairs and building repaired party-char outputs...")
                    repair.apply_repairs_and_build_final()
                    record_step(steps, "repair_apply", "completed", repair_apply_outputs)
        else:
            say("Skipping the repair pipeline because --skip-repair was provided.")
            record_step(steps, "repair_pipeline", "skipped", note="--skip-repair")
            prefer_repaired = False

        # ------------------------------------------------------------------
        # 8) Finalize backbone with chosen party-char file
        # ------------------------------------------------------------------
        selected_party_char = infer_party_char_path(base_dir, args, prefer_repaired=prefer_repaired, ai_ran_this_time=ai_ran_this_time)
        if not args.skip_finalize:
            finalize_outputs = [paths["final_auto"]]
            if skip_existing and all_exist(finalize_outputs):
                say("Skipping finalize step because the finalized auto file already exists.")
                record_step(steps, "finalize_backbone", "skipped", finalize_outputs, note=f"outputs already exist; would have used {selected_party_char}")
            else:
                say(f"Finalizing backbone with party-char file: {selected_party_char}")
                bb = load_script_module(scripts_dir / "build_pyblp_backbone_spyder_v2.py", "build_pyblp_backbone_spyder_v2_finalize_runtime", base_dir)
                configure_backbone_module(bb, base_dir, validated_path, party_char_path=selected_party_char)
                bb.finalize_with_party_char(bb.AUTO_OUT, bb.PARTY_CHAR_PATH, bb.FINAL_AUTO_WITH_CHAR)
                record_step(steps, "finalize_backbone", "completed", finalize_outputs, note=f"party_char_path={selected_party_char}")
        else:
            say("Skipping finalize step because --skip-finalize was provided.")
            record_step(steps, "finalize_backbone", "skipped", note="--skip-finalize")

        # ------------------------------------------------------------------
        # 9) Optional Iceland patch
        # ------------------------------------------------------------------
        if args.patch_iceland:
            patch_outputs = [paths["iceland_patch_report_csv"], paths["iceland_patch_report_json"]]
            if skip_existing and all_exist(patch_outputs):
                say("Skipping Iceland patch because its report files already exist.")
                record_step(steps, "patch_iceland_2017", "skipped", patch_outputs, note="outputs already exist")
            else:
                say("Running Iceland 2017 patch...")
                patch = load_script_module(scripts_dir / "patch_iceland_2017_leaders_spyder.py", "patch_iceland_2017_leaders_runtime", base_dir)
                configure_patch_module(patch, base_dir)
                patch.main()
                record_step(steps, "patch_iceland_2017", "completed", patch_outputs)
        else:
            record_step(steps, "patch_iceland_2017", "skipped", note="not requested")

        # ------------------------------------------------------------------
        # 10) Optional final PyBLP run
        # ------------------------------------------------------------------
        if not args.skip_pyblp:
            pyblp_outputs = [
                paths["pyblp_out_dir"] / "demand_common_support_complete_case_multicost.csv",
                paths["pyblp_out_dir"] / "supply_multicost_specs.csv",
                paths["pyblp_out_dir"] / "supply_multicost_sample_summary.csv",
                paths["pyblp_out_dir"] / "party_election_supply_panel.csv",
            ]
            if skip_existing and all_exist(pyblp_outputs):
                say("Skipping PyBLP run because the main analysis outputs already exist.")
                record_step(steps, "run_pyblp", "skipped", pyblp_outputs, note="outputs already exist")
            else:
                say("Running the final PyBLP analysis...")
                try:
                    pyb = load_script_module(scripts_dir / "pyblp_multicost_supply_spyder_v5.py", "pyblp_multicost_supply_runtime", base_dir)
                except ModuleNotFoundError as e:
                    raise RuntimeError(
                        "The final PyBLP stage could not start because a required package is missing. "
                        "Install pyblp and statsmodels in the same environment, then rerun."
                    ) from e
                configure_pyblp_module(pyb, base_dir, args)
                pyb.main()
                record_step(steps, "run_pyblp", "completed", pyblp_outputs)
        else:
            say("Skipping the final PyBLP analysis because --skip-pyblp was provided.")
            record_step(steps, "run_pyblp", "skipped", note="--skip-pyblp")

        write_run_log(paths["master_log"], args, steps, failed=False)
        say("Pipeline finished successfully.")
        say(f"Run log written to: {paths['master_log']}")
        return 0

    except Exception as e:
        write_run_log(paths["master_log"], args, steps, failed=True, error=f"{type(e).__name__}: {e}")
        say("Pipeline failed.")
        say(f"Run log written to: {paths['master_log']}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
