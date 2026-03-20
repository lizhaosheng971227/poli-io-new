
#!/usr/bin/env python3
from __future__ import annotations

"""
Spyder-ready repair pass for the backbone-based party leader AI results.

Purpose
-------
Run a targeted second pass only on party-election rows that are still problematic
after the main GPT-5.4 collection:
- unresolved rows
- rows missing candidate_age_ai
- rows missing candidate_gender_ai_male1
- optionally, low-confidence rows

The repair pass is conservative:
- if a previous pass already identified a leader, the repair prompt treats that
  leader (or leaders array) as the starting point and asks the model to fill
  birth date / birth year / gender, changing leader identity only if clearly
  contradicted by authoritative evidence.
- unresolved rows get a fresh full-resolution prompt.

Outputs
-------
Main outputs go in `BASE_DIR/pyblp_backbone`:
- party_char_repair_queue_from_backbone.csv
- ai_results_campaignavg_gpt54_from_backbone_repairs.jsonl
- ai_results_campaignavg_gpt54_from_backbone_repairs.csv
- ai_results_campaignavg_gpt54_from_backbone_repaired.csv
- party_char_v3_campaignavg_repaired.csv
- party_char_v3_campaignavg_repaired.dta
- party_char_repair_remaining_after_pass.csv
- party_char_repair_patch_summary.csv

Typical Spyder run order
------------------------
1) Build the repair queue only:
   RUN_BUILD_REPAIR_QUEUE = True
   RUN_COLLECT_REPAIRS = False
   RUN_APPLY_REPAIRS_AND_MERGE = False

2) Collect the repairs:
   RUN_BUILD_REPAIR_QUEUE = False
   RUN_COLLECT_REPAIRS = True
   RUN_APPLY_REPAIRS_AND_MERGE = False

3) Apply repairs + build final repaired party-char file:
   RUN_BUILD_REPAIR_QUEUE = False
   RUN_COLLECT_REPAIRS = False
   RUN_APPLY_REPAIRS_AND_MERGE = True
"""

import json
import os
import re
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

# =============================================================================
# SETTINGS FOR SPYDER
# =============================================================================
BASE_DIR = Path(r"/Users/zl279/Downloads/Poli-IO/raw_data")
BACKBONE_DIR = BASE_DIR / "pyblp_backbone"
BACKBONE_DIR.mkdir(parents=True, exist_ok=True)

TARGETS_CSV = BACKBONE_DIR / "party_leader_targets_from_backbone.csv"
TARGETS_DTA = BACKBONE_DIR / "party_leader_targets_from_backbone.dta"

AI_RESULTS_CSV = BACKBONE_DIR / "ai_results_campaignavg_gpt54_from_backbone.csv"
AI_RESULTS_JSONL = BACKBONE_DIR / "ai_results_campaignavg_gpt54_from_backbone.jsonl"

REPAIR_QUEUE_CSV = BACKBONE_DIR / "party_char_repair_queue_from_backbone.csv"
REPAIR_JSONL = BACKBONE_DIR / "ai_results_campaignavg_gpt54_from_backbone_repairs.jsonl"
REPAIR_CSV = BACKBONE_DIR / "ai_results_campaignavg_gpt54_from_backbone_repairs.csv"

PATCHED_AI_RESULTS_CSV = BACKBONE_DIR / "ai_results_campaignavg_gpt54_from_backbone_repaired.csv"
FINAL_REPAIRED_CSV = BACKBONE_DIR / "party_char_v3_campaignavg_repaired.csv"
FINAL_REPAIRED_DTA = BACKBONE_DIR / "party_char_v3_campaignavg_repaired.dta"
PATCH_SUMMARY_CSV = BACKBONE_DIR / "party_char_repair_patch_summary.csv"
REMAINING_AFTER_REPAIR_CSV = BACKBONE_DIR / "party_char_repair_remaining_after_pass.csv"

OPENAI_API_KEY = ""
MODEL = "gpt-5.4"

INCLUDE_LEGACY_HINTS_ON_REPAIR = True
SLEEP_SECONDS = 0.0

RUN_BUILD_REPAIR_QUEUE = False
RUN_COLLECT_REPAIRS = False
RUN_APPLY_REPAIRS_AND_MERGE = True

# Optional small pilot of the repair queue. Set to None to repair all flagged rows.
REPAIR_LIMIT = None

# Optional extra rows to repair:
INCLUDE_LOW_CONFIDENCE = False
CONFIDENCE_THRESHOLD = 0.85

# API behavior
WEB_SEARCH_TOOL = {"type": "web_search", "search_context_size": "low"}
MAX_TOOL_CALLS_FIXED = 3
MAX_TOOL_CALLS_FULL = 5

PARTY_LEADER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "resolved": {"type": ["boolean", "null"]},
        "resolution_status": {"type": ["string", "null"]},
        "leader_name": {"type": ["string", "null"]},
        "leader_role": {"type": ["string", "null"]},
        "leader_role_basis": {"type": ["string", "null"]},
        "is_collective_leadership": {"type": ["boolean", "null"]},
        "is_multiple_leader_case": {"type": ["boolean", "null"]},
        "multiple_leader_basis": {"type": ["string", "null"]},
        "leaders": {
            "type": "array",
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "full_name": {"type": ["string", "null"]},
                    "role": {"type": ["string", "null"]},
                    "role_basis": {"type": ["string", "null"]},
                    "birth_date_iso": {"type": ["string", "null"]},
                    "birth_year": {"type": ["integer", "null"]},
                    "birth_precision": {"type": ["string", "null"]},
                    "gender_text": {"type": ["string", "null"]},
                    "include_in_average": {"type": ["boolean", "null"]},
                },
                "required": [
                    "full_name", "role", "role_basis", "birth_date_iso", "birth_year",
                    "birth_precision", "gender_text", "include_in_average",
                ],
                "additionalProperties": False,
            },
        },
        "birth_date_iso": {"type": ["string", "null"]},
        "birth_year": {"type": ["integer", "null"]},
        "birth_precision": {"type": ["string", "null"]},
        "gender_text": {"type": ["string", "null"]},
        "confidence": {"type": ["number", "null"]},
        "notes": {"type": ["string", "null"]},
        "sources": {
            "type": "array",
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": ["string", "null"]},
                    "url": {"type": ["string", "null"]},
                    "publisher": {"type": ["string", "null"]},
                    "why_relevant": {"type": ["string", "null"]},
                },
                "required": ["title", "url", "publisher", "why_relevant"],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "resolved", "resolution_status", "leader_name", "leader_role", "leader_role_basis",
        "is_collective_leadership", "is_multiple_leader_case", "multiple_leader_basis",
        "leaders", "birth_date_iso", "birth_year", "birth_precision", "gender_text",
        "confidence", "notes", "sources",
    ],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You repair previously collected historical data about who most clearly led a "
    "political party's NATIONAL election campaign on a specific election date. "
    "Prefer the nationally visible campaign leader / lead candidate / "
    "Spitzenkandidat / public campaign face. Use the formal party president only "
    "if no distinct campaign leader is evident. In repair mode, treat previously "
    "identified leader names as fixed starting points unless authoritative evidence "
    "clearly contradicts them. For multiple or collective leadership, return all "
    "relevant leaders in `leaders`, marking include_in_average=true for those who "
    "should count in the average. We will average age and a male-coded gender "
    "variable across included leaders (female=0, male=1). Be conservative. Never "
    "guess. Do not infer gender from name alone. Use as few web searches as "
    "needed and stop once the evidence is sufficient. Prefer official party, "
    "parliament, government, or institutional sources, then strong reference "
    "sources. Return JSON only."
)

# =============================================================================
# Generic helpers
# =============================================================================
def clean_text_scalar(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none", "<na>"}:
        return ""
    return s


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def ensure_datetime(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.Series([pd.NaT] * len(s), index=s.index)


def format_date_for_prompt(x: Any) -> str:
    if pd.isna(x):
        return ""
    try:
        return pd.Timestamp(x).strftime("%Y-%m-%d")
    except Exception:
        return clean_text_scalar(x)


def json_dumps_compact(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"), sort_keys=False)


def parse_json_maybe(x: Any) -> Any:
    s = clean_text_scalar(x)
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def bool_like(x: Any) -> bool | None:
    if pd.isna(x):
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    s = clean_text_scalar(x).lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return None


def nonmissing_scalar(x: Any) -> bool:
    if x is None:
        return False
    if pd.isna(x):
        return False
    if isinstance(x, str) and clean_text_scalar(x) == "":
        return False
    return True


def gender_to_male1(x: Any) -> float | None:
    s = clean_text_scalar(x).lower()
    if not s:
        return None
    male = {"m", "male", "man", "masculine"}
    female = {"f", "female", "woman", "feminine"}
    if s in male:
        return 1.0
    if s in female:
        return 0.0
    if "female" in s and "male" not in s:
        return 0.0
    if "male" in s and "female" not in s:
        return 1.0
    return None


def male1_to_female1(x: Any) -> float | None:
    if x is None or pd.isna(x):
        return None
    try:
        return float(1.0 - float(x))
    except Exception:
        return None


def parse_birth_date_iso(x: Any) -> pd.Timestamp | pd.NaT:
    s = clean_text_scalar(x)
    if not s:
        return pd.NaT
    ts = pd.to_datetime(s, errors="coerce")
    return ts if pd.notna(ts) else pd.NaT


def compute_age_on_election(election_date: Any, birth_date_iso: Any, birth_year: Any) -> tuple[float | None, str | None]:
    ed = pd.to_datetime(election_date, errors="coerce")
    if pd.notna(ed):
        bd = parse_birth_date_iso(birth_date_iso)
        if pd.notna(bd):
            age = ed.year - bd.year - ((ed.month, ed.day) < (bd.month, bd.day))
            return float(age), "exact_date"
    try:
        by = int(birth_year) if birth_year is not None and not pd.isna(birth_year) else None
    except Exception:
        by = None
    if by is not None and pd.notna(ed):
        return float(int(pd.Timestamp(ed).year) - by), "birth_year_only"
    return None, None


def summarize_age_aggregation(methods: list[str | None]) -> str | None:
    methods = [m for m in methods if m]
    if not methods:
        return None
    uniq = set(methods)
    if len(methods) == 1:
        return methods[0]
    if uniq == {"exact_date"}:
        return "mean_exact_date"
    if uniq == {"birth_year_only"}:
        return "mean_birth_year_only"
    return "mean_mixed"


def normalize_leaders_for_aggregation(parsed: dict[str, Any], election_date: Any) -> list[dict[str, Any]]:
    leaders_raw = parsed.get("leaders")
    out: list[dict[str, Any]] = []
    if isinstance(leaders_raw, list):
        for i, item in enumerate(leaders_raw, start=1):
            if not isinstance(item, dict):
                continue
            include = item.get("include_in_average")
            if include is False:
                continue
            age_val, age_method = compute_age_on_election(election_date, item.get("birth_date_iso"), item.get("birth_year"))
            gender_male1 = gender_to_male1(item.get("gender_text"))
            out.append({
                "leader_index": i,
                "full_name": clean_text_scalar(item.get("full_name")),
                "role": clean_text_scalar(item.get("role")),
                "role_basis": clean_text_scalar(item.get("role_basis")),
                "birth_date_iso": item.get("birth_date_iso"),
                "birth_year": item.get("birth_year"),
                "birth_precision": item.get("birth_precision"),
                "gender_text": item.get("gender_text"),
                "include_in_average": True if include is None else bool(include),
                "age_on_election": age_val,
                "age_method": age_method,
                "gender_male1": gender_male1,
                "gender_female1_compat": male1_to_female1(gender_male1),
            })
    if out:
        return out

    fallback_name = clean_text_scalar(parsed.get("leader_name"))
    if fallback_name or parsed.get("birth_date_iso") is not None or parsed.get("birth_year") is not None or parsed.get("gender_text") is not None:
        age_val, age_method = compute_age_on_election(election_date, parsed.get("birth_date_iso"), parsed.get("birth_year"))
        gender_male1 = gender_to_male1(parsed.get("gender_text"))
        return [{
            "leader_index": 1,
            "full_name": fallback_name,
            "role": clean_text_scalar(parsed.get("leader_role")),
            "role_basis": clean_text_scalar(parsed.get("leader_role_basis")),
            "birth_date_iso": parsed.get("birth_date_iso"),
            "birth_year": parsed.get("birth_year"),
            "birth_precision": parsed.get("birth_precision"),
            "gender_text": parsed.get("gender_text"),
            "include_in_average": True,
            "age_on_election": age_val,
            "age_method": age_method,
            "gender_male1": gender_male1,
            "gender_female1_compat": male1_to_female1(gender_male1),
        }]
    return []


def aggregate_leader_metrics(parsed: dict[str, Any], election_date: Any) -> dict[str, Any]:
    leaders = normalize_leaders_for_aggregation(parsed, election_date)
    ages = [float(x["age_on_election"]) for x in leaders if x.get("age_on_election") is not None and not pd.isna(x.get("age_on_election"))]
    genders_male1 = [float(x["gender_male1"]) for x in leaders if x.get("gender_male1") is not None and not pd.isna(x.get("gender_male1"))]
    leader_names = [clean_text_scalar(x.get("full_name")) for x in leaders if clean_text_scalar(x.get("full_name")) != ""]
    primary_name = clean_text_scalar(parsed.get("leader_name")) or (leader_names[0] if leader_names else "")
    joined_name = "; ".join(leader_names) if len(leader_names) > 1 else primary_name
    return {
        "leaders_norm": leaders,
        "leader_count_total": len(leaders),
        "leader_count_age_nonmissing": len(ages),
        "leader_count_gender_nonmissing": len(genders_male1),
        "leader_name_primary": primary_name if primary_name else None,
        "leader_name_joined": joined_name if joined_name else None,
        "candidate_age_ai": float(np.mean(ages)) if ages else None,
        "candidate_age_method": summarize_age_aggregation([x.get("age_method") for x in leaders]),
        "candidate_gender_ai_male1": float(np.mean(genders_male1)) if genders_male1 else None,
        "candidate_gender_ai": male1_to_female1(float(np.mean(genders_male1))) if genders_male1 else None,
    }


def write_csv_and_dta(df: pd.DataFrame, csv_path: Path, dta_path: Path | None = None) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    if dta_path is not None:
        out = df.copy()
        for c in out.columns:
            if out[c].dtype == object:
                non_null = out[c].dropna()
                if len(non_null) == 0:
                    out[c] = pd.Series([""] * len(out), index=out.index, dtype="string")
                    continue
                coerced = pd.to_numeric(non_null, errors="coerce")
                if coerced.notna().all():
                    out[c] = pd.to_numeric(out[c], errors="coerce")
                else:
                    out[c] = out[c].astype("string").fillna("")
            elif str(out[c].dtype) == "Int64":
                out[c] = out[c].astype(float)
        try:
            out.to_stata(dta_path, write_index=False, version=118)
        except Exception:
            pass


def obj_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def extract_response_usage(resp_obj: Any) -> dict[str, Any]:
    usage = obj_get(resp_obj, "usage", None)
    input_details = obj_get(usage, "input_tokens_details", None)
    output_details = obj_get(usage, "output_tokens_details", None)
    return {
        "usage_input_tokens": obj_get(usage, "input_tokens", np.nan),
        "usage_output_tokens": obj_get(usage, "output_tokens", np.nan),
        "usage_total_tokens": obj_get(usage, "total_tokens", np.nan),
        "usage_cached_input_tokens": obj_get(input_details, "cached_tokens", np.nan),
        "usage_reasoning_tokens": obj_get(output_details, "reasoning_tokens", np.nan),
    }


def count_web_search_calls(resp_obj: Any) -> int:
    output = obj_get(resp_obj, "output", None)
    n = 0
    try:
        if output is not None:
            for item in output:
                if obj_get(item, "type", None) == "web_search_call":
                    n += 1
    except Exception:
        pass
    return int(n)


# =============================================================================
# Load data
# =============================================================================
def read_targets() -> pd.DataFrame:
    if TARGETS_CSV.exists():
        parse_cols = ["edate_mpd"] if "edate_mpd" in pd.read_csv(TARGETS_CSV, nrows=0).columns else None
        return pd.read_csv(TARGETS_CSV, parse_dates=parse_cols)
    if TARGETS_DTA.exists():
        df = pd.read_stata(TARGETS_DTA, convert_categoricals=False)
        if "edate_mpd" in df.columns:
            df["edate_mpd"] = ensure_datetime(df["edate_mpd"])
        return df
    raise FileNotFoundError(f"Could not find {TARGETS_CSV} or {TARGETS_DTA}.")


def read_ai_results() -> pd.DataFrame:
    if not AI_RESULTS_CSV.exists():
        raise FileNotFoundError(f"Missing {AI_RESULTS_CSV}. Run the full collection first.")
    df = pd.read_csv(AI_RESULTS_CSV)
    return df


# =============================================================================
# Repair queue
# =============================================================================
def choose_repair_mode(row: pd.Series) -> str:
    resolved = bool_like(row.get("resolved")) is True
    leader_name = clean_text_scalar(row.get("leader_name"))
    is_multiple = bool_like(row.get("is_multiple_leader_case")) is True
    is_collective = bool_like(row.get("is_collective_leadership")) is True
    leaders = parse_json_maybe(row.get("leaders_json"))
    leader_count = int(row.get("leader_count_total")) if nonmissing_scalar(row.get("leader_count_total")) else 0
    if not resolved:
        return "full_resolution"
    if is_multiple or is_collective or leader_count > 1 or (isinstance(leaders, list) and len(leaders) > 1):
        return "repair_leaders_array"
    if leader_name:
        return "repair_fixed_single_leader"
    return "full_resolution"


def build_repair_queue(targets: pd.DataFrame, ai: pd.DataFrame) -> pd.DataFrame:
    out = targets.merge(ai, on=["target_id", "party_election_id"], how="inner", validate="1:1")

    resolved = out.get("resolved", pd.Series(np.nan, index=out.index)).map(bool_like)
    missing_age = out.get("candidate_age_ai", pd.Series(np.nan, index=out.index)).isna()
    missing_gen = out.get("candidate_gender_ai_male1", pd.Series(np.nan, index=out.index)).isna()
    unresolved = resolved != True
    status_bad = out.get("status", pd.Series("", index=out.index)).astype(str).ne("ok")
    low_conf = pd.to_numeric(out.get("confidence", pd.Series(np.nan, index=out.index)), errors="coerce") < float(CONFIDENCE_THRESHOLD)

    need = status_bad | unresolved | missing_age | missing_gen
    if INCLUDE_LOW_CONFIDENCE:
        need = need | low_conf.fillna(False)

    q = out.loc[need].copy()
    q["repair_needs_unresolved"] = unresolved.loc[q.index]
    q["repair_needs_age"] = missing_age.loc[q.index]
    q["repair_needs_gender"] = missing_gen.loc[q.index]
    q["repair_needs_status"] = status_bad.loc[q.index]
    q["repair_needs_low_conf"] = low_conf.loc[q.index].fillna(False)
    q["repair_mode"] = q.apply(choose_repair_mode, axis=1)

    def repair_reason(row: pd.Series) -> str:
        parts = []
        if bool(row.get("repair_needs_status")):
            parts.append("status")
        if bool(row.get("repair_needs_unresolved")):
            parts.append("unresolved")
        if bool(row.get("repair_needs_age")):
            parts.append("missing_age")
        if bool(row.get("repair_needs_gender")):
            parts.append("missing_gender")
        if bool(row.get("repair_needs_low_conf")):
            parts.append("low_conf")
        return "|".join(parts)

    q["repair_reason"] = q.apply(repair_reason, axis=1)

    front = [
        "target_id", "party_election_id", "repair_mode", "repair_reason",
        "country", "iso3c", "election_year", "edate_mpd", "mpd_party_id", "parfam",
        "partyname", "partyname_eng", "share_mpd", "local_aliases", "eng_aliases",
        "queue_reason",
        "status", "resolved", "resolution_status", "leader_name", "leader_role",
        "is_collective_leadership", "is_multiple_leader_case",
        "candidate_age_ai", "candidate_gender_ai_male1", "candidate_gender_ai",
        "confidence", "notes",
    ]
    existing_front = [c for c in front if c in q.columns]
    rest = [c for c in q.columns if c not in existing_front]
    q = q[existing_front + rest].reset_index(drop=True)

    if REPAIR_LIMIT is not None:
        q = q.head(int(REPAIR_LIMIT)).copy()

    return q


# =============================================================================
# Prompt building
# =============================================================================
def _legacy_hint_block(row: pd.Series) -> str:
    if not INCLUDE_LEGACY_HINTS_ON_REPAIR:
        return ""
    if pd.isna(row.get("old_rowid")):
        return ""
    urls = [clean_text_scalar(row.get(c)) for c in ["old_wiki_url", "old_url1", "old_url2", "old_url3"]]
    urls = [u for u in urls if u]
    return f"""

Legacy handwritten clues (verify independently; they may be wrong)
- old_leader_name_raw: {clean_text_scalar(row.get('old_leader_name_raw'))}
- old_role_raw: {clean_text_scalar(row.get('old_role_raw'))}
- old_age_raw: {row.get('old_age_raw')}
- old_gender_raw: {clean_text_scalar(row.get('old_gender_raw'))}
- old_year_born_raw: {clean_text_scalar(row.get('old_year_born_raw'))}
- legacy_urls: {' | '.join(urls)}
""".rstrip()


def make_repair_user_prompt(row: pd.Series) -> str:
    country = clean_text_scalar(row.get("country"))
    iso3c = clean_text_scalar(row.get("iso3c"))
    party_local = clean_text_scalar(row.get("partyname"))
    party_eng = clean_text_scalar(row.get("partyname_eng"))
    share = row.get("share_mpd")
    share_txt = "" if pd.isna(share) else f"{float(share):.3f}"
    parfam = clean_text_scalar(row.get("parfam"))
    mpd_party_id = row.get("mpd_party_id")
    ey = row.get("election_year")
    ed = format_date_for_prompt(row.get("edate_mpd"))
    aliases = []
    for lab, col in [("local_aliases", "local_aliases"), ("english_aliases", "eng_aliases")]:
        txt = clean_text_scalar(row.get(col))
        if txt:
            aliases.append(f"- {lab}: {txt}")
    alias_block = "\n".join(aliases)

    leaders_json = parse_json_maybe(row.get("leaders_json"))
    leaders_pretty = json.dumps(leaders_json, ensure_ascii=False, indent=2) if leaders_json is not None else "[]"

    missing_bits = []
    if bool(row.get("repair_needs_age")):
        missing_bits.append("age")
    if bool(row.get("repair_needs_gender")):
        missing_bits.append("gender")
    if bool(row.get("repair_needs_unresolved")):
        missing_bits.append("leader resolution")
    missing_txt = ", ".join(missing_bits) if missing_bits else "none"

    base = f"""
Target party-election record
- target_id: {row['target_id']}
- party_election_id: {row.get('party_election_id')}
- country: {country}
- iso3c: {iso3c}
- election_year: {ey}
- election_date: {ed}
- mpd_party_id: {mpd_party_id}
- party_family: {parfam}
- party_name_local_or_primary: {party_local}
- party_name_english: {party_eng}
- vote_share_mpd_percent: {share_txt}
{alias_block}

Current first-pass output (starting point only, not guaranteed)
- repair_mode: {clean_text_scalar(row.get('repair_mode'))}
- repair_reason: {clean_text_scalar(row.get('repair_reason'))}
- resolved: {row.get('resolved')}
- resolution_status: {clean_text_scalar(row.get('resolution_status'))}
- leader_name: {clean_text_scalar(row.get('leader_name'))}
- leader_role: {clean_text_scalar(row.get('leader_role'))}
- is_collective_leadership: {row.get('is_collective_leadership')}
- is_multiple_leader_case: {row.get('is_multiple_leader_case')}
- leader_count_total: {row.get('leader_count_total')}
- current_missing_fields: {missing_txt}
- current_confidence: {row.get('confidence')}
- current_notes: {clean_text_scalar(row.get('notes'))}
- current_leaders_json:
{leaders_pretty}
""".strip()

    mode = clean_text_scalar(row.get("repair_mode"))

    if mode == "repair_fixed_single_leader":
        task = """
Repair task
1) Treat the current leader_name as the fixed starting point unless authoritative evidence clearly shows it is wrong.
2) Your main job is to fill the missing birth date / birth year and/or gender for this same leader.
3) Keep the same leader identity if at all possible.
4) If exact birth date is unavailable, return birth_year if reliable.
5) Return that same person in both the top-level leader fields and in `leaders` with include_in_average=true.
6) Use at most 3 sources and stop once the evidence is sufficient.
7) Never guess. If a field cannot be verified, leave it null.
""".strip()
    elif mode == "repair_leaders_array":
        task = """
Repair task
1) Use the existing leaders_json / notes as the starting point.
2) Verify and complete the relevant leader set for this election.
3) If the case is genuinely multiple-leader or collective, return all clearly relevant leaders in `leaders`.
4) Fill birth date / birth year and gender for as many included leaders as can be verified.
5) Mark include_in_average=true for leaders who should count in the average.
6) If the prior leader set is clearly wrong, correct it, but explain briefly in notes.
7) Use at most 5 sources and stop once the evidence is sufficient.
""".strip()
    else:
        task = """
Repair task
1) Re-run the leader identification from scratch for this exact party-election record.
2) Prefer the nationally visible campaign leader / lead candidate / public campaign face.
3) For alliances or coalitions, identify the alliance's national face(s), not merely a constituent-party officer.
4) If reliable sources show joint or collective leadership, return all relevant leaders in `leaders`.
5) If unresolved, set resolved=false and explain briefly.
6) Use at most 5 sources and stop once the evidence is sufficient.
""".strip()

    output_rules = """
Output rules
- Return JSON only.
- If there is one clear leader, fill the top-level leader fields and also include that same person in `leaders`.
- If there are multiple relevant leaders, set is_multiple_leader_case=true and use `leaders` for the people who should be averaged.
- If leadership is collective and no single person dominates, set is_collective_leadership=true.
- For included leaders, fill birth_date_iso if known; otherwise birth_year if known.
- Do not infer gender from name alone.
- Never guess.
""".strip()

    return base + _legacy_hint_block(row) + "\n\n" + task + "\n\n" + output_rules


def build_response_request_body(row: pd.Series, model: str) -> dict[str, Any]:
    mode = clean_text_scalar(row.get("repair_mode"))
    max_tool_calls = MAX_TOOL_CALLS_FULL if mode in {"full_resolution", "repair_leaders_array"} else MAX_TOOL_CALLS_FIXED
    return {
        "model": model,
        "store": False,
        "tools": [WEB_SEARCH_TOOL],
        "tool_choice": "auto",
        "max_tool_calls": int(max_tool_calls),
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_repair_user_prompt(row)},
        ],
        "text": {
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": "party_leader_record_repair",
                "schema": PARTY_LEADER_SCHEMA,
                "strict": True,
            },
        },
    }


# =============================================================================
# OpenAI collection
# =============================================================================
def _set_openai_key_for_session() -> None:
    key = clean_text_scalar(OPENAI_API_KEY)
    if key:
        os.environ["OPENAI_API_KEY"] = key


def collect_repairs(queue_df: pd.DataFrame, out_jsonl: Path, out_csv: Path, model: str) -> None:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("The `openai` package is required. Install it with `pip install openai`.") from exc

    _set_openai_key_for_session()
    client = OpenAI()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    completed: set[str] = set()
    if out_csv.exists():
        try:
            prev = pd.read_csv(out_csv)
            if "target_id" in prev.columns:
                completed.update(prev["target_id"].astype(str).tolist())
        except Exception:
            pass

    flat_rows: list[dict[str, Any]] = []
    if out_csv.exists():
        try:
            flat_rows = pd.read_csv(out_csv).to_dict(orient="records")
        except Exception:
            flat_rows = []

    def flush_flat() -> None:
        if flat_rows:
            pd.DataFrame(flat_rows).to_csv(out_csv, index=False)

    with out_jsonl.open("a", encoding="utf-8") as fout:
        for _, row in queue_df.iterrows():
            tid = str(row["target_id"])
            if tid in completed:
                continue

            body = build_response_request_body(row, model=model)
            last_err = None
            resp_obj = None
            for attempt in range(5):
                try:
                    resp_obj = client.responses.create(**body)
                    last_err = None
                    break
                except Exception as exc:
                    last_err = exc
                    time.sleep(min(2 ** attempt, 20))
            if resp_obj is None:
                rec = {
                    "target_id": tid,
                    "party_election_id": clean_text_scalar(row.get("party_election_id")) or tid,
                    "status": "error",
                    "error": repr(last_err),
                    "repair_mode": clean_text_scalar(row.get("repair_mode")),
                    "repair_reason": clean_text_scalar(row.get("repair_reason")),
                }
                fout.write(json_dumps_compact(rec) + "\n")
                fout.flush()
                if SLEEP_SECONDS > 0:
                    time.sleep(float(SLEEP_SECONDS))
                continue

            output_text = getattr(resp_obj, "output_text", None)
            if output_text is None:
                rec = {
                    "target_id": tid,
                    "party_election_id": clean_text_scalar(row.get("party_election_id")) or tid,
                    "status": "no_output_text",
                    "repair_mode": clean_text_scalar(row.get("repair_mode")),
                    "repair_reason": clean_text_scalar(row.get("repair_reason")),
                    "usage": extract_response_usage(resp_obj),
                    "tool_calls_web_search": count_web_search_calls(resp_obj),
                }
                fout.write(json_dumps_compact(rec) + "\n")
                fout.flush()
                if SLEEP_SECONDS > 0:
                    time.sleep(float(SLEEP_SECONDS))
                continue

            try:
                parsed = json.loads(output_text)
            except Exception as exc:
                rec = {
                    "target_id": tid,
                    "party_election_id": clean_text_scalar(row.get("party_election_id")) or tid,
                    "status": "invalid_json",
                    "error": repr(exc),
                    "repair_mode": clean_text_scalar(row.get("repair_mode")),
                    "repair_reason": clean_text_scalar(row.get("repair_reason")),
                    "output_text": output_text,
                    "usage": extract_response_usage(resp_obj),
                    "tool_calls_web_search": count_web_search_calls(resp_obj),
                }
                fout.write(json_dumps_compact(rec) + "\n")
                fout.flush()
                if SLEEP_SECONDS > 0:
                    time.sleep(float(SLEEP_SECONDS))
                continue

            agg = aggregate_leader_metrics(parsed, row.get("edate_mpd"))
            usage = extract_response_usage(resp_obj)
            tool_calls_web_search = count_web_search_calls(resp_obj)
            leader_name_value = agg.get("leader_name_joined") or clean_text_scalar(parsed.get("leader_name")) or agg.get("leader_name_primary")

            flat = {
                "target_id": tid,
                "party_election_id": clean_text_scalar(row.get("party_election_id")) or tid,
                "status": "ok",
                "repair_mode": clean_text_scalar(row.get("repair_mode")),
                "repair_reason": clean_text_scalar(row.get("repair_reason")),
                "model_requested": model,
                "resolved": parsed.get("resolved"),
                "resolution_status": parsed.get("resolution_status"),
                "leader_name": leader_name_value,
                "leader_name_primary": agg.get("leader_name_primary"),
                "leader_role": parsed.get("leader_role"),
                "leader_role_basis": parsed.get("leader_role_basis"),
                "is_collective_leadership": parsed.get("is_collective_leadership"),
                "is_multiple_leader_case": parsed.get("is_multiple_leader_case"),
                "multiple_leader_basis": parsed.get("multiple_leader_basis"),
                "leader_count_total": agg.get("leader_count_total"),
                "leader_count_age_nonmissing": agg.get("leader_count_age_nonmissing"),
                "leader_count_gender_nonmissing": agg.get("leader_count_gender_nonmissing"),
                "birth_date_iso": parsed.get("birth_date_iso"),
                "birth_year": parsed.get("birth_year"),
                "birth_precision": parsed.get("birth_precision"),
                "gender_text": parsed.get("gender_text"),
                "candidate_age_ai": agg.get("candidate_age_ai"),
                "candidate_age_method": agg.get("candidate_age_method"),
                "candidate_gender_ai_male1": agg.get("candidate_gender_ai_male1"),
                "candidate_gender_ai": agg.get("candidate_gender_ai"),
                "confidence": parsed.get("confidence"),
                "notes": parsed.get("notes"),
                "sources_json": json_dumps_compact(parsed.get("sources", [])),
                "leaders_json": json_dumps_compact(agg.get("leaders_norm", [])),
                "tool_calls_web_search": tool_calls_web_search,
                **usage,
            }
            flat_rows.append(flat)
            rec = {
                "target_id": tid,
                "party_election_id": clean_text_scalar(row.get("party_election_id")) or tid,
                "status": "ok",
                "request_body": body,
                "output_text": output_text,
                "parsed": parsed,
                "flat": flat,
            }
            fout.write(json_dumps_compact(rec) + "\n")
            fout.flush()
            flush_flat()
            completed.add(tid)
            if SLEEP_SECONDS > 0:
                time.sleep(float(SLEEP_SECONDS))
    flush_flat()


# =============================================================================
# Patch and final merge
# =============================================================================
BOOL_COLS = {"resolved", "is_collective_leadership", "is_multiple_leader_case"}
NUMERIC_PREF_COLS = {
    "candidate_age_ai", "candidate_gender_ai_male1", "candidate_gender_ai",
    "leader_count_total", "leader_count_age_nonmissing", "leader_count_gender_nonmissing",
    "confidence", "tool_calls_web_search", "usage_input_tokens", "usage_output_tokens",
    "usage_total_tokens", "usage_cached_input_tokens", "usage_reasoning_tokens",
}
TEXT_PREF_COLS = {
    "status", "model_requested", "resolution_status", "leader_name", "leader_name_primary",
    "leader_role", "leader_role_basis", "multiple_leader_basis", "birth_date_iso", "birth_precision",
    "gender_text", "candidate_age_method", "notes", "sources_json", "leaders_json",
}


def coalesce_prefer_new(new_val: Any, old_val: Any) -> Any:
    return new_val if nonmissing_scalar(new_val) else old_val


def coalesce_bool_prefer_new(new_val: Any, old_val: Any) -> Any:
    bnew = bool_like(new_val)
    bold = bool_like(old_val)
    if bnew is not None:
        return bnew
    if bold is not None:
        return bold
    return np.nan


def merge_old_and_repair_row(old: pd.Series, new: pd.Series) -> dict[str, Any]:
    out: dict[str, Any] = {}
    all_cols = list(dict.fromkeys(list(old.index) + list(new.index)))
    for c in all_cols:
        ov = old.get(c, np.nan)
        nv = new.get(c, np.nan)
        if c in BOOL_COLS:
            out[c] = coalesce_bool_prefer_new(nv, ov)
        elif c in NUMERIC_PREF_COLS:
            out[c] = nv if nonmissing_scalar(nv) else ov
        elif c in TEXT_PREF_COLS:
            out[c] = clean_text_scalar(nv) if clean_text_scalar(nv) != "" else ov
        else:
            out[c] = coalesce_prefer_new(nv, ov)
    # Preserve keys from the original.
    out["target_id"] = clean_text_scalar(old.get("target_id")) or clean_text_scalar(new.get("target_id"))
    out["party_election_id"] = clean_text_scalar(old.get("party_election_id")) or clean_text_scalar(new.get("party_election_id"))
    return out


def repair_is_improvement(old: pd.Series, merged: dict[str, Any]) -> bool:
    old_resolved = bool_like(old.get("resolved")) is True
    new_resolved = bool_like(merged.get("resolved")) is True
    old_age = nonmissing_scalar(old.get("candidate_age_ai"))
    new_age = nonmissing_scalar(merged.get("candidate_age_ai"))
    old_gen = nonmissing_scalar(old.get("candidate_gender_ai_male1"))
    new_gen = nonmissing_scalar(merged.get("candidate_gender_ai_male1"))
    old_leader = clean_text_scalar(old.get("leader_name")) != "" or (nonmissing_scalar(old.get("leader_count_total")) and float(old.get("leader_count_total")) > 0)
    new_leader = clean_text_scalar(merged.get("leader_name")) != "" or (nonmissing_scalar(merged.get("leader_count_total")) and float(merged.get("leader_count_total")) > 0)

    improved = False
    if clean_text_scalar(old.get("status")) != "ok" and clean_text_scalar(merged.get("status")) == "ok":
        improved = True
    if (not old_resolved) and new_resolved:
        improved = True
    if (not old_age) and new_age:
        improved = True
    if (not old_gen) and new_gen:
        improved = True
    if (not old_leader) and new_leader:
        improved = True

    if not improved:
        old_conf = pd.to_numeric(pd.Series([old.get("confidence")]), errors="coerce").iloc[0]
        new_conf = pd.to_numeric(pd.Series([merged.get("confidence")]), errors="coerce").iloc[0]
        if (not old_resolved) and (not new_resolved) and new_leader and (not old_leader or (pd.notna(new_conf) and pd.notna(old_conf) and new_conf > old_conf + 0.15)):
            improved = True
    return bool(improved)


def age_check_status(old_age: Any, ai_age: Any) -> str:
    if pd.isna(old_age) and (ai_age is None or pd.isna(ai_age)):
        return "both_missing"
    if pd.isna(old_age):
        return "old_missing"
    if ai_age is None or pd.isna(ai_age):
        return "ai_missing"
    diff = abs(float(old_age) - float(ai_age))
    if diff < 1e-9:
        return "exact_match"
    if abs(diff - 1.0) < 1e-9:
        return "off_by_one"
    return "mismatch"


def gender_check_status(old_gender_female1: Any, ai_gender_female1: Any) -> str:
    if pd.isna(old_gender_female1) and (ai_gender_female1 is None or pd.isna(ai_gender_female1)):
        return "both_missing"
    if pd.isna(old_gender_female1):
        return "old_missing"
    if ai_gender_female1 is None or pd.isna(ai_gender_female1):
        return "ai_missing"
    if abs(float(old_gender_female1) - float(ai_gender_female1)) < 1e-9:
        return "match"
    return "mismatch"


def final_choice_status(row: pd.Series) -> str:
    resolved = row.get("resolved")
    ai_has = pd.notna(row.get("candidate_age_ai")) or pd.notna(row.get("candidate_gender_ai"))
    old_has = pd.notna(row.get("old_candidate_age")) or pd.notna(row.get("old_candidate_gender"))
    old_quality = clean_text_scalar(row.get("old_quality_class"))
    age_status = clean_text_scalar(row.get("age_check"))
    gender_status = clean_text_scalar(row.get("gender_check"))
    collective = bool(row.get("is_collective_leadership")) if pd.notna(row.get("is_collective_leadership")) else False
    multiple = bool(row.get("is_multiple_leader_case")) if pd.notna(row.get("is_multiple_leader_case")) else False

    if bool(resolved) and collective and not ai_has:
        return "manual_review_collective_leadership"
    if bool(resolved) and ai_has:
        if multiple:
            return "ai_resolved_multiple_leader_case"
        if age_status in {"exact_match", "off_by_one", "old_missing", "ai_missing", "both_missing"} and gender_status in {"match", "old_missing", "ai_missing", "both_missing"}:
            if old_quality == "clean_old_data":
                return "ai_confirmed_or_extended_old"
            return "ai_resolved_old_problematic_row" if old_has else "new_ai_collected"
        return "manual_review_conflict"
    if old_has and old_quality == "clean_old_data":
        return "keep_old_unverified"
    if old_has and old_quality != "clean_old_data":
        return "keep_old_but_flag_for_review"
    if ai_has and not old_has:
        return "new_ai_collected"
    return "unresolved"


def merge_results(targets: pd.DataFrame, ai: pd.DataFrame, out_csv: Path, out_dta: Path | None = None) -> pd.DataFrame:
    out = targets.merge(ai, on=["target_id", "party_election_id"], how="left", validate="1:1")

    if "old_candidate_gender" in out.columns:
        out["old_candidate_gender_male1"] = np.where(pd.notna(out.get("old_candidate_gender")), 1.0 - out.get("old_candidate_gender"), np.nan)
    else:
        out["old_candidate_gender_male1"] = np.nan

    out["age_check"] = [age_check_status(o, a) for o, a in zip(out.get("old_candidate_age", pd.Series(np.nan, index=out.index)), out.get("candidate_age_ai", pd.Series(np.nan, index=out.index)))]
    out["gender_check"] = [gender_check_status(o, a) for o, a in zip(out.get("old_candidate_gender", pd.Series(np.nan, index=out.index)), out.get("candidate_gender_ai", pd.Series(np.nan, index=out.index)))]

    out["candidate_age"] = np.where(pd.notna(out.get("candidate_age_ai")), out.get("candidate_age_ai"), out.get("old_candidate_age"))
    out["candidate_gender"] = np.where(pd.notna(out.get("candidate_gender_ai")), out.get("candidate_gender_ai"), out.get("old_candidate_gender"))
    out["candidate_gender_male1"] = np.where(pd.notna(out.get("candidate_gender_ai_male1")), out.get("candidate_gender_ai_male1"), out.get("old_candidate_gender_male1"))
    out["final_status"] = out.apply(final_choice_status, axis=1)

    keep_front = [
        "party_election_id", "target_id", "iso3c", "election_year", "edate_mpd", "mpd_party_id", "parfam",
        "partyname", "partyname_eng", "share_mpd",
        "candidate_age", "candidate_gender", "candidate_gender_male1",
        "leader_name", "leader_name_primary", "leader_role", "leader_role_basis",
        "is_collective_leadership", "is_multiple_leader_case", "multiple_leader_basis",
        "leader_count_total", "leader_count_age_nonmissing", "leader_count_gender_nonmissing",
        "birth_date_iso", "birth_year", "birth_precision",
        "candidate_age_ai", "candidate_age_method", "candidate_gender_ai", "candidate_gender_ai_male1",
        "old_candidate_age", "old_candidate_gender", "old_candidate_gender_male1",
        "old_leader_name_raw", "old_gender_raw", "old_year_born_raw", "old_role_raw",
        "old_quality_class", "old_match_method", "old_match_score", "age_check", "gender_check",
        "confidence", "final_status", "queue_reason", "sources_json", "leaders_json",
        "tool_calls_web_search", "usage_input_tokens", "usage_output_tokens", "usage_total_tokens",
        "usage_cached_input_tokens", "usage_reasoning_tokens",
    ]
    existing = [c for c in keep_front if c in out.columns]
    remaining = [c for c in out.columns if c not in existing]
    out = out[existing + remaining]
    write_csv_and_dta(out, out_csv, out_dta)
    return out


def summarize_ai(df: pd.DataFrame, label: str) -> None:
    if df.empty:
        print(f"{label}: no rows")
        return
    ok = (df.get("status") == "ok").sum() if "status" in df.columns else 0
    resolved = df.get("resolved", pd.Series(np.nan, index=df.index)).map(bool_like)
    miss_age = df.get("candidate_age_ai", pd.Series(np.nan, index=df.index)).isna().sum()
    miss_gen = df.get("candidate_gender_ai_male1", pd.Series(np.nan, index=df.index)).isna().sum()
    conf = pd.to_numeric(df.get("confidence", pd.Series(np.nan, index=df.index)), errors="coerce")
    multi = (df.get("is_multiple_leader_case") == True).sum() if "is_multiple_leader_case" in df.columns else 0
    collective = (df.get("is_collective_leadership") == True).sum() if "is_collective_leadership" in df.columns else 0
    print(f"\n{label}:")
    print(f"- rows: {len(df)}")
    print(f"- status=ok: {int(ok)}")
    print(f"- resolved=true: {int((resolved == True).sum())}")
    print(f"- multiple leader cases: {int(multi)}")
    print(f"- collective leadership cases: {int(collective)}")
    print(f"- missing AI age: {int(miss_age)}")
    print(f"- missing AI gender (male1): {int(miss_gen)}")
    if conf.notna().any():
        print(f"- mean confidence: {conf.mean():.4f}")
        print(f"- median confidence: {conf.median():.4f}")


def apply_repairs_and_build_final() -> None:
    if not TARGETS_CSV.exists():
        raise FileNotFoundError(f"Missing {TARGETS_CSV}.")
    if not AI_RESULTS_CSV.exists():
        raise FileNotFoundError(f"Missing {AI_RESULTS_CSV}.")
    if not REPAIR_CSV.exists():
        raise FileNotFoundError(f"Missing {REPAIR_CSV}. Run the repair collection first.")

    targets = read_targets()
    orig = pd.read_csv(AI_RESULTS_CSV)
    repairs = pd.read_csv(REPAIR_CSV)

    summarize_ai(orig, "Original AI results")
    summarize_ai(repairs, "Repair results")

    repaired_ok = repairs.loc[repairs["status"].astype(str) == "ok"].copy()

    orig_by_tid = {str(r["target_id"]): r for _, r in orig.iterrows()}
    patch_rows: list[dict[str, Any]] = []
    patch_summary_rows: list[dict[str, Any]] = []

    for _, new in repaired_ok.iterrows():
        tid = str(new["target_id"])
        old = orig_by_tid.get(tid)
        if old is None:
            merged = dict(new)
            apply_patch = True
            reason = "new_row_not_in_original"
        else:
            merged = merge_old_and_repair_row(pd.Series(old), pd.Series(new))
            apply_patch = repair_is_improvement(pd.Series(old), merged)
            reason = "repair_improves_missing_or_unresolved" if apply_patch else "repair_not_better_keep_original"

        if apply_patch:
            patch_rows.append(merged)

        patch_summary_rows.append({
            "target_id": tid,
            "party_election_id": clean_text_scalar(new.get("party_election_id")),
            "repair_mode": clean_text_scalar(new.get("repair_mode")),
            "repair_reason": clean_text_scalar(new.get("repair_reason")),
            "apply_patch": apply_patch,
            "decision_reason": reason,
            "orig_resolved": old.get("resolved") if old is not None else np.nan,
            "new_resolved": new.get("resolved"),
            "orig_candidate_age_ai": old.get("candidate_age_ai") if old is not None else np.nan,
            "new_candidate_age_ai": new.get("candidate_age_ai"),
            "orig_candidate_gender_ai_male1": old.get("candidate_gender_ai_male1") if old is not None else np.nan,
            "new_candidate_gender_ai_male1": new.get("candidate_gender_ai_male1"),
            "orig_leader_name": old.get("leader_name") if old is not None else "",
            "new_leader_name": new.get("leader_name"),
            "orig_confidence": old.get("confidence") if old is not None else np.nan,
            "new_confidence": new.get("confidence"),
        })

    patched = orig.copy()
    if patch_rows:
        patch_df = pd.DataFrame(patch_rows)
        patched = patched.loc[~patched["target_id"].astype(str).isin(patch_df["target_id"].astype(str))].copy()
        patched = pd.concat([patched, patch_df], ignore_index=True, sort=False)
        patched = patched.sort_values(["target_id"]).reset_index(drop=True)

    patched.to_csv(PATCHED_AI_RESULTS_CSV, index=False)
    pd.DataFrame(patch_summary_rows).to_csv(PATCH_SUMMARY_CSV, index=False)

    # Remaining issues after the repair pass.
    resolved = patched.get("resolved", pd.Series(np.nan, index=patched.index)).map(bool_like)
    remaining = patched.loc[
        (patched.get("status", pd.Series("", index=patched.index)).astype(str) != "ok") |
        (resolved != True) |
        (patched.get("candidate_age_ai", pd.Series(np.nan, index=patched.index)).isna()) |
        (patched.get("candidate_gender_ai_male1", pd.Series(np.nan, index=patched.index)).isna())
    ].copy()
    remaining.to_csv(REMAINING_AFTER_REPAIR_CSV, index=False)

    # Final repaired party-char file.
    final = merge_results(targets, patched, FINAL_REPAIRED_CSV, FINAL_REPAIRED_DTA)

    summarize_ai(patched, "Patched AI results")
    print(f"\nPatched AI results written to: {PATCHED_AI_RESULTS_CSV}")
    print(f"Patch summary written to: {PATCH_SUMMARY_CSV}")
    print(f"Remaining issues written to: {REMAINING_AFTER_REPAIR_CSV}")
    print(f"Final repaired party-char CSV written to: {FINAL_REPAIRED_CSV}")
    print(f"Final repaired party-char DTA written to: {FINAL_REPAIRED_DTA}")
    print("\nFinal status counts:")
    print(final["final_status"].value_counts(dropna=False).to_string())


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("Starting backbone-based party leader repair pipeline in Spyder mode...")
    print(f"Base directory: {BASE_DIR}")
    print(f"Backbone directory: {BACKBONE_DIR}")

    if RUN_BUILD_REPAIR_QUEUE:
        print("\n--- BUILD REPAIR QUEUE ---")
        targets = read_targets()
        ai = read_ai_results()
        q = build_repair_queue(targets, ai)
        q.to_csv(REPAIR_QUEUE_CSV, index=False)
        print(f"Saved repair queue: {REPAIR_QUEUE_CSV}")
        print(f"Rows in repair queue: {len(q)}")
        if "repair_mode" in q.columns:
            print("\nRepair modes:")
            print(q["repair_mode"].value_counts(dropna=False).to_string())
        if "repair_reason" in q.columns:
            print("\nRepair reasons:")
            print(q["repair_reason"].value_counts(dropna=False).to_string())

    if RUN_COLLECT_REPAIRS:
        print("\n--- COLLECT REPAIRS ---")
        if not REPAIR_QUEUE_CSV.exists():
            raise FileNotFoundError(f"Missing {REPAIR_QUEUE_CSV}. Build the repair queue first.")
        parse_cols = ["edate_mpd"] if "edate_mpd" in pd.read_csv(REPAIR_QUEUE_CSV, nrows=0).columns else None
        queue_df = pd.read_csv(REPAIR_QUEUE_CSV, parse_dates=parse_cols)
        print(f"Rows to repair this run: {len(queue_df)}")
        if len(queue_df) == 0:
            print("Nothing to repair.")
        else:
            collect_repairs(queue_df, REPAIR_JSONL, REPAIR_CSV, MODEL)
            print(f"Saved repair JSONL: {REPAIR_JSONL}")
            print(f"Saved repair CSV: {REPAIR_CSV}")
            summarize_ai(pd.read_csv(REPAIR_CSV), "Collected repair rows")

    if RUN_APPLY_REPAIRS_AND_MERGE:
        print("\n--- APPLY REPAIRS + BUILD FINAL ---")
        apply_repairs_and_build_final()

    if not any([RUN_BUILD_REPAIR_QUEUE, RUN_COLLECT_REPAIRS, RUN_APPLY_REPAIRS_AND_MERGE]):
        print("No step selected. Set at least one of RUN_BUILD_REPAIR_QUEUE, RUN_COLLECT_REPAIRS, RUN_APPLY_REPAIRS_AND_MERGE to True.")
