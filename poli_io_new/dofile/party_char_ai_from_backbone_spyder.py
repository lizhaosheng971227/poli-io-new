#!/usr/bin/env python3
from __future__ import annotations

"""
Spyder-ready AI collection pipeline for party leader characteristics,
aligned to the PyBLP backbone built by `build_pyblp_backbone_spyder_v2.py`.

What this version changes
-------------------------
- Uses the canonical `party_leader_targets_seed` table from `pyblp_backbone`
  instead of rebuilding party-election targets from `ess_mpd_matched_validated.dta`.
- Keeps the stable key `party_election_id`, so the final leader file can be merged
  back into `auto_regression_pyblp_backbone.dta` via `party_election_id`.
- Preserves the GPT-5.4 campaign-leader logic and the multiple-leader averaging rule:
    * candidate_age = mean age across included leaders
    * candidate_gender_male1 = mean gender where female=0, male=1
    * candidate_gender = backward-compatible field where female=1, male=0
- Optionally matches the old manual CSV (`elections_with_wiki_TEST50_important.csv`)
  onto the backbone seed for audit queues and hinting. This is optional and can be
  turned off.

Outputs
-------
Main outputs go in `BASE_DIR/pyblp_backbone`:
- party_leader_targets_from_backbone.csv / .dta
- ai_results_campaignavg_gpt54_from_backbone.csv / .jsonl
- party_char_v3_campaignavg.csv / .dta

The final `.dta` is designed to be consumed by the finalize step in
`build_pyblp_backbone_spyder_v2.py`.
"""

import json
import os
import re
import time
import unicodedata
import warnings
from pathlib import Path
from typing import Any, Iterable
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

# Quiet some noisy SDK serialization warnings when web search tool calls are present.
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

# =============================================================================
# SETTINGS FOR SPYDER
# =============================================================================
BASE_DIR = Path(r"/Users/zl279/Downloads/Poli-IO/raw_data")
BACKBONE_DIR = BASE_DIR / "pyblp_backbone"
BACKBONE_DIR.mkdir(parents=True, exist_ok=True)

SEED_DTA = BACKBONE_DIR / "party_leader_targets_seed.dta"
SEED_CSV = BACKBONE_DIR / "party_leader_targets_seed.csv"
PARTY_ELECTION_DTA = BACKBONE_DIR / "party_election_backbone.dta"
PARTY_ELECTION_CSV = BACKBONE_DIR / "party_election_backbone.csv"
LEGACY_CSV_PATH = BASE_DIR / "elections_with_wiki_TEST50_important.csv"

OUT_DIR = BACKBONE_DIR
TARGETS_CSV = OUT_DIR / "party_leader_targets_from_backbone.csv"
TARGETS_DTA = OUT_DIR / "party_leader_targets_from_backbone.dta"
MANUAL_REVIEW_CSV = OUT_DIR / "party_char_manual_review_suggestions_from_backbone.csv"
BATCH_JSONL = OUT_DIR / "openai_party_leader_requests_from_backbone.jsonl"

AI_RESULTS_JSONL = OUT_DIR / "ai_results_campaignavg_gpt54_from_backbone.jsonl"
AI_RESULTS_CSV = OUT_DIR / "ai_results_campaignavg_gpt54_from_backbone.csv"

FINAL_CSV = OUT_DIR / "party_char_v3_campaignavg.csv"
FINAL_DTA = OUT_DIR / "party_char_v3_campaignavg.dta"

# If you prefer environment variables, leave this blank.
# For Spyder-only use, you may paste your API key here.
OPENAI_API_KEY = ""
MODEL = "gpt-5.4"

# Optional: use the old handwritten CSV to create audit queues and optional hints.
USE_LEGACY_CSV_FOR_HINTS_AND_QUEUES = True
INCLUDE_LEGACY_HINTS = False  # blind by default

SLEEP_SECONDS = 0.0
RUN_PREP = False
RUN_COLLECT = True
RUN_MERGE = False

PILOT_LIMIT = None  # set to None for all rows
# Example filters:
# ONLY_QUEUE_REASONS = ["collect_missing_old_data"]
# ONLY_QUEUE_REASONS = ["audit_old_ambiguous_or_multiperson", "collect_missing_old_data"]
ONLY_QUEUE_REASONS = None

# =============================================================================
# MODEL / API CONFIG
# =============================================================================
# GPT-5.4 + web_search tool + structured JSON output.
# Verified against current Responses API docs with `web_search` and `json_schema`.
WEB_SEARCH_TOOL = {"type": "web_search", "search_context_size": "low"}

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
            "maxItems": 4,
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
            "maxItems": 4,
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
    "You collect factual historical data about who most clearly led a political "
    "party's national election campaign on a specific election date. Core rule: "
    "prefer the nationally visible campaign leader / lead candidate / "
    "Spitzenkandidat / public campaign face for that election. Use the formal "
    "party president or organizational leader only if no distinct campaign "
    "leader is evident. If reliable sources show joint, co-, or collective "
    "national campaign leadership, return all clearly relevant leaders in the "
    "`leaders` array and mark them for inclusion in the average. We will "
    "average age and a male-coded gender variable across included leaders "
    "(female=0, male=1). Be conservative. Never guess. Do not infer gender from "
    "name alone. Minimize browsing and stop once you have enough evidence. "
    "Prefer official party, parliament, government, or institutional sources, "
    "then strong reference sources. Return JSON only."
)

# =============================================================================
# Helpers
# =============================================================================
def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def clean_text_scalar(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none", "<na>"}:
        return ""
    return s


def ascii_fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s


def maybe_repair_mojibake(s: str) -> str:
    s = clean_text_scalar(s)
    if not s:
        return s
    if not any(tok in s for tok in ["Ã", "â", "ð", "Ð", "Å", "Ä"]):
        return s
    for src_enc in ("latin1", "cp1252"):
        try:
            repaired = s.encode(src_enc).decode("utf-8")
            bad_before = sum(tok in s for tok in ["Ã", "â", "ð", "Ð", "Å", "Ä"])
            bad_after = sum(tok in repaired for tok in ["Ã", "â", "ð", "Ð", "Å", "Ä"])
            if bad_after < bad_before:
                return repaired
        except Exception:
            continue
    return s


def norm_name_scalar(x: Any) -> str:
    s = maybe_repair_mojibake(clean_text_scalar(x))
    if not s:
        return ""
    s = ascii_fold(s).lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def sequence_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return 100.0 * SequenceMatcher(None, a, b).ratio()


def safe_round_share(x: Any, digits: int = 3) -> float | None:
    try:
        if pd.isna(x):
            return None
        return round(float(x), digits)
    except Exception:
        return None


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
        "multiple_leader_case_ai": bool(parsed.get("is_multiple_leader_case")) or len(leaders) > 1,
        "is_collective_leadership_ai": bool(parsed.get("is_collective_leadership")) if parsed.get("is_collective_leadership") is not None else False,
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


def read_seed_table() -> pd.DataFrame:
    if SEED_DTA.exists():
        seed = pd.read_stata(SEED_DTA, convert_categoricals=False)
    elif SEED_CSV.exists():
        seed = pd.read_csv(SEED_CSV, parse_dates=["edate_mpd"] if "edate_mpd" in pd.read_csv(SEED_CSV, nrows=0).columns else None)
    else:
        raise FileNotFoundError(f"Could not find {SEED_DTA} or {SEED_CSV}. Run the backbone builder first.")

    out = seed.copy()
    for c in ["country", "iso2c", "iso3c", "partyname", "partyname_eng", "local_aliases", "eng_aliases"]:
        if c in out.columns:
            out[c] = out[c].map(clean_text_scalar)
    if "election_year" in out.columns:
        out["election_year"] = to_num(out["election_year"]).astype("Int64")
    if "mpd_party_id" in out.columns:
        out["mpd_party_id"] = to_num(out["mpd_party_id"]).astype("Int64")
    if "parfam" in out.columns:
        out["parfam"] = to_num(out["parfam"]).astype("Int64")
    if "share_mpd" in out.columns:
        out["share_mpd"] = to_num(out["share_mpd"])
    if "edate_mpd" in out.columns:
        out["edate_mpd"] = ensure_datetime(out["edate_mpd"])

    if "party_election_id" not in out.columns:
        raise ValueError("Seed table must contain `party_election_id`.")
    if "target_id" not in out.columns:
        out["target_id"] = out["party_election_id"]
    out["target_id"] = out["target_id"].map(clean_text_scalar)
    out["party_election_id"] = out["party_election_id"].map(clean_text_scalar)

    if "partyname_norm" not in out.columns:
        out["partyname_norm"] = out.get("partyname", pd.Series("", index=out.index)).map(norm_name_scalar)
    if "partyname_eng_norm" not in out.columns:
        out["partyname_eng_norm"] = out.get("partyname_eng", pd.Series("", index=out.index)).map(norm_name_scalar)
    if "share_key" not in out.columns:
        out["share_key"] = out.get("share_mpd", pd.Series(np.nan, index=out.index)).map(safe_round_share)

    out = out.drop_duplicates(subset=["party_election_id"]).reset_index(drop=True)
    return out


# =============================================================================
# Optional legacy matching for queues/hints
# =============================================================================
FUZZY_ACCEPT_SCORE = 94.0
FUZZY_MARGIN = 5.0
SHARE_TOLERANCE = 1.0


def contains_multiperson_hint(s: str) -> bool:
    s0 = clean_text_scalar(s).lower()
    if not s0:
        return False
    if s0 in {"m/f", "f/m"}:
        return True
    return bool(re.search(r"[/;]| and | & ", s0))


def role_looks_ambiguous(s: str) -> bool:
    s0 = clean_text_scalar(s).lower()
    if not s0:
        return False
    if s0 in {"?", "many leader", "chamber of deputies", "l"}:
        return True
    good_keywords = ["leader", "lead candidate", "candidate", "president", "prime minister", "top candidate", "chair"]
    return not any(k in s0 for k in good_keywords)


def legacy_age_source(age_num: float | None, yearborn_num: float | None, election_year: Any) -> str:
    has_age = age_num is not None and not pd.isna(age_num)
    has_yb = yearborn_num is not None and not pd.isna(yearborn_num)
    if has_age and has_yb:
        calc = float(election_year) - float(yearborn_num) if election_year is not None and not pd.isna(election_year) else np.nan
        if np.isfinite(calc):
            diff = abs(float(age_num) - float(calc))
            if diff < 1e-9:
                return "both_consistent"
            if abs(diff - 1.0) < 1e-9:
                return "both_off_by_one"
            return "both_inconsistent"
        return "both_present_no_check"
    if has_age:
        return "age_field_only"
    if has_yb:
        return "yearborn_only"
    return "missing"


def load_legacy_manual_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    cols = {c.lower(): c for c in raw.columns}
    def pick(*names: str) -> str | None:
        for n in names:
            if n in raw.columns:
                return n
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    req = {
        "iso3c": pick("iso3c", "iso3"),
        "election_year": pick("election_year", "year"),
        "partyname_eng": pick("partyname_eng"),
        "share_mpd": pick("share_mpd"),
    }
    missing = [k for k, v in req.items() if v is None]
    if missing:
        raise ValueError(f"Legacy manual CSV missing required columns: {missing}")

    out = pd.DataFrame(index=raw.index)
    out["old_iso3c"] = raw[req["iso3c"]].map(clean_text_scalar).str.upper()
    out["old_election_year"] = to_num(raw[req["election_year"]]).astype("Int64")
    out["old_partyname"] = raw[pick("partyname")].map(maybe_repair_mojibake) if pick("partyname") else pd.NA
    out["old_partyname_eng"] = raw[req["partyname_eng"]].map(maybe_repair_mojibake)
    out["old_share_mpd"] = to_num(raw[req["share_mpd"]])
    out["old_leader_name_raw"] = raw[pick("Name", "name")].map(maybe_repair_mojibake) if pick("Name", "name") else ""
    out["old_age_raw"] = to_num(raw[pick("Age", "age")]) if pick("Age", "age") else np.nan
    out["old_gender_raw"] = raw[pick("Gender", "gender")].map(clean_text_scalar) if pick("Gender", "gender") else ""
    out["old_year_born_raw"] = raw[pick("Year Born", "yearborn", "YearBorn")].map(clean_text_scalar) if pick("Year Born", "yearborn", "YearBorn") else ""
    out["old_role_raw"] = raw[pick("Candidate/President/Leader", "role")].map(clean_text_scalar) if pick("Candidate/President/Leader", "role") else ""
    out["old_wiki_url"] = raw[pick("wiki_url")].map(clean_text_scalar) if pick("wiki_url") else ""
    out["old_url1"] = raw[pick("url1")].map(clean_text_scalar) if pick("url1") else ""
    out["old_url2"] = raw[pick("url2")].map(clean_text_scalar) if pick("url2") else ""
    out["old_url3"] = raw[pick("url3")].map(clean_text_scalar) if pick("url3") else ""
    out["old_note"] = raw[pick("Note", "note")].map(clean_text_scalar) if pick("Note", "note") else ""

    out["old_partyname_norm"] = out["old_partyname"].map(norm_name_scalar)
    out["old_partyname_eng_norm"] = out["old_partyname_eng"].map(norm_name_scalar)
    out["old_share_key"] = out["old_share_mpd"].map(safe_round_share)
    out["old_rowid"] = np.arange(1, len(out) + 1).astype(int)
    out["old_rowid"] = out["old_rowid"].map(lambda x: f"o{x:06d}")

    out["old_year_born_first4"] = pd.to_numeric(out["old_year_born_raw"].astype(str).str.extract(r"(\d{4})", expand=False), errors="coerce")
    out["old_candidate_age_legacy"] = out["old_age_raw"].copy()
    mask_age_missing = out["old_candidate_age_legacy"].isna() & out["old_year_born_first4"].notna()
    out.loc[mask_age_missing, "old_candidate_age_legacy"] = out.loc[mask_age_missing, "old_election_year"] - out.loc[mask_age_missing, "old_year_born_first4"]

    out["old_candidate_gender_legacy"] = 1.0
    gnorm = out["old_gender_raw"].astype(str).str.strip().str.lower().replace({"m/f": "f"})
    out.loc[gnorm == "m", "old_candidate_gender_legacy"] = 0.0

    out["old_multi_person_flag"] = (
        out["old_leader_name_raw"].map(contains_multiperson_hint)
        | out["old_year_born_raw"].map(contains_multiperson_hint)
        | out["old_gender_raw"].map(contains_multiperson_hint)
    )
    out["old_role_ambiguous_flag"] = out["old_role_raw"].map(role_looks_ambiguous)
    out["old_has_any_url"] = (out[["old_wiki_url", "old_url1", "old_url2", "old_url3"]] != "").any(axis=1)
    out["old_age_source"] = [legacy_age_source(a, y, ey) for a, y, ey in zip(out["old_age_raw"], out["old_year_born_first4"], out["old_election_year"])]

    def age_internal_status(row: pd.Series) -> str:
        src = row["old_age_source"]
        if src in {"both_consistent", "both_off_by_one"}:
            return src
        if src == "both_inconsistent":
            return "both_inconsistent"
        if src == "age_field_only":
            return "age_only"
        if src == "yearborn_only":
            return "yearborn_derived"
        return "missing"

    out["old_age_internal_status"] = out.apply(age_internal_status, axis=1)

    def quality(row: pd.Series) -> str:
        if pd.isna(row["old_candidate_age_legacy"]):
            return "missing_old_data"
        if bool(row["old_multi_person_flag"]) or bool(row["old_role_ambiguous_flag"]):
            return "ambiguous_old_data"
        if row["old_age_internal_status"] == "both_inconsistent":
            return "inconsistent_old_data"
        return "clean_old_data"

    out["old_quality_class"] = out.apply(quality, axis=1)
    return out


LEGACY_COPY_COLS = [
    "old_rowid", "old_quality_class", "old_candidate_age_legacy", "old_candidate_gender_legacy",
    "old_leader_name_raw", "old_age_raw", "old_gender_raw", "old_year_born_raw", "old_role_raw",
    "old_wiki_url", "old_url1", "old_url2", "old_url3", "old_note", "old_has_any_url",
    "old_multi_person_flag", "old_role_ambiguous_flag", "old_age_source", "old_age_internal_status",
]


def unique_key_lookup(df: pd.DataFrame, key_cols: list[str]) -> dict[tuple[Any, ...], dict[str, Any]]:
    g = df.groupby(key_cols, dropna=False, as_index=False)
    counts = g.size().rename(columns={"size": "_n"})
    uniq_keys = counts.loc[counts["_n"] == 1, key_cols]
    if uniq_keys.empty:
        return {}
    m = df.merge(uniq_keys.assign(_keep=1), on=key_cols, how="inner")
    out: dict[tuple[Any, ...], dict[str, Any]] = {}
    for _, row in m.iterrows():
        out[tuple(row[c] for c in key_cols)] = row.to_dict()
    return out


def exact_old_match(targets: pd.DataFrame, old: pd.DataFrame) -> pd.DataFrame:
    out = targets.copy()
    out["old_match_method"] = pd.NA
    out["old_match_score"] = np.nan
    for c in LEGACY_COPY_COLS:
        if c not in out.columns:
            out[c] = pd.NA
    passes = [
        ("exact_iso3_year_eng_share", ["iso3c", "election_year", "partyname_eng_norm", "share_key"], ["old_iso3c", "old_election_year", "old_partyname_eng_norm", "old_share_key"]),
        ("exact_iso3_year_local_share", ["iso3c", "election_year", "partyname_norm", "share_key"], ["old_iso3c", "old_election_year", "old_partyname_norm", "old_share_key"]),
        ("exact_iso3_year_eng", ["iso3c", "election_year", "partyname_eng_norm"], ["old_iso3c", "old_election_year", "old_partyname_eng_norm"]),
        ("exact_iso3_year_local", ["iso3c", "election_year", "partyname_norm"], ["old_iso3c", "old_election_year", "old_partyname_norm"]),
    ]
    for method, tcols, ocols in passes:
        lookup = unique_key_lookup(old, ocols)
        need = out["old_rowid"].isna()
        if not lookup or not need.any():
            continue
        for idx, row in out.loc[need, tcols].iterrows():
            key = tuple(row[c] for c in tcols)
            hit = lookup.get(key)
            if hit is None:
                continue
            out.at[idx, "old_match_method"] = method
            out.at[idx, "old_match_score"] = 100.0
            for c in LEGACY_COPY_COLS:
                out.at[idx, c] = hit.get(c)
    return out


def fuzzy_old_match(targets: pd.DataFrame, old: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = targets.copy()
    manual_rows: list[dict[str, Any]] = []
    need = out["old_rowid"].isna()
    if not need.any():
        return out, pd.DataFrame()

    old_by_group: dict[tuple[str, int], pd.DataFrame] = {}
    for (iso3c, year), g in old.groupby(["old_iso3c", "old_election_year"], dropna=True):
        if clean_text_scalar(iso3c) == "" or pd.isna(year):
            continue
        old_by_group[(str(iso3c), int(year))] = g.copy().reset_index(drop=True)

    for idx, row in out.loc[need].iterrows():
        iso3 = clean_text_scalar(row.get("iso3c"))
        year = row.get("election_year")
        if not iso3 or pd.isna(year):
            continue
        cand = old_by_group.get((iso3, int(year)))
        if cand is None or cand.empty:
            continue

        if row["share_key"] is not None and not pd.isna(row["share_key"]):
            diff = (cand["old_share_mpd"] - float(row["share_key"])).abs()
            cand_pool = cand.loc[(diff <= SHARE_TOLERANCE) | diff.isna()].copy()
            if cand_pool.empty:
                cand_pool = cand.copy()
        else:
            cand_pool = cand.copy()

        scores: list[tuple[float, int, dict[str, Any], float | None]] = []
        t_local = row["partyname_norm"]
        t_eng = row["partyname_eng_norm"]
        t_combo = " ".join([x for x in [t_local, t_eng] if x]).strip()
        for j, crow in cand_pool.iterrows():
            c_local = crow["old_partyname_norm"]
            c_eng = crow["old_partyname_eng_norm"]
            c_combo = " ".join([x for x in [c_local, c_eng] if x]).strip()
            name_score = max(
                sequence_similarity(t_local, c_local),
                sequence_similarity(t_eng, c_eng),
                sequence_similarity(t_combo, c_combo),
                sequence_similarity(t_local, c_eng),
                sequence_similarity(t_eng, c_local),
            )
            share_diff = None
            if row["share_key"] is not None and not pd.isna(row["share_key"]) and pd.notna(crow["old_share_mpd"]):
                share_diff = abs(float(row["share_key"]) - float(crow["old_share_mpd"]))
                score = name_score - min(share_diff, 5.0)
            else:
                score = name_score
            scores.append((score, j, crow.to_dict(), share_diff))
        if not scores:
            continue
        scores = sorted(scores, key=lambda x: x[0], reverse=True)
        best_score, _, best_row, best_share_diff = scores[0]
        second_score = scores[1][0] if len(scores) > 1 else -np.inf
        if best_score >= FUZZY_ACCEPT_SCORE and (best_score - second_score) >= FUZZY_MARGIN:
            out.at[idx, "old_match_method"] = "fuzzy_iso3_year_name"
            out.at[idx, "old_match_score"] = float(best_score)
            for c in LEGACY_COPY_COLS:
                out.at[idx, c] = best_row.get(c)
        else:
            manual_rows.append({
                "target_id": row["target_id"],
                "party_election_id": row.get("party_election_id"),
                "iso3c": iso3,
                "election_year": int(year),
                "partyname": row.get("partyname"),
                "partyname_eng": row.get("partyname_eng"),
                "share_mpd": row.get("share_mpd"),
                "best_score": float(best_score),
                "second_score": float(second_score) if np.isfinite(second_score) else np.nan,
                "best_share_diff": best_share_diff,
                "suggested_old_rowid": best_row.get("old_rowid"),
                "suggested_old_partyname": best_row.get("old_partyname"),
                "suggested_old_partyname_eng": best_row.get("old_partyname_eng"),
                "suggested_old_share_mpd": best_row.get("old_share_mpd"),
                "suggested_old_quality_class": best_row.get("old_quality_class"),
            })
    return out, pd.DataFrame(manual_rows)


def build_target_master_from_seed(use_legacy: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = read_seed_table().copy()
    manual = pd.DataFrame()

    # Ensure essential columns exist.
    for c in ["country", "iso2c", "iso3c", "election_year", "edate_mpd", "mpd_party_id", "parfam", "partyname", "partyname_eng", "share_mpd", "share_key", "partyname_norm", "partyname_eng_norm", "party_election_id", "target_id"]:
        if c not in targets.columns:
            targets[c] = pd.NA

    if use_legacy and LEGACY_CSV_PATH.exists():
        old = load_legacy_manual_csv(LEGACY_CSV_PATH)
        targets = exact_old_match(targets, old)
        targets, manual = fuzzy_old_match(targets, old)
    else:
        # Create empty legacy columns so downstream code stays simple.
        for c in LEGACY_COPY_COLS + ["old_match_method", "old_match_score"]:
            targets[c] = pd.NA

    def reason(row: pd.Series) -> str:
        if pd.isna(row.get("old_rowid")):
            return "collect_missing_old_data"
        q = clean_text_scalar(row.get("old_quality_class"))
        if q == "clean_old_data":
            return "audit_old_clean_single_person"
        if q == "ambiguous_old_data":
            return "audit_old_ambiguous_or_multiperson"
        if q == "inconsistent_old_data":
            return "audit_old_internally_inconsistent"
        return "audit_old_other"

    targets["queue_reason"] = targets.apply(reason, axis=1)
    targets["old_candidate_age"] = targets.get("old_candidate_age_legacy")
    targets["old_candidate_gender"] = targets.get("old_candidate_gender_legacy")
    return targets, manual


# =============================================================================
# OpenAI helpers
# =============================================================================
def make_user_prompt(row: pd.Series, include_legacy_hints: bool = False) -> str:
    country = clean_text_scalar(row.get("country"))
    iso2c = clean_text_scalar(row.get("iso2c"))
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
    local_aliases = clean_text_scalar(row.get("local_aliases"))
    eng_aliases = clean_text_scalar(row.get("eng_aliases"))
    if local_aliases:
        aliases.append(f"- local_aliases: {local_aliases}")
    if eng_aliases:
        aliases.append(f"- english_aliases: {eng_aliases}")
    alias_block = "\n".join(aliases)

    prompt = f"""
Target party-election record
- target_id: {row['target_id']}
- party_election_id: {row.get('party_election_id')}
- country: {country}
- iso2c: {iso2c}
- iso3c: {iso3c}
- election_year: {ey}
- election_date: {ed}
- mpd_party_id: {mpd_party_id}
- party_family: {parfam}
- party_name_local_or_primary: {party_local}
- party_name_english: {party_eng}
- vote_share_mpd_percent: {share_txt}
{alias_block}

Task
1) Decide who most clearly led this party's NATIONAL campaign for this election date.
2) Prefer the nationally designated lead candidate / campaign leader / Spitzenkandidat / public campaign face.
3) Use the formal party president only if no distinct campaign leader is evident.
4) If reliable sources show joint or collective national campaign leadership, return all clearly relevant leaders in `leaders` and mark include_in_average=true for each one.
5) For alliances / coalitions, identify the national face(s) of the alliance for this election, not merely a constituent party officer.
6) Use as few web searches as needed and stop once the evidence is sufficient.
7) Use at most 4 sources total.

Output rules
- If there is one clear leader, fill the top-level leader fields and also include that same person in `leaders`.
- If there are multiple relevant leaders, set is_multiple_leader_case=true and use `leaders` for the people who should be averaged.
- If leadership is collective and no single leader dominates, set is_collective_leadership=true.
- If unresolved, set resolved=false and explain briefly.
- Never guess.
- Return JSON only.
""".strip()

    if include_legacy_hints and pd.notna(row.get("old_rowid")):
        legacy_urls = [clean_text_scalar(row.get(c)) for c in ["old_wiki_url", "old_url1", "old_url2", "old_url3"]]
        legacy_urls = [u for u in legacy_urls if u]
        prompt += f"""

Legacy handwritten notes (possible clues only; verify independently because they may be wrong)
- old_leader_name_raw: {clean_text_scalar(row.get('old_leader_name_raw'))}
- old_role_raw: {clean_text_scalar(row.get('old_role_raw'))}
- old_age_raw: {row.get('old_age_raw')}
- old_gender_raw: {clean_text_scalar(row.get('old_gender_raw'))}
- old_year_born_raw: {clean_text_scalar(row.get('old_year_born_raw'))}
- legacy_urls: {' | '.join(legacy_urls)}
"""
    return prompt


def build_response_request_body(row: pd.Series, model: str, include_legacy_hints: bool = False) -> dict[str, Any]:
    return {
        "model": model,
        "store": False,
        "tools": [WEB_SEARCH_TOOL],
        "tool_choice": "auto",
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_user_prompt(row, include_legacy_hints=include_legacy_hints)},
        ],
        "text": {
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": "party_leader_record",
                "schema": PARTY_LEADER_SCHEMA,
                "strict": True,
            },
        },
    }


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


def json_dumps_compact(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"), sort_keys=False)


def write_batch_jsonl(targets: pd.DataFrame, out_path: Path, model: str, include_legacy_hints: bool = False) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for _, row in targets.iterrows():
            req = {
                "custom_id": row["target_id"],
                "method": "POST",
                "url": "/v1/responses",
                "body": build_response_request_body(row, model=model, include_legacy_hints=include_legacy_hints),
            }
            f.write(json_dumps_compact(req) + "\n")


def collect_with_openai(targets: pd.DataFrame, out_jsonl: Path, out_csv: Path, model: str, sleep_seconds: float = 0.0, include_legacy_hints: bool = False) -> None:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("The `openai` package is required. Install it with `pip install openai`.") from exc

    client = OpenAI()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    completed: set[str] = set()
    if out_csv.exists():
        try:
            flat_existing = pd.read_csv(out_csv)
            if "target_id" in flat_existing.columns:
                completed.update(flat_existing["target_id"].astype(str).tolist())
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
        for _, row in targets.iterrows():
            tid = str(row["target_id"])
            if tid in completed:
                continue

            body = build_response_request_body(row, model=model, include_legacy_hints=include_legacy_hints)
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
                rec = {"target_id": tid, "status": "error", "error": repr(last_err)}
                fout.write(json_dumps_compact(rec) + "\n")
                fout.flush()
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
                continue

            output_text = getattr(resp_obj, "output_text", None)
            if output_text is None:
                rec = {
                    "target_id": tid,
                    "status": "no_output_text",
                    "usage": extract_response_usage(resp_obj),
                    "tool_calls_web_search": count_web_search_calls(resp_obj),
                }
                fout.write(json_dumps_compact(rec) + "\n")
                fout.flush()
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
                continue

            try:
                parsed = json.loads(output_text)
            except Exception as exc:
                rec = {
                    "target_id": tid,
                    "status": "invalid_json",
                    "error": repr(exc),
                    "output_text": output_text,
                    "usage": extract_response_usage(resp_obj),
                    "tool_calls_web_search": count_web_search_calls(resp_obj),
                }
                fout.write(json_dumps_compact(rec) + "\n")
                fout.flush()
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
                continue

            agg = aggregate_leader_metrics(parsed, row.get("edate_mpd"))
            usage = extract_response_usage(resp_obj)
            tool_calls_web_search = count_web_search_calls(resp_obj)
            leader_name_value = agg.get("leader_name_joined") or clean_text_scalar(parsed.get("leader_name")) or agg.get("leader_name_primary")

            flat = {
                "target_id": tid,
                "party_election_id": clean_text_scalar(row.get("party_election_id")) or tid,
                "status": "ok",
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
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
    flush_flat()


# =============================================================================
# Merge + summaries
# =============================================================================
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


def merge_results(targets_path: Path, ai_results_csv: Path, out_csv: Path, out_dta: Path | None = None) -> pd.DataFrame:
    parse_cols = ["edate_mpd"] if "edate_mpd" in pd.read_csv(targets_path, nrows=0).columns else None
    targets = pd.read_csv(targets_path, parse_dates=parse_cols)
    ai = pd.read_csv(ai_results_csv)
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


def summarize_collect_results(ai_csv: Path) -> None:
    if not ai_csv.exists():
        return
    df = pd.read_csv(ai_csv)
    if df.empty:
        print("No collected rows yet.")
        return
    ok = (df.get("status") == "ok").sum() if "status" in df.columns else 0
    multi = (df.get("is_multiple_leader_case") == True).sum() if "is_multiple_leader_case" in df.columns else 0
    collective = (df.get("is_collective_leadership") == True).sum() if "is_collective_leadership" in df.columns else 0
    miss_age = df.get("candidate_age_ai", pd.Series(np.nan, index=df.index)).isna().sum()
    miss_gen = df.get("candidate_gender_ai", pd.Series(np.nan, index=df.index)).isna().sum()
    conf = pd.to_numeric(df.get("confidence", pd.Series(np.nan, index=df.index)), errors="coerce")
    web_calls = pd.to_numeric(df.get("tool_calls_web_search", pd.Series(np.nan, index=df.index)), errors="coerce")
    print("\nCollection summary:")
    print(f"- rows in file: {len(df)}")
    print(f"- status=ok: {int(ok)}")
    print(f"- multiple leader cases: {int(multi)}")
    print(f"- collective leadership cases: {int(collective)}")
    print(f"- missing AI age: {int(miss_age)}")
    print(f"- missing AI gender: {int(miss_gen)}")
    if conf.notna().any():
        print(f"- mean confidence: {conf.mean():.4f}")
        print(f"- median confidence: {conf.median():.4f}")
    if web_calls.notna().any():
        print(f"- mean web searches per row: {web_calls.mean():.3f}")
        print(f"- total web searches: {int(web_calls.sum())}")


# =============================================================================
# Spyder runners
# =============================================================================
def _set_openai_key_for_session() -> None:
    key = clean_text_scalar(OPENAI_API_KEY)
    if key:
        os.environ["OPENAI_API_KEY"] = key


def run_prep_step() -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    targets, manual = build_target_master_from_seed(use_legacy=bool(USE_LEGACY_CSV_FOR_HINTS_AND_QUEUES))
    write_csv_and_dta(targets, TARGETS_CSV, TARGETS_DTA)
    if not manual.empty:
        manual.to_csv(MANUAL_REVIEW_CSV, index=False)
    write_batch_jsonl(targets, BATCH_JSONL, model=MODEL, include_legacy_hints=bool(INCLUDE_LEGACY_HINTS))
    print(f"Saved target master: {TARGETS_CSV}")
    print(f"Saved target master Stata file: {TARGETS_DTA}")
    print(f"Saved batch request file: {BATCH_JSONL}")
    if not manual.empty:
        print(f"Saved manual review suggestions: {MANUAL_REVIEW_CSV}")
    print("\nQueue counts:")
    print(targets["queue_reason"].value_counts(dropna=False).to_string())
    return targets


def run_collect_step() -> pd.DataFrame:
    _set_openai_key_for_session()
    if not TARGETS_CSV.exists():
        raise FileNotFoundError(f"Missing {TARGETS_CSV}. Run the prep step first.")
    parse_cols = ["edate_mpd"] if "edate_mpd" in pd.read_csv(TARGETS_CSV, nrows=0).columns else None
    targets = pd.read_csv(TARGETS_CSV, parse_dates=parse_cols)
    if ONLY_QUEUE_REASONS:
        targets = targets[targets["queue_reason"].isin(list(ONLY_QUEUE_REASONS))].copy()
    if PILOT_LIMIT is not None:
        targets = targets.head(int(PILOT_LIMIT)).copy()
    print(f"Rows to collect this run: {len(targets)}")
    if len(targets) == 0:
        print("Nothing to collect.")
        return targets
    collect_with_openai(targets, out_jsonl=AI_RESULTS_JSONL, out_csv=AI_RESULTS_CSV, model=MODEL, sleep_seconds=float(SLEEP_SECONDS), include_legacy_hints=bool(INCLUDE_LEGACY_HINTS))
    print(f"Saved AI results JSONL: {AI_RESULTS_JSONL}")
    print(f"Saved AI results CSV: {AI_RESULTS_CSV}")
    summarize_collect_results(AI_RESULTS_CSV)
    return pd.read_csv(AI_RESULTS_CSV)


def run_merge_step() -> pd.DataFrame:
    if not TARGETS_CSV.exists():
        raise FileNotFoundError(f"Missing {TARGETS_CSV}. Run the prep step first.")
    if not AI_RESULTS_CSV.exists():
        raise FileNotFoundError(f"Missing {AI_RESULTS_CSV}. Run the collect step first.")
    out = merge_results(TARGETS_CSV, AI_RESULTS_CSV, FINAL_CSV, FINAL_DTA)
    print(f"Saved merged file: {FINAL_CSV}")
    print(f"Saved merged Stata file: {FINAL_DTA}")
    print("\nFinal status counts:")
    print(out["final_status"].value_counts(dropna=False).to_string())
    return out


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("Starting party leader AI pipeline from PyBLP backbone in Spyder mode...")
    print(f"Base directory: {BASE_DIR}")
    print(f"Backbone directory: {BACKBONE_DIR}")
    print(f"Seed input: {SEED_DTA if SEED_DTA.exists() else SEED_CSV}")

    if RUN_PREP:
        print("\n--- PREP STEP ---")
        run_prep_step()
    if RUN_COLLECT:
        print("\n--- COLLECT STEP ---")
        run_collect_step()
    if RUN_MERGE:
        print("\n--- MERGE STEP ---")
        run_merge_step()
    if not any([RUN_PREP, RUN_COLLECT, RUN_MERGE]):
        print("No step selected. Set at least one of RUN_PREP, RUN_COLLECT, RUN_MERGE to True.")
