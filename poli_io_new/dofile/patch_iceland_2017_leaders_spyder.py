#!/usr/bin/env python3
from __future__ import annotations

"""
Spyder-ready patch for missing Iceland 2017 party leader characteristics.

Why this exists
---------------
A few market-party products remain missing candidate_age / candidate_gender after
merging the AI-built party-characteristics file back into the PyBLP backbone.
This patch fills the two Iceland 2017 party-election IDs that are easy to verify:

- pe000636 : Left-Green Movement (Iceland, 2017) -> Katrín Jakobsdóttir
- pe000640 : Independence Party (Iceland, 2017) -> Bjarni Benediktsson

The patch writes *_patched copies by default and does NOT overwrite originals.
It can patch any of these files if they exist:

- ai_results_campaignavg_gpt54_from_backbone.csv
- ai_results_campaignavg_gpt54_from_backbone_repaired.csv
- party_char_v3_campaignavg.csv / .dta
- party_char_v3_campaignavg_repaired.csv / .dta
- auto_regression_pyblp_ready_with_char.dta

Facts used
----------
- 2017 Icelandic parliamentary election date: 2017-10-28
- Left-Green leader in that election: Katrín Jakobsdóttir
- Independence Party leader in that election: Bjarni Benediktsson
- Katrín Jakobsdóttir DOB: 1976-02-01
- Bjarni Benediktsson DOB: 1970-01-26

The patch computes candidate_age at the election date and fills:
- candidate_age
- candidate_gender           (BLP-compatible scale: female=1, male=0)
- candidate_gender_male1     (requested scale: female=0, male=1)

For AI-style files it also fills the AI-specific fields when those columns exist.
"""

from pathlib import Path
import json
import re
from typing import Any

import numpy as np
import pandas as pd

# =============================================================================
# SETTINGS
# =============================================================================
BASE_DIR = Path("/Users/zl279/Downloads/Poli-IO/raw_data")
BACKBONE_DIR = BASE_DIR / "pyblp_backbone"

WRITE_PATCHED_COPIES = True   # recommended
OVERWRITE_ORIGINALS = False   # set True only if you want in-place modification

REPORT_CSV = BACKBONE_DIR / "iceland_2017_patch_report.csv"
REPORT_JSON = BACKBONE_DIR / "iceland_2017_patch_report.json"

TARGET_FILES = [
    BACKBONE_DIR / "ai_results_campaignavg_gpt54_from_backbone.csv",
    BACKBONE_DIR / "ai_results_campaignavg_gpt54_from_backbone_repaired.csv",
    BACKBONE_DIR / "party_char_v3_campaignavg.csv",
    BACKBONE_DIR / "party_char_v3_campaignavg.dta",
    BACKBONE_DIR / "party_char_v3_campaignavg_repaired.csv",
    BACKBONE_DIR / "party_char_v3_campaignavg_repaired.dta",
    BACKBONE_DIR / "auto_regression_pyblp_ready_with_char.dta",
]

PATCH_SPECS = {
    "pe000636": {
        "country": "Iceland",
        "iso3c": "ISL",
        "election_year": 2017,
        "partyname_eng": "Left-Green Movement",
        "leader_name": "Katrín Jakobsdóttir",
        "leader_role": "campaign leader / party chair",
        "leader_role_basis": "party leader and national campaign leader in 2017 election",
        "birth_date_iso": "1976-02-01",
        "birth_year": 1976,
        "birth_precision": "exact_date",
        "gender_text": "female",
        "candidate_gender": 1.0,          # female=1, male=0 (legacy / BLP-compatible)
        "candidate_gender_male1": 0.0,    # female=0, male=1
        "confidence": 0.99,
    },
    "pe000640": {
        "country": "Iceland",
        "iso3c": "ISL",
        "election_year": 2017,
        "partyname_eng": "Independence Party",
        "leader_name": "Bjarni Benediktsson",
        "leader_role": "campaign leader / party chair",
        "leader_role_basis": "party leader and national campaign leader in 2017 election",
        "birth_date_iso": "1970-01-26",
        "birth_year": 1970,
        "birth_precision": "exact_date",
        "gender_text": "male",
        "candidate_gender": 0.0,          # female=1, male=0
        "candidate_gender_male1": 1.0,    # female=0, male=1
        "confidence": 0.99,
    },
}

ELECTION_DATE = pd.Timestamp("2017-10-28")
SOURCE_NOTE = (
    "Patched manually from official Alþingi member pages and the 2017 Icelandic "
    "parliamentary election page."
)

SOURCE_LIST = [
    {
        "title": "2017 Icelandic parliamentary election",
        "url": "https://en.wikipedia.org/wiki/2017_Icelandic_parliamentary_election",
        "publisher": "Wikipedia",
        "why_relevant": "Confirms election date and lists party leaders for the 2017 election."
    },
    {
        "title": "Katrín Jakobsdóttir | Members of Parliament",
        "url": "https://www.althingi.is/altext/cv/en/?ksfaerslunr=109",
        "publisher": "Alþingi",
        "why_relevant": "Official parliamentary profile with party affiliation and date of birth."
    },
    {
        "title": "Bjarni Benediktsson | Members of Parliament",
        "url": "https://www.althingi.is/altext/cv/en/?ksfaerslunr=40",
        "publisher": "Alþingi",
        "why_relevant": "Official parliamentary profile with party affiliation and date of birth."
    },
]

# =============================================================================
# Helpers
# =============================================================================
def clean_text_scalar(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).replace("\xa0", " ").strip()
    return "" if s.lower() in {"", "nan", "none", "<na>"} else s


def normalize_text_scalar(x: Any) -> str:
    s = clean_text_scalar(x).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def age_on_date(birth_date_iso: str, on_date: pd.Timestamp) -> float:
    bd = pd.Timestamp(birth_date_iso)
    years = on_date.year - bd.year
    had_birthday = (on_date.month, on_date.day) >= (bd.month, bd.day)
    return float(years if had_birthday else years - 1)


def cast_stata_friendly(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if pd.api.types.is_object_dtype(s.dtype):
            non_na = s.dropna()
            if len(non_na) == 0:
                out[c] = pd.Series([pd.NA] * len(s), index=s.index, dtype="string")
            elif non_na.map(lambda x: isinstance(x, str)).all():
                out[c] = s.astype("string")
            elif non_na.map(lambda x: isinstance(x, (bool, np.bool_))).all():
                out[c] = s.astype("boolean")
            else:
                maybe_num = pd.to_numeric(s, errors="coerce")
                if maybe_num.notna().sum() == non_na.shape[0]:
                    out[c] = maybe_num
                else:
                    out[c] = s.map(lambda x: pd.NA if pd.isna(x) else str(x)).astype("string")
    return out


def write_stata_safe(df: pd.DataFrame, path: Path) -> None:
    out = cast_stata_friendly(df)
    out.to_stata(path, write_index=False, version=118)


def read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".dta":
        return pd.read_stata(path, convert_categoricals=False)
    return pd.read_csv(path)


def write_like(df: pd.DataFrame, src_path: Path, dst_path: Path) -> None:
    if dst_path.suffix.lower() == ".dta":
        write_stata_safe(df, dst_path)
    else:
        df.to_csv(dst_path, index=False)


def build_patch_payloads() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for peid, spec in PATCH_SPECS.items():
        age = age_on_date(spec["birth_date_iso"], ELECTION_DATE)
        leaders_json = json.dumps([
            {
                "full_name": spec["leader_name"],
                "role": spec["leader_role"],
                "role_basis": spec["leader_role_basis"],
                "birth_date_iso": spec["birth_date_iso"],
                "birth_year": spec["birth_year"],
                "birth_precision": spec["birth_precision"],
                "gender_text": spec["gender_text"],
                "include_in_average": True,
            }
        ], ensure_ascii=False)
        sources_json = json.dumps(SOURCE_LIST, ensure_ascii=False)
        out[peid] = {
            **spec,
            "candidate_age": age,
            "candidate_age_ai": age,
            "candidate_age_method": "exact_birth_date_at_election",
            "candidate_gender_ai": spec["candidate_gender"],
            "candidate_gender_ai_male1": spec["candidate_gender_male1"],
            "resolved": True,
            "resolution_status": "resolved_manual_patch",
            "status": "ok",
            "is_collective_leadership": False,
            "is_multiple_leader_case": False,
            "multiple_leader_basis": "",
            "leader_count_total": 1,
            "leader_count_age_nonmissing": 1,
            "leader_count_gender_nonmissing": 1,
            "notes": SOURCE_NOTE,
            "sources_json": sources_json,
            "leaders_json": leaders_json,
            "final_status": "manual_patch",
        }
    return out


def mask_for_peid(df: pd.DataFrame, peid: str, spec: dict[str, Any]) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    if "party_election_id" in df.columns:
        mask = mask | (df["party_election_id"].astype(str) == peid)
    # Fallback matching in case some file lacks party_election_id.
    fallback = pd.Series(True, index=df.index)
    used = False
    if "iso3c" in df.columns:
        fallback &= df["iso3c"].astype(str).str.upper().eq(spec["iso3c"]) 
        used = True
    if "country" in df.columns:
        fallback &= df["country"].map(normalize_text_scalar).eq(normalize_text_scalar(spec["country"]))
        used = True
    if "election_year" in df.columns:
        fallback &= pd.to_numeric(df["election_year"], errors="coerce").round().astype("Int64").eq(int(spec["election_year"]))
        used = True
    if "partyname_eng" in df.columns:
        fallback &= df["partyname_eng"].map(normalize_text_scalar).eq(normalize_text_scalar(spec["partyname_eng"]))
        used = True
    if used:
        mask = mask | fallback
    return mask


def patch_frame(df: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    out = df.copy()
    payloads = build_patch_payloads()
    rows: list[dict[str, Any]] = []

    for peid, payload in payloads.items():
        mask = mask_for_peid(out, peid, payload)
        n = int(mask.sum())
        rows.append({
            "file": str(path),
            "party_election_id": peid,
            "matched_rows": n,
        })
        if n == 0:
            continue

        # General fields that are safe to write only when present.
        for col, val in payload.items():
            if col in out.columns:
                out.loc[mask, col] = val

        # Keep an extra human-readable source note when possible.
        if "patch_note" in out.columns:
            out.loc[mask, "patch_note"] = SOURCE_NOTE

    return out, rows


def out_path_for(src: Path) -> Path:
    if OVERWRITE_ORIGINALS:
        return src
    stem = src.stem + "_patched"
    return src.with_name(stem + src.suffix)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    print("Starting Iceland 2017 leader patch...")
    print(f"Backbone directory: {BACKBONE_DIR}")
    report_rows: list[dict[str, Any]] = []

    for src in TARGET_FILES:
        if not src.exists():
            continue
        print(f"\nPatching: {src.name}")
        df = read_any(src)
        patched, rows = patch_frame(df, src)
        dst = out_path_for(src) if WRITE_PATCHED_COPIES or OVERWRITE_ORIGINALS else src
        write_like(patched, src, dst)
        print(f"Saved: {dst}")
        report_rows.extend([{**r, "saved_to": str(dst)} for r in rows])

    rep = pd.DataFrame(report_rows)
    if not rep.empty:
        rep.to_csv(REPORT_CSV, index=False)
        with open(REPORT_JSON, "w", encoding="utf-8") as f:
            json.dump(report_rows, f, ensure_ascii=False, indent=2)
        print(f"\nSaved report: {REPORT_CSV}")

        # Compact summary by file.
        print("\nPatch summary:")
        grp = rep.groupby("saved_to", as_index=False)["matched_rows"].sum()
        for _, row in grp.iterrows():
            print(f"- {row['saved_to']}: {int(row['matched_rows'])} matched rows")
    else:
        print("No target files found to patch.")

    print("Done.")


if __name__ == "__main__":
    main()
