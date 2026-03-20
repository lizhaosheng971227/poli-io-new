#!/usr/bin/env python3
from __future__ import annotations

"""
Spyder-ready builder for PyBLP backbone files from the ESS–MPD validated merge.

Purpose
-------
This script creates a clean backbone for the current `pyblp_260305.py` workflow,
while fixing the main issues in Pablo's ESS–MPD construction for *your* use case:

1) It uses `anweight` consistently for the BLP respondent backbone and demo file.
2) It does NOT use Pablo's custom respondent-level econ/culture indices. Instead,
   it preserves the raw MPD `per*` variables so your existing Python demand code
   can reconstruct indices from the sign spreadsheet exactly as before.
3) It requires a real geographic region variable (not `domicil`) because markets
   in the current BLP setup are defined as `(region4, election_year)`.
4) It creates a stable `party_election_id` and a deduplicated party-election table
   that can be used as the canonical bridge to the AI leader-characteristics flow
   and later merged back into the main respondent file.
5) It intentionally does NOT merge party characteristics here. That keeps the
   backbone clean and avoids blocking on the leader-collection step.

Outputs
-------
In `OUTPUT_DIR`, the script writes:

- auto_regression_pyblp_backbone.dta
    Respondent-level backbone for the BLP pipeline.

- demo_file_pyblp_backbone.dta
    Agent / demographic file for PyBLP demand and supply IVs.

- party_election_backbone.csv
- party_election_backbone.dta
    Canonical one-row-per-party-election table with stable keys and aliases.

- party_leader_targets_seed.csv
- party_leader_targets_seed.dta
    AI-friendly seed table with the core target columns.

- pyblp_backbone_build_report.csv
- pyblp_backbone_build_report.json
    Diagnostics and counts.

Optional later step
-------------------
When you have a leader file (e.g. from the GPT-5.4 pipeline), set
`RUN_FINALIZE_WITH_PARTY_CHAR = True` and point `PARTY_CHAR_PATH` to a file with
`party_election_id` (preferred) or the stable key columns. The helper will merge
`candidate_age` and `candidate_gender` into the backbone and write
`auto_regression_pyblp_ready_with_char.dta`.
"""

from pathlib import Path
import json
import math
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd


# =============================================================================
# SPYDER SETTINGS
# =============================================================================
BASE_DIR = Path("/Users/zl279/Downloads/Poli-IO/raw_data")
DATA0_DIR = BASE_DIR / "0_data"
PROC_DIR = DATA0_DIR / "proc"
OUTPUT_DIR = BASE_DIR / "pyblp_backbone"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Default validated input. If missing, we fall back to BASE_DIR.
VALIDATED_PATH = PROC_DIR / "ess_mpd_matched_validated.dta"
if not VALIDATED_PATH.exists():
    VALIDATED_PATH = BASE_DIR / "ess_mpd_matched_validated.dta"

# Leave party characteristics out of the backbone by default.
RUN_BUILD_BACKBONE = True
RUN_FINALIZE_WITH_PARTY_CHAR = False
PARTY_CHAR_PATH = OUTPUT_DIR / "party_char_v3_campaignavg.dta"  # edit later if needed

# Backbone output files
AUTO_OUT = OUTPUT_DIR / "auto_regression_pyblp_backbone.dta"
DEMO_OUT = OUTPUT_DIR / "demo_file_pyblp_backbone.dta"
PARTY_ELECTION_CSV = OUTPUT_DIR / "party_election_backbone.csv"
PARTY_ELECTION_DTA = OUTPUT_DIR / "party_election_backbone.dta"
AI_SEED_CSV = OUTPUT_DIR / "party_leader_targets_seed.csv"
AI_SEED_DTA = OUTPUT_DIR / "party_leader_targets_seed.dta"
REPORT_CSV = OUTPUT_DIR / "pyblp_backbone_build_report.csv"
REPORT_JSON = OUTPUT_DIR / "pyblp_backbone_build_report.json"
FINAL_AUTO_WITH_CHAR = OUTPUT_DIR / "auto_regression_pyblp_ready_with_char.dta"

# Market / sample settings
KEEP_ONLY_VOTE_1_2 = True
KEEP_UNMATCHED_PARTY_VOTERS = True   # unmatched valid voters help define outside share
AGENTS_PER_MARKET = 200              # matches current pyblp integration size
RANDOM_SEED = 20260307

# Geographic region candidates. IMPORTANT: do not put `domicil` here.
REGION_CANDIDATES = [
    "region", "region_code", "regionname", "regunit", "nuts", "nuts1",
    "nuts2", "nuts3", "essregion", "geo_region"
]

# Columns to try for canonical harmonization
COUNTRY_CANDIDATES = ["country", "cntryname"]
ISO2_CANDIDATES = ["iso2c"]
ISO3_CANDIDATES = ["iso3c"]
ELECTION_YEAR_CANDIDATES = ["election_year", "year"]
EDATE_MPD_CANDIDATES = ["edate_mpd", "election_date", "date_mpd"]
EDATE_ESS_CANDIDATES = ["edate_ess"]
MPD_PARTY_ID_CANDIDATES = ["mpd_party_id", "party_id_mpd", "party"]
PARFAM_CANDIDATES = ["parfam"]
PARTYNAME_CANDIDATES = ["partyname", "partyname_mpd", "ess_party", "partyname_ess"]
PARTYNAME_ENG_CANDIDATES = ["partyname_eng"]
SHARE_MPD_CANDIDATES = ["share_mpd", "pervote"]
VOTE_CANDIDATES = ["vote"]
LRSCALE_CANDIDATES = ["lrscale", "leftright_scale"]
AGE_CANDIDATES = ["age", "agea"]
DOB_CANDIDATES = ["dob", "yrbrn"]
FEMALE_CANDIDATES = ["female"]


# =============================================================================
# Helpers
# =============================================================================
def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def clean_text_scalar(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x)
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_text(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("").str.replace("\xa0", " ", regex=False).str.strip()


def ensure_datetime(s: pd.Series) -> pd.Series:
    # pandas read_stata often already preserves dates; this is a safe fallback.
    try:
        out = pd.to_datetime(s, errors="coerce")
    except Exception:
        out = pd.to_datetime(pd.Series(s), errors="coerce")
    return out


def first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_election(region: pd.Series, election_year: pd.Series) -> pd.Series:
    reg4 = normalize_text(region).str[:4]
    ey = pd.to_numeric(election_year, errors="coerce").round().astype("Int64").astype("string")
    return (reg4 + ey).str.lower().str.replace(" ", "", regex=False)


def safe_round_share(x: Any, digits: int = 3) -> float | None:
    try:
        x = float(x)
    except Exception:
        return None
    if not np.isfinite(x):
        return None
    return round(x, digits)


def looks_generic_party_label(x: Any) -> bool:
    s = clean_text_scalar(x).lower()
    if not s:
        return True
    generic = {
        "alliance", "coalition", "list", "independent", "others",
        "other", "party", "movement", "front"
    }
    return s in generic


def norm_name_scalar(x: Any) -> str:
    s = clean_text_scalar(x).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def cast_stata_friendly(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_bool_dtype(out[c]):
            out[c] = out[c].astype(float)
        elif str(out[c].dtype) == "Int64":
            out[c] = out[c].astype(float)
    return out


def weighted_resample_market(g: pd.DataFrame, n_draws: int, weight_col: str, rng: np.random.Generator) -> pd.DataFrame:
    g = g.reset_index(drop=True).copy()
    if g.empty:
        return g
    w = to_num(g[weight_col]).to_numpy(dtype=float) if weight_col in g.columns else np.ones(len(g), dtype=float)
    valid = np.isfinite(w) & (w > 0)
    if not valid.any():
        probs = np.repeat(1.0 / len(g), len(g))
    else:
        w = np.where(valid, w, 0.0)
        probs = w / w.sum()
    draw_idx = rng.choice(np.arange(len(g)), size=int(n_draws), replace=True, p=probs)
    out = g.iloc[draw_idx].copy().reset_index(drop=True)
    out["agent_ids"] = np.arange(1, len(out) + 1, dtype=int)
    return out


def derive_weights(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    psp = to_num(df["pspwght"]) if "pspwght" in df.columns else pd.Series(np.nan, index=df.index)
    pwt = to_num(df["pweight"]) if "pweight" in df.columns else pd.Series(np.nan, index=df.index)
    anw = to_num(df["anweight"]) if "anweight" in df.columns else pd.Series(np.nan, index=df.index)
    constructed = psp * pwt
    anw = anw.where(anw.notna(), constructed)
    return anw, psp, pwt


def derive_age(df: pd.DataFrame) -> pd.Series:
    age = pd.Series(np.nan, index=df.index, dtype=float)
    c = first_existing(df, AGE_CANDIDATES)
    if c is not None:
        age = to_num(df[c])
    if "yrbrn" in df.columns:
        year = to_num(df["year"]) if "year" in df.columns else pd.Series(np.nan, index=df.index)
        age = age.where(age.notna(), year - to_num(df["yrbrn"]))
    return age


def derive_dob(df: pd.DataFrame, age: pd.Series) -> pd.Series:
    dob = pd.Series(np.nan, index=df.index, dtype=float)
    c = first_existing(df, DOB_CANDIDATES)
    if c is not None:
        dob = to_num(df[c])
    if "year" in df.columns:
        year = to_num(df["year"])
        dob = dob.where(dob.notna(), year - age)
    return dob


def derive_female(df: pd.DataFrame) -> pd.Series:
    if "female" in df.columns:
        out = to_num(df["female"])
    else:
        out = pd.Series(np.nan, index=df.index, dtype=float)
    if "gndr" in df.columns:
        gndr = to_num(df["gndr"])
        out = out.where(out.notna(), np.where(gndr.notna(), (gndr == 2).astype(float), np.nan))
    if "sex" in df.columns:
        sex = to_num(df["sex"])
        out = out.where(out.notna(), np.where(sex.notna(), (sex == 2).astype(float), np.nan))
    return out


def derive_educ_cat(df: pd.DataFrame) -> pd.Series:
    # Ports the main Stata clean_for_reg education logic enough for current BLP use.
    if "educ_cat" in df.columns:
        educ_cat = to_num(df["educ_cat"])
        if educ_cat.notna().any():
            return educ_cat

    educ = pd.Series(np.nan, index=df.index, dtype=float)
    if "edulvla" in df.columns:
        educ = educ.where(educ.notna(), to_num(df["edulvla"]))

    if "edulvlb" in df.columns:
        edb = to_num(df["edulvlb"])
        mapping = {
            0: 1, 113: 1,
            129: 2, 212: 2, 213: 2, 221: 2, 222: 2, 223: 2,
            229: 3, 311: 3, 312: 3, 313: 3, 321: 3, 322: 3, 323: 3,
            412: 4, 413: 4, 421: 4, 422: 4, 423: 4,
            510: 5, 520: 5, 610: 5, 620: 5, 710: 5, 720: 5, 800: 5,
            5555: 55,
        }
        educ_from_b = edb.map(mapping)
        educ = educ.where(educ.notna(), educ_from_b)

    out = pd.Series(np.nan, index=df.index, dtype=float)
    out = np.where(np.isin(educ, [1, 2]), 1.0, out)
    out = np.where(np.isin(educ, [3, 4]), 2.0, out)
    out = np.where(np.isin(educ, [5]), 3.0, out)
    return pd.Series(out, index=df.index)


def choose_canonical_name(values: Iterable[Any]) -> str:
    vals = [clean_text_scalar(v) for v in values if clean_text_scalar(v) != ""]
    if not vals:
        return ""
    vals = sorted(set(vals), key=lambda x: (looks_generic_party_label(x), -len(x), x.lower()))
    return vals[0]


# =============================================================================
# Core builders
# =============================================================================
def load_and_harmonize(validated_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not validated_path.exists():
        raise FileNotFoundError(f"Validated file not found: {validated_path}")

    raw = pd.read_stata(validated_path, convert_categoricals=False).copy()
    meta: dict[str, Any] = {"validated_path": str(validated_path), "n_input_rows": int(len(raw))}

    df = raw.copy()

    # Canonical identifiers
    c_country = first_existing(df, COUNTRY_CANDIDATES)
    c_iso2 = first_existing(df, ISO2_CANDIDATES)
    c_iso3 = first_existing(df, ISO3_CANDIDATES)
    c_year = first_existing(df, ELECTION_YEAR_CANDIDATES)
    c_edate_mpd = first_existing(df, EDATE_MPD_CANDIDATES)
    c_edate_ess = first_existing(df, EDATE_ESS_CANDIDATES)
    c_pid = first_existing(df, MPD_PARTY_ID_CANDIDATES)
    c_parfam = first_existing(df, PARFAM_CANDIDATES)
    c_party = first_existing(df, PARTYNAME_CANDIDATES)
    c_party_eng = first_existing(df, PARTYNAME_ENG_CANDIDATES)
    c_share = first_existing(df, SHARE_MPD_CANDIDATES)
    c_vote = first_existing(df, VOTE_CANDIDATES)
    c_lr = first_existing(df, LRSCALE_CANDIDATES)
    c_region = first_existing(df, REGION_CANDIDATES)

    if c_year is None or c_pid is None:
        raise ValueError("Validated file must contain election year and MPD party ID columns.")
    if c_region is None:
        raise ValueError(
            "No real geographic region variable found. Edit REGION_CANDIDATES. "
            "Do not substitute 'domicil'."
        )

    meta["region_column_used"] = c_region

    out = pd.DataFrame(index=df.index)
    out["country"] = normalize_text(df[c_country]).replace({"": pd.NA}) if c_country else pd.NA
    out["iso2c"] = normalize_text(df[c_iso2]).str.upper().replace({"": pd.NA}) if c_iso2 else pd.NA
    out["iso3c"] = normalize_text(df[c_iso3]).str.upper().replace({"": pd.NA}) if c_iso3 else pd.NA
    out["election_year"] = to_num(df[c_year]).round().astype("Int64")
    out["edate_mpd"] = ensure_datetime(df[c_edate_mpd]) if c_edate_mpd else pd.NaT
    out["edate_ess"] = ensure_datetime(df[c_edate_ess]) if c_edate_ess else pd.NaT
    out["mpd_party_id"] = to_num(df[c_pid]).round().astype("Int64")
    out["parfam"] = to_num(df[c_parfam]).round().astype("Int64") if c_parfam else pd.Series(pd.NA, index=df.index, dtype="Int64")
    out["partyname"] = normalize_text(df[c_party]).replace({"": pd.NA}) if c_party else pd.NA
    out["partyname_eng"] = normalize_text(df[c_party_eng]).replace({"": pd.NA}) if c_party_eng else pd.NA
    out["share_mpd"] = to_num(df[c_share]) if c_share else np.nan
    out["vote"] = to_num(df[c_vote]) if c_vote else np.nan
    out["lrscale"] = to_num(df[c_lr]) if c_lr else np.nan
    out["region"] = normalize_text(df[c_region]).replace({"": pd.NA})

    if "partyfacts_id" in df.columns:
        out["partyfacts_id"] = normalize_text(df["partyfacts_id"]).replace({"": pd.NA})
    if "manual" in df.columns:
        out["manual"] = to_num(df["manual"])

    # Respondent covariates
    out["age"] = derive_age(df)
    out["dob"] = derive_dob(df, out["age"])
    out["female"] = derive_female(df)
    out["educ_cat"] = derive_educ_cat(df)
    out["college"] = np.where(out["educ_cat"] == 3, 1.0, np.where(out["educ_cat"].notna(), 0.0, np.nan))

    out["age_at_election"] = out["election_year"].astype(float) - out["dob"].astype(float)
    out["young"] = np.where(out["age_at_election"].between(18, 34, inclusive="both"), 1.0,
                            np.where(out["age_at_election"].notna(), 0.0, np.nan))
    out["middle"] = np.where(out["age_at_election"].between(35, 64, inclusive="both"), 1.0,
                             np.where(out["age_at_election"].notna(), 0.0, np.nan))
    out["old"] = np.where(out["age_at_election"] >= 65, 1.0,
                          np.where(out["age_at_election"].notna(), 0.0, np.nan))
    out["boomer"] = np.where(out["dob"].between(1946, 1964, inclusive="both"), 1.0,
                             np.where(out["dob"].notna(), 0.0, np.nan))

    # Weights
    out["anweight"], out["pspwght"], out["pweight"] = derive_weights(df)

    # Keep raw MPD policy content variables for the existing Python index logic.
    per_cols = sorted([c for c in df.columns if re.match(r"^per\d+", c, flags=re.IGNORECASE)])
    for c in per_cols:
        out[c] = to_num(df[c])
    meta["n_per_cols"] = len(per_cols)

    # Derived market labels
    out["region4"] = normalize_text(out["region"]).str[:4]
    out["election"] = normalize_election(out["region"], out["election_year"])

    # Stable merge key used throughout
    stable_iso = out["iso3c"].fillna("")
    stable_year = out["election_year"].astype("Int64").astype("string").fillna("")
    stable_pid = out["mpd_party_id"].astype("Int64").astype("string").fillna("")
    stable_date = out["edate_mpd"].dt.strftime("%Y-%m-%d").fillna("") if pd.api.types.is_datetime64_any_dtype(out["edate_mpd"]) else pd.Series("", index=out.index)
    out["stable_party_election_key"] = stable_iso + "|" + stable_date + "|" + stable_year + "|" + stable_pid

    return out, meta


def build_party_election_table(h: pd.DataFrame) -> pd.DataFrame:
    base = h.copy()
    base = base.dropna(subset=["election_year", "mpd_party_id"]).copy()
    base = base[(base["partyname"].fillna("") != "") | (base["partyname_eng"].fillna("") != "")].copy()

    rows = []
    for stable_key, g in base.groupby("stable_party_election_key", dropna=False, sort=False):
        if clean_text_scalar(stable_key) == "":
            continue
        g = g.copy()
        local_aliases = sorted({clean_text_scalar(x) for x in g["partyname"] if clean_text_scalar(x) != ""})
        eng_aliases = sorted({clean_text_scalar(x) for x in g["partyname_eng"] if clean_text_scalar(x) != ""})

        vote12 = g[g["vote"].isin([1, 2])].copy()
        weighted_all = float(to_num(g["anweight"]).fillna(0).sum())
        weighted_vote12 = float(to_num(vote12["anweight"]).fillna(0).sum())

        share_nonmissing = g["share_mpd"].dropna()
        share_pick = float(share_nonmissing.iloc[0]) if len(share_nonmissing) else np.nan

        row = {
            "stable_party_election_key": stable_key,
            "country": choose_canonical_name(g["country"]),
            "iso2c": choose_canonical_name(g["iso2c"]),
            "iso3c": choose_canonical_name(g["iso3c"]),
            "election_year": g["election_year"].dropna().astype(int).iloc[0] if g["election_year"].notna().any() else pd.NA,
            "edate_mpd": g["edate_mpd"].dropna().iloc[0] if g["edate_mpd"].notna().any() else pd.NaT,
            "parfam": g["parfam"].dropna().iloc[0] if g["parfam"].notna().any() else pd.NA,
            "mpd_party_id": g["mpd_party_id"].dropna().astype(int).iloc[0] if g["mpd_party_id"].notna().any() else pd.NA,
            "partyname": choose_canonical_name(g["partyname"]),
            "partyname_eng": choose_canonical_name(g["partyname_eng"]),
            "share_mpd": share_pick,
            "share_key": safe_round_share(share_pick),
            "partyname_norm": norm_name_scalar(choose_canonical_name(g["partyname"])),
            "partyname_eng_norm": norm_name_scalar(choose_canonical_name(g["partyname_eng"])),
            "local_aliases": " || ".join(local_aliases),
            "eng_aliases": " || ".join(eng_aliases),
            "n_alias_local": len(local_aliases),
            "n_alias_eng": len(eng_aliases),
            "ess_rows_all": int(len(g)),
            "ess_rows_vote12": int(len(vote12)),
            "ess_weight_all": weighted_all,
            "ess_weight_vote12": weighted_vote12,
            "n_markets_vote12": int(vote12["election"].nunique()) if len(vote12) else 0,
            "n_regions_vote12": int(vote12["region4"].nunique()) if len(vote12) else 0,
            "has_any_manual_flag": int(g["manual"].fillna(0).astype(float).gt(0).any()) if "manual" in g.columns else 0,
            "partyfacts_id_any": choose_canonical_name(g["partyfacts_id"]) if "partyfacts_id" in g.columns else "",
        }
        rows.append(row)

    pe = pd.DataFrame(rows)
    if pe.empty:
        return pe

    pe = pe.sort_values(["iso3c", "election_year", "mpd_party_id", "partyname_eng", "partyname"], kind="mergesort").reset_index(drop=True)
    pe["party_election_id"] = np.arange(1, len(pe) + 1)
    pe["party_election_id"] = pe["party_election_id"].map(lambda x: f"pe{x:06d}")
    pe["target_id"] = pe["party_election_id"]
    return pe



def attach_party_election_id(h: pd.DataFrame, pe: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["stable_party_election_key", "party_election_id"]
    return h.merge(pe[key_cols], on="stable_party_election_key", how="left", validate="m:1")



def build_auto_backbone(h: pd.DataFrame) -> pd.DataFrame:
    auto = h.copy()
    if KEEP_ONLY_VOTE_1_2:
        auto = auto[auto["vote"].isin([1, 2])].copy()
    if not KEEP_UNMATCHED_PARTY_VOTERS:
        auto = auto[~((auto["vote"] == 1) & (auto["mpd_party_id"].isna()))].copy()

    # Match the current pyblp cleaning step as closely as possible, except party chars.
    auto = auto.dropna(subset=["female", "educ_cat", "age", "anweight", "lrscale", "election_year", "region"]).copy()
    auto = auto[(auto["region4"].notna()) & (auto["region4"] != "")].copy()

    # Keep raw MPD policy content variables.
    per_cols = sorted([c for c in auto.columns if re.match(r"^per\d+", c, flags=re.IGNORECASE)])

    keep = [
        "party_election_id", "stable_party_election_key",
        "country", "iso2c", "iso3c", "election_year", "edate_mpd", "edate_ess",
        "region", "region4", "election",
        "vote", "anweight", "pspwght", "pweight",
        "female", "educ_cat", "age", "college", "lrscale",
        "mpd_party_id", "parfam", "partyname", "partyname_eng", "share_mpd",
        "partyfacts_id", "manual",
    ] + per_cols
    keep = [c for c in keep if c in auto.columns]
    auto = auto[keep].copy()

    # IMPORTANT: do NOT include empty candidate columns here. The current pyblp script
    # uses their presence/absence to decide whether to merge party_char later.
    for bad in ["candidate_age", "candidate_gender", "candidate_gender_male1"]:
        if bad in auto.columns:
            auto = auto.drop(columns=[bad])

    return auto



def build_demo_backbone(h: pd.DataFrame) -> pd.DataFrame:
    demo = h.copy()
    demo = demo.dropna(subset=["election", "region4", "age", "college"]).copy()
    demo = demo[(demo["region4"] != "")].copy()

    keep = [
        "country", "iso2c", "iso3c", "election_year", "region", "region4", "election",
        "anweight", "pspwght", "pweight",
        "age", "dob", "age_at_election", "female", "college", "lrscale",
        "young", "middle", "old", "boomer",
    ]
    keep = [c for c in keep if c in demo.columns]
    demo = demo[keep].copy()

    rng = np.random.default_rng(RANDOM_SEED)
    demo_list: list[pd.DataFrame] = []
    for _, g in demo.groupby("election", sort=True):
        gg = weighted_resample_market(g, AGENTS_PER_MARKET, "anweight", rng)
        demo_list.append(gg)
    demo = pd.concat(demo_list, ignore_index=True) if demo_list else demo.iloc[0:0].copy()
    return demo



def build_ai_seed(pe: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "target_id", "party_election_id", "country", "iso2c", "iso3c", "election_year",
        "edate_mpd", "mpd_party_id", "parfam", "partyname", "partyname_eng", "share_mpd",
        "share_key", "partyname_norm", "partyname_eng_norm", "local_aliases", "eng_aliases",
        "ess_rows_vote12", "ess_weight_vote12", "n_markets_vote12",
    ]
    cols = [c for c in cols if c in pe.columns]
    seed = pe[cols].copy()
    return seed



def save_report(report_rows: list[dict[str, Any]]) -> None:
    rep = pd.DataFrame(report_rows)
    rep.to_csv(REPORT_CSV, index=False)
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report_rows, f, ensure_ascii=False, indent=2, default=str)



def finalize_with_party_char(auto_path: Path, party_char_path: Path, out_path: Path) -> pd.DataFrame:
    if not auto_path.exists():
        raise FileNotFoundError(f"Auto backbone not found: {auto_path}")
    if not party_char_path.exists():
        raise FileNotFoundError(f"Party char file not found: {party_char_path}")

    auto = pd.read_stata(auto_path, convert_categoricals=False)
    pc = pd.read_stata(party_char_path, convert_categoricals=False)

    # Preferred merge: party_election_id.
    if "party_election_id" in auto.columns and "party_election_id" in pc.columns:
        keys = ["party_election_id"]
    else:
        need = ["iso3c", "election_year", "mpd_party_id"]
        if not all(k in auto.columns for k in need) or not all(k in pc.columns for k in need):
            raise ValueError(
                "Party char merge needs either `party_election_id` in both files or the stable key columns "
                "`iso3c`, `election_year`, `mpd_party_id`."
            )
        keys = need

    # Keep BLP-friendly candidate fields. If the leader file also contains the male=1 scale,
    # preserve it too, but the current pyblp code uses `candidate_gender`.
    keep = keys + [c for c in ["candidate_age", "candidate_gender", "candidate_gender_male1"] if c in pc.columns]
    pc_sub = pc[keep].drop_duplicates(keys)
    out = auto.merge(pc_sub, on=keys, how="left", validate="m:1")
    out = cast_stata_friendly(out)
    out.to_stata(out_path, write_index=False, version=118)
    print(f"Saved finalized auto file with candidate characteristics: {out_path}")
    return out


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    print("Starting PyBLP backbone builder...")
    print(f"Base directory: {BASE_DIR}")
    print(f"Validated input: {VALIDATED_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")

    report_rows: list[dict[str, Any]] = []

    if RUN_BUILD_BACKBONE:
        print("\n--- BUILD BACKBONE STEP ---")
        h, meta = load_and_harmonize(VALIDATED_PATH)
        pe = build_party_election_table(h)
        h = attach_party_election_id(h, pe)
        auto = build_auto_backbone(h)
        demo = build_demo_backbone(h)
        seed = build_ai_seed(pe)

        auto = cast_stata_friendly(auto)
        demo = cast_stata_friendly(demo)
        pe = cast_stata_friendly(pe)
        seed = cast_stata_friendly(seed)

        auto.to_stata(AUTO_OUT, write_index=False, version=118)
        demo.to_stata(DEMO_OUT, write_index=False, version=118)
        pe.to_csv(PARTY_ELECTION_CSV, index=False)
        pe.to_stata(PARTY_ELECTION_DTA, write_index=False, version=118)
        seed.to_csv(AI_SEED_CSV, index=False)
        seed.to_stata(AI_SEED_DTA, write_index=False, version=118)

        print(f"Saved auto backbone: {AUTO_OUT}")
        print(f"Saved demo backbone: {DEMO_OUT}")
        print(f"Saved party-election table: {PARTY_ELECTION_CSV}")
        print(f"Saved AI seed table: {AI_SEED_CSV}")

        report_rows.extend([
            {"metric": "validated_input_rows", "value": int(meta["n_input_rows"]), "note": "rows in ess_mpd_matched_validated.dta"},
            {"metric": "region_column_used", "value": str(meta["region_column_used"]), "note": "true geographic region variable used for markets"},
            {"metric": "n_raw_per_cols", "value": int(meta["n_per_cols"]), "note": "raw MPD per* columns preserved"},
            {"metric": "auto_rows", "value": int(len(auto)), "note": "respondent rows in BLP backbone"},
            {"metric": "auto_markets", "value": int(auto["election"].nunique()) if "election" in auto.columns else 0, "note": "unique region4 × election_year markets"},
            {"metric": "demo_rows", "value": int(len(demo)), "note": "agent rows in demo file"},
            {"metric": "demo_markets", "value": int(demo["election"].nunique()) if "election" in demo.columns else 0, "note": "markets represented in demo file"},
            {"metric": "demo_rows_per_market_min", "value": int(demo.groupby("election").size().min()) if len(demo) else 0, "note": "minimum pseudo-agents per market"},
            {"metric": "demo_rows_per_market_max", "value": int(demo.groupby("election").size().max()) if len(demo) else 0, "note": "maximum pseudo-agents per market"},
            {"metric": "party_election_rows", "value": int(len(pe)), "note": "unique party-election backbone rows"},
            {"metric": "party_election_with_vote12", "value": int((pe["ess_rows_vote12"] > 0).sum()) if "ess_rows_vote12" in pe.columns else 0, "note": "party-elections with at least one vote∈{1,2} respondent"},
            {"metric": "auto_unmatched_party_vote12_rows", "value": int(((auto["vote"] == 1) & (auto["mpd_party_id"].isna())).sum()) if {"vote", "mpd_party_id"}.issubset(auto.columns) else 0, "note": "kept as outside-share mass"},
            {"metric": "keep_unmatched_party_voters", "value": int(KEEP_UNMATCHED_PARTY_VOTERS), "note": "1=yes"},
            {"metric": "agents_per_market", "value": int(AGENTS_PER_MARKET), "note": "demo pseudo-sample target"},
            {"metric": "weight_used", "value": "anweight", "note": "analysis weight used throughout backbone and demo construction"},
        ])

    if RUN_FINALIZE_WITH_PARTY_CHAR:
        print("\n--- FINALIZE WITH PARTY CHARACTERISTICS STEP ---")
        out = finalize_with_party_char(AUTO_OUT, PARTY_CHAR_PATH, FINAL_AUTO_WITH_CHAR)
        report_rows.append({
            "metric": "final_auto_with_char_rows",
            "value": int(len(out)),
            "note": "rows after stable-key merge of candidate_age/candidate_gender"
        })

    save_report(report_rows)
    print(f"Saved build report: {REPORT_CSV}")
    print("Done.")


if __name__ == "__main__":
    main()
