from __future__ import annotations

"""
Spyder-ready, audit-first AI extension for ESS-MPD unmatched cases (v6.2 queue cleanup for missing labels + low-value ballot options).

Principles
----------
- Treat all existing matches in Pablo's validated file as locked truth.
- Use AI only on still-unmatched ESS party-election cases.
- Never overwrite a non-missing validated MPD match.
- Write every output to NEW filenames so the original validated file stays untouched.
- Default to review-first: AI suggestions are written to a review CSV with
  trust scores, concise justifications, and source links. Applying them back
  into a respondent-level file requires explicit approval.

Main outputs
------------
- ess_mpd_ai_rigorous/ess_mpd_ai_unmatched_queue_rigorous_v6_2.csv
- ess_mpd_ai_rigorous/ess_mpd_ai_results_rigorous_v6_2.csv
- ess_mpd_ai_rigorous/ess_mpd_ai_review_rigorous_v6_2.csv
- ess_mpd_ai_rigorous/ess_mpd_ai_approved_matches_v6_2.csv
- ess_mpd_matched_validated_plus_ai_rigorous_v6_2.dta / .csv
"""

import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd

# =============================================================================
# Inlined shared helpers
# =============================================================================
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Sequence
import zipfile
import struct
import zlib

import numpy as np
import pandas as pd
from pandas.io.stata import StataReader

# -----------------------------------------------------------------------------
# Country code helpers
# -----------------------------------------------------------------------------
ISO2_TO_ISO3: dict[str, str] = {
    "AL": "ALB", "AT": "AUT", "BA": "BIH", "BE": "BEL", "BG": "BGR",
    "CH": "CHE", "CY": "CYP", "CZ": "CZE", "DE": "DEU", "DK": "DNK",
    "EE": "EST", "ES": "ESP", "FI": "FIN", "FR": "FRA", "GB": "GBR",
    "GR": "GRC", "HR": "HRV", "HU": "HUN", "IE": "IRL", "IL": "ISR",
    "IS": "ISL", "IT": "ITA", "LT": "LTU", "LU": "LUX", "LV": "LVA",
    "ME": "MNE", "MK": "MKD", "NL": "NLD", "NO": "NOR", "PL": "POL",
    "PT": "PRT", "RO": "ROU", "RS": "SRB", "RU": "RUS", "SE": "SWE",
    "SI": "SVN", "SK": "SVK", "TR": "TUR", "UA": "UKR",
}

ISO2_TO_COUNTRY: dict[str, str] = {
    "AL": "Albania", "AT": "Austria", "BA": "Bosnia-Herzegovina", "BE": "Belgium",
    "BG": "Bulgaria", "CH": "Switzerland", "CY": "Cyprus", "CZ": "Czech Republic",
    "DE": "Germany", "DK": "Denmark", "EE": "Estonia", "ES": "Spain",
    "FI": "Finland", "FR": "France", "GB": "United Kingdom", "GR": "Greece",
    "HR": "Croatia", "HU": "Hungary", "IE": "Ireland", "IL": "Israel",
    "IS": "Iceland", "IT": "Italy", "LT": "Lithuania", "LU": "Luxembourg",
    "LV": "Latvia", "ME": "Montenegro", "MK": "North Macedonia", "NL": "Netherlands",
    "NO": "Norway", "PL": "Poland", "PT": "Portugal", "RO": "Romania",
    "RS": "Serbia", "RU": "Russia", "SE": "Sweden", "SI": "Slovenia",
    "SK": "Slovakia", "TR": "Turkey", "UA": "Ukraine",
}

COUNTRY_NAME_TO_ISO2: dict[str, str] = {
    "Albania": "AL",
    "Austria": "AT",
    "Azerbaijan": "AZ",
    "Belarus": "BY",
    "Belgium": "BE",
    "Bosnia-Herzegovina": "BA",
    "Bulgaria": "BG",
    "Canada": "CA",
    "Chile": "CL",
    "Colombia": "CO",
    "Costa Rica": "CR",
    "Croatia": "HR",
    "Cyprus": "CY",
    "Czech Republic": "CZ",
    "Denmark": "DK",
    "Dominican Republic": "DO",
    "Ecuador": "EC",
    "Estonia": "EE",
    "Finland": "FI",
    "France": "FR",
    "Georgia": "GE",
    "German Democratic Republic": "DD",
    "Germany": "DE",
    "Greece": "GR",
    "Hungary": "HU",
    "Iceland": "IS",
    "Ireland": "IE",
    "Israel": "IL",
    "Italy": "IT",
    "Japan": "JP",
    "Latvia": "LV",
    "Lithuania": "LT",
    "Luxembourg": "LU",
    "Malta": "MT",
    "Mexico": "MX",
    "Moldova": "MD",
    "Montenegro": "ME",
    "Netherlands": "NL",
    "New Zealand": "NZ",
    "North Macedonia": "MK",
    "Northern Ireland": "GB",
    "Norway": "NO",
    "Panama": "PA",
    "Poland": "PL",
    "Portugal": "PT",
    "Romania": "RO",
    "Russia": "RU",
    "Serbia": "RS",
    "Slovakia": "SK",
    "Slovenia": "SI",
    "South Africa": "ZA",
    "South Korea": "KR",
    "Spain": "ES",
    "Sweden": "SE",
    "Switzerland": "CH",
    "Turkey": "TR",
    "Ukraine": "UA",
    "United Kingdom": "GB",
    "United States": "US",
}

GENERIC_NONPARTY_PATTERNS = [
    r"\bother\b",
    r"\bblank\b",
    r"\bblanc\b",
    r"\bblanco\b",
    r"\bnulo\b",
    r"\bnull\b",
    r"\bspoiled\b",
    r"\binvalid\b",
    r"\bdid not vote\b",
    r"\bdidn['’]?t vote\b",
    r"\bdon['’]?t know\b",
    r"\bdk\b",
    r"\brefus",
    r"\brefuse",
    r"\bnone\b",
    r"\bno party\b",
    r"\bwhite ballot\b",
    r"\banother party\b",
    r"\bmaggioritario\b",
    r"\bmajoritarian\b",
    r"\bnot eligible\b",
]
GENERIC_NONPARTY_RE = re.compile("|".join(GENERIC_NONPARTY_PATTERNS), re.I)


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
def clean_text_scalar(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    s = str(x).strip()
    if s.lower() in {"", "nan", "none", "<na>"}:
        return ""
    return s


def ascii_fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", clean_text_scalar(s))
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def normalize_text_for_match(s: str) -> str:
    s = ascii_fold(s).lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def name_similarity(a: str, b: str) -> float:
    from difflib import SequenceMatcher

    a_norm = normalize_text_for_match(a)
    b_norm = normalize_text_for_match(b)
    if not a_norm or not b_norm:
        return 0.0
    return float(SequenceMatcher(None, a_norm, b_norm).ratio())


def is_generic_nonparty_option(*names: Any) -> bool:
    joined = " | ".join(clean_text_scalar(x) for x in names if clean_text_scalar(x))
    return bool(joined and GENERIC_NONPARTY_RE.search(joined))


LOW_VALUE_BALLOT_OPTION_LABELS_NORMALIZED = {
    "independent",
    "mixed vote",
    "against all",
    "nul",
}


def is_missing_display_name(name: Any) -> bool:
    return clean_text_scalar(name) == ""


def is_low_value_ballot_option_label(name: Any) -> bool:
    return normalize_text_for_match(name) in LOW_VALUE_BALLOT_OPTION_LABELS_NORMALIZED


def find_first_existing(paths: Sequence[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def resolve_input_path(label: str, candidates: Sequence[Path]) -> Path:
    found = find_first_existing(candidates)
    if found is None:
        joined = "\n".join(f"  - {p}" for p in candidates)
        raise FileNotFoundError(f"Could not find {label}. Tried:\n{joined}")
    return found


def _recover_single_file_from_broken_zip(zip_path: Path, extract_dir: Path) -> Path:
    """
    Recover a single-file zip archive when the central directory is missing but the
    local file header and compressed payload are still intact.

    This is intentionally conservative and only supports the cases we need here:
    one member, .dta payload, deflate or stored compression.
    """
    with zip_path.open("rb") as f:
        header = f.read(30)
        if len(header) < 30:
            raise zipfile.BadZipFile(f"{zip_path} is too short to contain a local zip header")
        sig, ver, flag, comp, _mtime, _mdate, _crc, comp_size, uncomp_size, fname_len, extra_len = struct.unpack(
            "<IHHHHHIIIHH", header
        )
        if sig != 0x04034B50:
            raise zipfile.BadZipFile(f"{zip_path} does not start with a local zip file header")
        fname = f.read(fname_len)
        f.read(extra_len)
        member_name = Path(fname.decode("utf-8", errors="replace")).name
        if not member_name.lower().endswith(".dta"):
            raise zipfile.BadZipFile(
                f"Broken zip fallback expected a single .dta member, found {member_name!r}"
            )
        target = extract_dir / member_name
        if target.exists() and target.stat().st_size > 0:
            return target

        if comp == 0 and not (flag & 0x08):
            payload = f.read(comp_size)
            target.write_bytes(payload)
            if uncomp_size and target.stat().st_size != uncomp_size:
                raise IOError(
                    f"Recovered {target} but size {target.stat().st_size} does not match expected {uncomp_size}"
                )
            return target

        if comp != 8:
            raise zipfile.BadZipFile(
                f"Broken zip fallback only supports stored/deflate compression; found method {comp}"
            )

        decomp = zlib.decompressobj(-15)
        with target.open("wb") as dst:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                out = decomp.decompress(chunk)
                if out:
                    dst.write(out)
                if decomp.eof:
                    break
            flush = decomp.flush()
            if flush:
                dst.write(flush)

        if target.stat().st_size == 0:
            raise IOError(f"Broken zip fallback produced an empty file for {target}")
        if uncomp_size and target.stat().st_size != uncomp_size:
            raise IOError(
                f"Recovered {target} but size {target.stat().st_size} does not match expected {uncomp_size}"
            )
        return target


def ensure_single_dta_from_zip(zip_path: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    marker = extract_dir / f"{zip_path.stem}.extracted.ok"
    existing_dtas = sorted(extract_dir.glob("*.dta"))
    if existing_dtas and marker.exists() and existing_dtas[0].stat().st_size > 0:
        return existing_dtas[0]
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            dta_members = [n for n in zf.namelist() if n.lower().endswith(".dta")]
            if len(dta_members) != 1:
                raise ValueError(f"Expected exactly one .dta inside {zip_path}, found {len(dta_members)}")
            member = dta_members[0]
            info = zf.getinfo(member)
            target = extract_dir / Path(member).name
            if target.exists() and target.stat().st_size != info.file_size:
                target.unlink()
            if not target.exists() or target.stat().st_size == 0:
                with zf.open(member) as src, target.open("wb") as dst:
                    while True:
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        dst.write(chunk)
            if target.stat().st_size != info.file_size:
                raise IOError(
                    f"Extracted {target} but size {target.stat().st_size} does not match zip member size {info.file_size}"
                )
            marker.write_text(member, encoding="utf-8")
            return target
    except zipfile.BadZipFile:
        target = _recover_single_file_from_broken_zip(zip_path, extract_dir)
        marker.write_text(target.name, encoding="utf-8")
        return target


def get_stata_columns(path: Path) -> list[str]:
    with pd.read_stata(path, iterator=True, convert_categoricals=False) as reader:
        df = reader.read(1)
    return list(df.columns)


def get_stata_variable_labels(path: Path) -> dict[str, str]:
    with StataReader(path) as reader:
        return dict(reader.variable_labels())


def cols_between(columns: Sequence[str], start: str, end: str) -> list[str]:
    if start not in columns or end not in columns:
        return []
    i0 = columns.index(start)
    i1 = columns.index(end)
    if i0 <= i1:
        return list(columns[i0 : i1 + 1])
    return list(columns[i1 : i0 + 1])


def parse_manual_date(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    # Fallback for a few compact forms like 24nov2002.
    mask = dt.isna() & s.notna()
    if mask.any():
        compact = s[mask].str.replace(r"(?i)^([0-9]{1,2})([A-Za-z]{3})([0-9]{2,4})$", r"\1-\2-\3", regex=True)
        dt.loc[mask] = pd.to_datetime(compact, errors="coerce", dayfirst=True)
    return dt


def to_numeric_or_na(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def pick_first_nonmissing(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    existing = [c for c in columns if c in df.columns]
    if not existing:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for col in existing:
        out = out.where(out.notna(), to_numeric_or_na(df[col]))
    return out


# -----------------------------------------------------------------------------
# Path resolution helpers for this project
# -----------------------------------------------------------------------------
def resolve_ess_input(base_dir: Path, tmp_dir: Path) -> Path:
    candidates = [
        base_dir / "0_data" / "raw" / "ESS" / "ESS_raw_july25_compressed.dta",
        base_dir / "0_data" / "raw" / "ESS" / "ESS_raw_july25.dta",
        base_dir / "0_data" / "raw" / "ESS" / "ESS_raw_july25.zip",
        base_dir / "ESS_raw_july25_compressed.dta",
        base_dir / "ESS_raw_july25.dta",
        base_dir / "ESS_raw_july25.zip",
        base_dir / "ESS1e06_7-ESS2e03_6-ESS3e03_7-ESS4e04_6-ESS5e03_5-ESS6e02_6-ESS7e02_3-ESS8e02_3-ESS9e03_2-ESS10-ESS10SC-ESS11-subset.dta",
    ]
    path = resolve_input_path("ESS respondent file", candidates)
    if path.suffix.lower() == ".zip":
        return ensure_single_dta_from_zip(path, tmp_dir / "ess_zip_extract")
    return path


def resolve_mpd_input(base_dir: Path) -> Path:
    return resolve_input_path(
        "MPD file",
        [
            base_dir / "0_data" / "raw" / "MPD" / "MPDataset_MPDS2025a_stata14.dta",
            base_dir / "MPDataset_MPDS2025a_stata14.dta",
        ],
    )


def resolve_essprt_all_input(base_dir: Path) -> Path:
    return resolve_input_path(
        "essprt-all.csv",
        [
            base_dir / "0_data" / "raw" / "partyfacts_match" / "essprt-all.csv",
            base_dir / "essprt-all.csv",
        ],
    )


def resolve_partyfacts_external_input(base_dir: Path) -> Path:
    return resolve_input_path(
        "partyfacts-external-parties.csv",
        [
            base_dir / "0_data" / "raw" / "partyfacts_match" / "partyfacts-external-parties.csv",
            base_dir / "partyfacts-external-parties.csv",
        ],
    )


def resolve_manual_match_dir(base_dir: Path) -> Path:
    candidates = [
        base_dir / "0_data" / "manual_MPD" / "match",
        base_dir / "manual_MPD" / "match",
        base_dir / "match",
    ]
    found = find_first_existing(candidates)
    if found is not None and found.is_dir():
        return found
    zip_candidates = [
        base_dir / "0_data" / "manual_MPD" / "match.zip",
        base_dir / "match.zip",
    ]
    zip_found = find_first_existing(zip_candidates)
    if zip_found is None:
        joined = "\n".join(f"  - {p}" for p in candidates + zip_candidates)
        raise FileNotFoundError(f"Could not find manual match directory. Tried:\n{joined}")
    extract_dir = base_dir / "_tmp_ess_mpd_match" / "manual_match_extract"
    extract_dir.mkdir(parents=True, exist_ok=True)
    marker = extract_dir / "manual_match_extracted.ok"
    if not marker.exists():
        with zipfile.ZipFile(zip_found, "r") as zf:
            zf.extractall(extract_dir)
        marker.write_text("ok", encoding="utf-8")
    match_dir = extract_dir / "match"
    if not match_dir.exists():
        raise FileNotFoundError(f"Extracted {zip_found} but could not find inner 'match/' folder")
    return match_dir


def iso2_to_iso3(iso2: Any) -> str:
    return ISO2_TO_ISO3.get(clean_text_scalar(iso2), "")


def iso2_to_country(iso2: Any) -> str:
    return ISO2_TO_COUNTRY.get(clean_text_scalar(iso2), "")


def mpd_country_to_iso2(countryname: Any) -> str:
    return COUNTRY_NAME_TO_ISO2.get(clean_text_scalar(countryname), "")


# -----------------------------------------------------------------------------
# Date and interview helpers
# -----------------------------------------------------------------------------
def _series_from_date_component(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return to_numeric_or_na(df[col])


def compute_interview_dates(df: pd.DataFrame, iso_col: str = "iso2c", round_col: str = "essround") -> pd.DataFrame:
    out = df.copy()

    year = _series_from_date_component(out, "inwyr")
    month = _series_from_date_component(out, "inwmm")
    day = _series_from_date_component(out, "inwdd")
    edate_ess = pd.to_datetime(
        dict(year=year, month=month, day=day), errors="coerce"
    )

    questcmp = pd.to_datetime(out["questcmp"], errors="coerce") if "questcmp" in out.columns else pd.Series(pd.NaT, index=out.index)
    inwde = pd.to_datetime(out["inwde"], errors="coerce") if "inwde" in out.columns else pd.Series(pd.NaT, index=out.index)

    year_end = _series_from_date_component(out, "inwyye")
    month_end = _series_from_date_component(out, "inwmme")
    day_end = _series_from_date_component(out, "inwdde")

    year_end = year_end.where(year_end.notna(), questcmp.dt.year)
    year_end = year_end.where(year_end.notna(), inwde.dt.year)
    month_end = month_end.where(month_end.notna(), questcmp.dt.month)
    month_end = month_end.where(month_end.notna(), inwde.dt.month)
    day_end = day_end.where(day_end.notna(), questcmp.dt.day)
    day_end = day_end.where(day_end.notna(), inwde.dt.day)
    day_end = day_end.fillna(15)

    edate_ess_end = pd.to_datetime(
        dict(year=year_end, month=month_end, day=day_end), errors="coerce"
    )

    year_supp = _series_from_date_component(out, "supqyr")
    month_supp = _series_from_date_component(out, "supqmm")
    day_supp = _series_from_date_component(out, "supqdd")
    edate_ess_supp = pd.to_datetime(
        dict(year=year_supp, month=month_supp, day=day_supp), errors="coerce"
    )

    edate_ess = edate_ess.where(edate_ess.notna(), edate_ess_end)

    if iso_col not in out.columns or round_col not in out.columns:
        raise KeyError(f"Missing required columns {iso_col!r} and/or {round_col!r}")

    def latest_mode(series: pd.Series) -> pd.Timestamp:
        s = series.dropna()
        if s.empty:
            return pd.NaT
        modes = s.mode(dropna=True)
        if len(modes) == 0:
            return pd.NaT
        return modes.max()

    mode_by_group = edate_ess.groupby([out[iso_col], out[round_col]], sort=False).transform(latest_mode)
    edate_ess = edate_ess.where(edate_ess.notna(), mode_by_group)
    edate_ess = edate_ess.where(edate_ess.notna(), edate_ess_supp)

    out["edate_ess"] = pd.to_datetime(edate_ess, errors="coerce")
    return out


def build_resp_id(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    key_df = df.loc[:, list(cols)].copy()
    # Preserve first-appearance order, similar to Stata's group() behavior on current data order.
    key_tuples = list(map(tuple, key_df.itertuples(index=False, name=None)))
    codes, _ = pd.factorize(key_tuples, sort=False)
    return pd.Series(codes + 1, index=df.index, dtype="int64")


# -----------------------------------------------------------------------------
# Data export helpers
# -----------------------------------------------------------------------------
def _coerce_stata_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == bool:
            out[col] = out[col].astype("int8")
        elif pd.api.types.is_object_dtype(out[col]):
            out[col] = out[col].where(out[col].notna(), None)
    return out



_STATA_RESERVED_WORDS = {
    "_all", "_b", "_coef", "_cons", "_crc", "_dta", "_eq", "_est", "_n", "_pi", "_pred",
    "_rc", "_skip", "_merge", "byte", "int", "long", "float", "double", "if", "in", "using",
    "with", "for", "while", "foreach", "program", "do", "end", "clear", "set", "scalar",
    "matrix", "global", "local",
}

_STATA_NAME_ALIASES = {
    "ai_match_vote_share_gap_soft_alert": "ai_match_vote_gap_alert",
    "ai_match_vote_share_gap_soft_alert_note": "ai_match_vote_gap_alert_note",
}

def _is_valid_stata_name(name: str) -> bool:
    s = clean_text_scalar(name)
    if not s or len(s) > 32:
        return False
    if s.lower() in _STATA_RESERVED_WORDS:
        return False
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", s):
        return False
    return True

def _make_stata_safe_name(name: str, used: set[str]) -> str:
    original = clean_text_scalar(name)
    if original in _STATA_NAME_ALIASES:
        alias = _STATA_NAME_ALIASES[original]
        if alias not in used and _is_valid_stata_name(alias):
            return alias

    base = ascii_fold(original)
    base = re.sub(r"[^A-Za-z0-9_]", "_", base)
    base = re.sub(r"_+", "_", base).strip("_")
    if not base:
        base = "var"
    if not re.match(r"^[A-Za-z_]", base):
        base = "_" + base
    if base.lower() in _STATA_RESERVED_WORDS:
        base = "_" + base

    candidate = base[:32]
    if candidate.lower() in _STATA_RESERVED_WORDS:
        candidate = ("_" + candidate)[:32]
    if candidate not in used and _is_valid_stata_name(candidate):
        return candidate

    stem = candidate[:28] if len(candidate) > 28 else candidate
    i = 1
    while True:
        suffix = f"_{i}"
        cand = (stem[: 32 - len(suffix)] + suffix)[:32]
        if cand.lower() in _STATA_RESERVED_WORDS:
            cand = ("_" + cand)[:32]
        if cand not in used and _is_valid_stata_name(cand):
            return cand
        i += 1

def make_stata_safe_rename_map(columns: Sequence[str]) -> dict[str, str]:
    rename_map: dict[str, str] = {}
    used: set[str] = set()
    for col in columns:
        col_str = clean_text_scalar(col)
        target = col_str
        if (not _is_valid_stata_name(col_str)) or (col_str in used):
            target = _make_stata_safe_name(col_str, used)
        used.add(target)
        if target != col:
            rename_map[col] = target
    return rename_map

def write_stata_name_map(rename_map: dict[str, str], dta_path: Path) -> Path | None:
    if not rename_map:
        return None
    out_path = dta_path.with_name(dta_path.stem + "_stata_name_map.csv")
    map_df = pd.DataFrame(
        [{"original_name": k, "stata_name": v} for k, v in rename_map.items()]
    ).sort_values(["original_name", "stata_name"], kind="stable")
    map_df.to_csv(out_path, index=False)
    return out_path

def save_stata(
    df: pd.DataFrame,
    path: Path,
    *,
    variable_labels: dict[str, str] | None = None,
    data_label: str = "",
    daily_date_cols: Iterable[str] | None = None,
) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = _coerce_stata_object_columns(df)
    rename_map = make_stata_safe_rename_map(list(out.columns))
    if rename_map:
        out = out.rename(columns=rename_map)

    convert_dates: dict[str, str] = {}
    if daily_date_cols is not None:
        for col in daily_date_cols:
            safe_col = rename_map.get(col, col)
            if safe_col in out.columns and pd.api.types.is_datetime64_any_dtype(out[safe_col]):
                convert_dates[safe_col] = "td"

    if variable_labels:
        varlabs = {}
        for k, v in variable_labels.items():
            safe_k = rename_map.get(k, k)
            if safe_k in out.columns:
                varlabs[safe_k] = clean_text_scalar(v)[:80]
    else:
        varlabs = None

    out.to_stata(
        path,
        write_index=False,
        version=118,
        variable_labels=varlabs,
        data_label=clean_text_scalar(data_label)[:80],
        convert_dates=convert_dates,
    )
    return rename_map


def json_dumps_compact(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"), sort_keys=False)


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
# SETTINGS FOR SPYDER
# =============================================================================
BASE_DIR = Path(os.environ.get("ESS_MPD_BASE_DIR", r"/Users/zl279/Downloads/Poli-IO/raw_data"))
DATA0_DIR = BASE_DIR / "0_data"
PROC_DIR = DATA0_DIR / "proc"
AI_DIR = PROC_DIR / "ess_mpd_ai_rigorous"
AI_DIR.mkdir(parents=True, exist_ok=True)

def resolve_validated_input(base_dir: Path) -> Path:
    direct_candidates = [
        PROC_DIR / "ess_mpd_matched_validated.dta",
        base_dir / "ess_mpd_matched_validated.dta",
        PROC_DIR / "ess_mpd_matched_validated_python.dta",
        base_dir / "recovered_ess_mpd_matched_validated.dta",
    ]
    for p in direct_candidates:
        if p.exists():
            return p
    zip_candidates = [
        PROC_DIR / "ess_mpd_matched_validated.zip",
        base_dir / "ess_mpd_matched_validated.zip",
    ]
    for z in zip_candidates:
        if z.exists():
            extract_dir = base_dir / "_tmp_ess_mpd_match" / "validated_extract"
            return ensure_single_dta_from_zip(z, extract_dir)
    raise FileNotFoundError(
        "Could not find a validated ESS-MPD file. Tried .dta and .zip locations under proc/ and BASE_DIR."
    )


SOURCE_VALIDATED_PATH = resolve_validated_input(BASE_DIR)

MODEL = "gpt-5.4"
OPENAI_API_KEY = ""
RUN_PREP = False
RUN_COLLECT = False
RUN_BUILD_REVIEW = False
RUN_APPLY = True
PILOT_LIMIT: int | None = None
SLEEP_SECONDS = 0.0
SKIP_GENERIC_NONPARTY_OPTIONS = True
SKIP_MISSING_DISPLAY_NAME_ROWS = True
SKIP_LOW_VALUE_BALLOT_OPTION_LABELS = True
PREFLIGHT_ONLY = False

# Review/apply policy
REQUIRE_MANUAL_APPROVAL_TO_APPLY = True
AUTO_APPLY_STRONG_MATCHES = False
MIN_TRUST_SCORE_TO_APPLY = 90

QUEUE_CSV = AI_DIR / "ess_mpd_ai_unmatched_queue_rigorous_v6_2.csv"
QUEUE_DTA = AI_DIR / "ess_mpd_ai_unmatched_queue_rigorous_v6_2.dta"
REQUESTS_JSONL = AI_DIR / "ess_mpd_ai_requests_rigorous_v6_2.jsonl"
RESULTS_JSONL = AI_DIR / "ess_mpd_ai_results_rigorous_v6_2.jsonl"
RESULTS_CSV = AI_DIR / "ess_mpd_ai_results_rigorous_v6_2.csv"
REVIEW_CSV = AI_DIR / "ess_mpd_ai_review_rigorous_v6_2.csv"
APPROVED_CSV = AI_DIR / "ess_mpd_ai_approved_matches_v6_2.csv"
FINAL_CSV = PROC_DIR / "ess_mpd_matched_validated_plus_ai_rigorous_v6_2.csv"
FINAL_DTA = PROC_DIR / "ess_mpd_matched_validated_plus_ai_rigorous_v6_2.dta"

PREFERRED_WEB_SEARCH_TOOL_TYPES = ["web_search", "web_search_preview", "web_search_2025_08_26", "web_search_preview_2025_03_11"]
WEB_SEARCH_TOOL = {"type": PREFERRED_WEB_SEARCH_TOOL_TYPES[0]}

MATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "resolved": {"type": ["boolean", "null"]},
        "resolution_status": {"type": ["string", "null"]},
        "selected_mpd_party_id": {"type": ["integer", "null"]},
        "selected_party_name_mpd": {"type": ["string", "null"]},
        "selected_from_candidate_list": {"type": ["boolean", "null"]},
        "candidate_rank_by_share": {"type": ["integer", "null"]},
        "confidence_0_100": {"type": ["integer", "null"]},
        "match_basis": {"type": ["string", "null"]},
        "nonmatch_reason_type": {"type": ["string", "null"]},
        "identified_party_if_absent": {"type": ["string", "null"]},
        "reason_summary": {"type": ["string", "null"]},
        "uncertainty_note": {"type": ["string", "null"]},
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
        "resolved",
        "resolution_status",
        "selected_mpd_party_id",
        "selected_party_name_mpd",
        "selected_from_candidate_list",
        "candidate_rank_by_share",
        "confidence_0_100",
        "match_basis",
        "nonmatch_reason_type",
        "identified_party_if_absent",
        "reason_summary",
        "uncertainty_note",
        "sources",
    ],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You match one unmatched ESS party-vote option to one MPD party within the SAME "
    "country and election date. Existing validated matches elsewhere are already treated "
    "as ground truth; do not reinterpret them. Only choose an MPD party if it is clearly "
    "the same party, alliance, or list for that specific election. Prefer no match over a "
    "weak guess. Generic answers like Other, Blank, DK, Refusal, or invalid ballots should "
    "normally remain unmatched. Use web search to verify renamed lists, electoral alliances, "
    "mergers, or transliterations whenever the match is not obvious from names alone. Prefer "
    "official or high-quality reference sources when available. If the ESS party/list is clearly identifiable but the corresponding true party/list is NOT present among the allowed MPD candidates, leave the case unmatched and record that explicitly in nonmatch_reason_type and identified_party_if_absent. Return JSON only. Do not reveal "
    "private chain-of-thought; provide only a concise, evidence-based summary."
)


# =============================================================================
# Candidate prep
# =============================================================================
def load_manifesto_candidate_table(mpd_path: Path, pf_external_path: Path) -> pd.DataFrame:
    pf_ext = pd.read_csv(pf_external_path)
    pf_ext = pf_ext[pf_ext["dataset_key"].eq("manifesto")].copy()
    pf_ext["partyfacts_id_cand"] = to_numeric_or_na(pf_ext["partyfacts_id"])
    pf_ext["party"] = to_numeric_or_na(pf_ext["dataset_party_id"])
    pf_ext = pf_ext[["party", "partyfacts_id_cand"]].dropna().drop_duplicates()

    mpd = pd.read_stata(
        mpd_path,
        columns=["countryname", "edate", "party", "partyabbrev", "partyname", "pervote"],
        convert_categoricals=False,
    )
    mpd["iso2c"] = mpd["countryname"].map(mpd_country_to_iso2)
    mpd.loc[mpd["countryname"].eq("Northern Ireland"), "iso2c"] = "GB"
    mpd = mpd.rename(columns={"edate": "edate_mpd"})
    mpd = mpd.merge(pf_ext, on="party", how="left")
    mpd["mpd_party_id"] = to_numeric_or_na(mpd["party"])
    mpd["share_mpd"] = to_numeric_or_na(mpd["pervote"])
    mpd["partyname_mpd"] = (
        mpd["partyabbrev"].fillna("").astype(str).str.strip() + ", " + mpd["partyname"].fillna("").astype(str).str.strip()
    ).str.strip()
    mask = mpd["partyabbrev"].isna() | (mpd["partyabbrev"].astype(str).str.strip() == "")
    mpd.loc[mask, "partyname_mpd"] = mpd.loc[mask, "partyname"].fillna("").astype(str).str.strip()
    mpd = mpd[["iso2c", "edate_mpd", "mpd_party_id", "partyfacts_id_cand", "partyname_mpd", "share_mpd"]].copy()
    return mpd


def build_unmatched_queue(validated_path: Path, candidate_table: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "iso3c", "iso2c", "country", "election_year", "edate_mpd", "vote", "ess_party_id",
        "partyfacts_id", "partyname_ess", "partyname", "partyname_eng", "share_ess",
        "mpd_party_id", "manual",
    ]
    key_cols = ["iso3c", "iso2c", "country", "election_year", "edate_mpd", "ess_party_id"]

    def first_nonempty(series: pd.Series) -> Any:
        for val in series:
            txt = clean_text_scalar(val)
            if txt:
                return txt
        return np.nan

    pieces: list[pd.DataFrame] = []
    with pd.read_stata(validated_path, iterator=True, convert_categoricals=False) as reader:
        while True:
            try:
                chunk = reader.read(50000, columns=cols)
            except StopIteration:
                break
            if chunk is None or chunk.empty:
                break
            chunk = chunk[chunk["vote"].eq(1) & chunk["ess_party_id"].notna()].copy()
            if chunk.empty:
                continue
            part = (
                chunk.groupby(key_cols, dropna=False, sort=False)
                .agg(
                    n_voters=("ess_party_id", "size"),
                    share_ess=("share_ess", "max"),
                    current_mpd_party_id=("mpd_party_id", lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan),
                    current_manual=("manual", lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan),
                    partyfacts_id=("partyfacts_id", lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan),
                    partyname_ess=("partyname_ess", first_nonempty),
                    partyname=("partyname", first_nonempty),
                    partyname_eng=("partyname_eng", first_nonempty),
                )
                .reset_index()
            )
            pieces.append(part)

    if not pieces:
        raise ValueError(f"No usable voter-party rows found in {validated_path}")

    agg = (
        pd.concat(pieces, ignore_index=True)
        .groupby(key_cols, dropna=False, sort=False)
        .agg(
            n_voters=("n_voters", "sum"),
            share_ess=("share_ess", "max"),
            current_mpd_party_id=("current_mpd_party_id", lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan),
            current_manual=("current_manual", lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan),
            partyfacts_id=("partyfacts_id", lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan),
            partyname_ess=("partyname_ess", first_nonempty),
            partyname=("partyname", first_nonempty),
            partyname_eng=("partyname_eng", first_nonempty),
        )
        .reset_index()
    )
    queue = agg[agg["current_mpd_party_id"].isna()].copy()

    queue["display_name"] = (
        queue["partyname_ess"].fillna("")
        .where(queue["partyname_ess"].notna() & (queue["partyname_ess"].astype(str).str.strip() != ""), queue["partyname"])
        .where(lambda s: s.notna() & (s.astype(str).str.strip() != ""), queue["partyname_eng"])
        .fillna("")
    )
    queue["generic_skip"] = [
        is_generic_nonparty_option(a, b, c)
        for a, b, c in zip(queue["partyname_ess"], queue["partyname"], queue["partyname_eng"])
    ]
    queue["missing_display_name"] = queue["display_name"].apply(is_missing_display_name)
    queue["low_value_display_skip"] = queue["display_name"].apply(is_low_value_ballot_option_label)

    def choose_collect_skip_reason(row: pd.Series) -> str:
        if SKIP_MISSING_DISPLAY_NAME_ROWS and bool(row.get("missing_display_name")):
            return "missing_display_name"
        if SKIP_LOW_VALUE_BALLOT_OPTION_LABELS and bool(row.get("low_value_display_skip")):
            return "low_value_ballot_option"
        if SKIP_GENERIC_NONPARTY_OPTIONS and bool(row.get("generic_skip")):
            return "generic_nonparty_option"
        return ""

    queue["collect_skip_reason"] = queue.apply(choose_collect_skip_reason, axis=1)
    queue["edate_mpd_str"] = pd.to_datetime(queue["edate_mpd"], errors="coerce").dt.strftime("%Y-%m-%d")
    queue["target_id"] = (
        queue["iso3c"].fillna("").astype(str)
        + "__"
        + queue["edate_mpd_str"].fillna("").astype(str)
        + "__"
        + queue["ess_party_id"].astype("Int64").astype(str)
    )

    cand_groups = {
        (
            iso2c,
            pd.Timestamp(edate).normalize() if pd.notna(edate) else pd.NaT,
        ): sub.sort_values(["share_mpd", "mpd_party_id"], ascending=[False, True]).reset_index(drop=True)
        for (iso2c, edate), sub in candidate_table.groupby(["iso2c", "edate_mpd"], dropna=False, sort=False)
    }

    cand_texts: list[str] = []
    cand_jsons: list[str] = []
    cand_counts: list[int] = []
    top_ids: list[str] = []
    top_scores: list[float] = []

    for row in queue.itertuples(index=False):
        key = (
            clean_text_scalar(row.iso2c),
            pd.Timestamp(row.edate_mpd).normalize() if pd.notna(row.edate_mpd) else pd.NaT,
        )
        cand = cand_groups.get(key)
        if cand is None:
            cand = pd.DataFrame(columns=["mpd_party_id", "partyfacts_id_cand", "partyname_mpd", "share_mpd"])
        target_names = [
            clean_text_scalar(row.partyname_ess),
            clean_text_scalar(row.partyname),
            clean_text_scalar(row.partyname_eng),
        ]
        valid_target_names = [n for n in target_names if n]
        cand = cand.copy()
        if not cand.empty:
            if valid_target_names:
                cand["name_score"] = cand["partyname_mpd"].apply(lambda x: max(name_similarity(x, n) for n in valid_target_names))
            else:
                cand["name_score"] = 0.0
            cand["rank_by_share"] = np.arange(1, len(cand) + 1)
        else:
            cand["name_score"] = pd.Series(dtype=float)
            cand["rank_by_share"] = pd.Series(dtype=int)

        lines = []
        json_rows = []
        for rec in cand.itertuples(index=False):
            item = {
                "rank_by_share": int(rec.rank_by_share),
                "mpd_party_id": int(rec.mpd_party_id) if pd.notna(rec.mpd_party_id) else None,
                "partyfacts_id_cand": int(rec.partyfacts_id_cand) if pd.notna(rec.partyfacts_id_cand) else None,
                "partyname_mpd": clean_text_scalar(rec.partyname_mpd),
                "share_mpd": None if pd.isna(rec.share_mpd) else float(rec.share_mpd),
                "name_score": None if pd.isna(rec.name_score) else float(rec.name_score),
            }
            json_rows.append(item)
            lines.append(
                f"- rank_by_share={item['rank_by_share']}; mpd_party_id={item['mpd_party_id']}; "
                f"partyfacts_id_cand={item['partyfacts_id_cand']}; partyname_mpd={item['partyname_mpd']}; "
                f"share_mpd={item['share_mpd']}; name_score={item['name_score']}"
            )

        cand_texts.append("\n".join(lines))
        cand_jsons.append(json.dumps(json_rows, ensure_ascii=False))
        cand_counts.append(int(len(cand)))
        top_ids.append("|".join(str(int(x)) for x in cand["mpd_party_id"].head(5).dropna().tolist()))
        top_scores.append(float(cand["name_score"].max()) if not cand.empty else np.nan)

    queue["candidate_count"] = cand_counts
    queue["top5_mpd_ids"] = top_ids
    queue["best_name_score"] = top_scores
    queue["cand_text"] = cand_texts
    queue["cand_json"] = cand_jsons

    if SKIP_GENERIC_NONPARTY_OPTIONS or SKIP_MISSING_DISPLAY_NAME_ROWS or SKIP_LOW_VALUE_BALLOT_OPTION_LABELS:
        queue["collect_flag"] = queue["collect_skip_reason"].eq("")
    else:
        queue["collect_flag"] = True

    queue = queue.sort_values(
        ["collect_flag", "candidate_count", "n_voters", "best_name_score"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    if PILOT_LIMIT is not None:
        queue = queue.head(int(PILOT_LIMIT)).copy()
    return queue


# =============================================================================
# OpenAI helpers
# =============================================================================
def make_user_prompt(row: pd.Series) -> str:
    return f"""
Target unmatched ESS party-election case
- target_id: {row['target_id']}
- country: {clean_text_scalar(row.get('country'))}
- iso2c: {clean_text_scalar(row.get('iso2c'))}
- iso3c: {clean_text_scalar(row.get('iso3c'))}
- election_year: {row.get('election_year')}
- election_date: {clean_text_scalar(row.get('edate_mpd_str'))}
- ess_party_id: {row.get('ess_party_id')}
- partyfacts_id_from_ess_mapping: {row.get('partyfacts_id')}
- ess_party_name_raw: {clean_text_scalar(row.get('partyname_ess'))}
- partyfacts_party_name_local: {clean_text_scalar(row.get('partyname'))}
- partyfacts_party_name_english: {clean_text_scalar(row.get('partyname_eng'))}
- ess_vote_share_percent: {row.get('share_ess')}
- respondents_in_this_key: {row.get('n_voters')}
- generic_nonparty_option_flag: {bool(row.get('generic_skip'))}

Candidate MPD parties from the same country and election
{clean_text_scalar(row.get('cand_text'))}

Task
1) Decide whether one of the candidate MPD parties is clearly the same party/list/alliance as the ESS option for this election.
2) Use ONLY the candidate list as the allowed MPD IDs. Do not invent a new MPD party ID.
3) Prefer no match over a weak guess.
4) Use web search whenever the match is not obvious from names alone.
5) Prefer official or high-quality reference sources when available.
6) If the ESS option is generic (Other/Blank/DK/invalid/etc.), usually leave it unmatched.

Output rules
- resolution_status should be one of: matched_candidate, generic_nonparty_option, insufficient_evidence, no_clear_party_match.
- selected_from_candidate_list should be true only if you choose an MPD party from the list.
- candidate_rank_by_share should be the candidate rank from the list if you choose one, else null.
- confidence_0_100 should reflect evidence quality on a conservative scale: reserve 95+ for cases supported by at least two independent high-quality sources, or one official/reference source plus one independent non-Wikipedia source from another domain; treat official party websites as supportive but usually not enough by themselves for 90+; use roughly 70-89 for plausible but still checkable matches; use below 70 when evidence is thin or ambiguous.
- match_basis should be one of: exact_name, abbreviation, translation, transliteration, alliance_or_joint_list, merger_or_renaming, partyfacts_hint, other, no_match.
- nonmatch_reason_type should be one of: not_applicable, candidate_absent_from_allowed_list, generic_nonparty_option, insufficient_evidence, no_clear_party_match.
- If the ESS party/list is clearly identifiable but not present in the allowed MPD candidate list, set resolved=false, resolution_status=no_clear_party_match, nonmatch_reason_type=candidate_absent_from_allowed_list, and identified_party_if_absent to the best concise party/list name.
- If the case is a generic option (Other/Blank/DK/invalid/etc.), set nonmatch_reason_type=generic_nonparty_option.
- If you resolve a positive match, set nonmatch_reason_type=not_applicable and identified_party_if_absent=null.
- reason_summary should be concise, factual, and easy for a human to verify.
- uncertainty_note should briefly note any remaining ambiguity.
- Include 1-4 sources if you relied on outside verification; include zero sources only when the no-match decision is obvious from the provided data. When a source is used, fill in title and publisher whenever they can be inferred from the page or domain, and prefer at least one official or institutional source when available.
- Return JSON only.
""".strip()



def build_response_request_body(row: pd.Series, web_tool_type: str | None = None) -> dict[str, Any]:
    if not web_tool_type:
        web_tool_type = PREFERRED_WEB_SEARCH_TOOL_TYPES[0]
    return {
        "model": MODEL,
        "store": False,
        "include": ["web_search_call.action.sources"],
        "tools": [{"type": web_tool_type}],
        "tool_choice": "required",
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_user_prompt(row)},
        ],
        "text": {
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": "ess_mpd_match_record_rigorous",
                "schema": MATCH_SCHEMA,
                "strict": True,
            },
        },
    }


def response_obj_to_builtin(resp_obj: Any) -> Any:
    if resp_obj is None:
        return None
    try:
        if hasattr(resp_obj, "model_dump"):
            return resp_obj.model_dump()
    except Exception:
        pass
    try:
        if hasattr(resp_obj, "to_dict"):
            return resp_obj.to_dict()
    except Exception:
        pass
    if isinstance(resp_obj, dict):
        return resp_obj
    return resp_obj


def extract_response_text(resp_obj: Any) -> str:
    txt = clean_text_scalar(getattr(resp_obj, "output_text", ""))
    if txt:
        return txt.strip()

    data = response_obj_to_builtin(resp_obj)
    found: list[str] = []

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            if x.get("type") in {"output_text", "text"} and clean_text_scalar(x.get("text")):
                found.append(clean_text_scalar(x.get("text")))
            elif "text" in x and isinstance(x.get("text"), str):
                found.append(clean_text_scalar(x.get("text")))
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for item in x:
                walk(item)

    walk(data)
    for item in found:
        if item:
            return item.strip()
    return ""


def parse_json_from_text(text: str) -> dict[str, Any]:
    txt = clean_text_scalar(text).strip()
    if not txt:
        raise ValueError("empty model output")
    if txt.startswith("```"):
        txt = re.sub(r"^```(?:json)?\s*", "", txt, flags=re.I)
        txt = re.sub(r"\s*```$", "", txt)
    try:
        parsed = json.loads(txt)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = txt.find("{")
    end = txt.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(txt[start:end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("could not parse JSON object from model output")


def dedupe_source_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for rec in records:
        if not isinstance(rec, dict):
            continue
        url = clean_text_scalar(rec.get("url"))
        title = clean_text_scalar(rec.get("title"))
        if not url and not title:
            continue
        key = (url.lower(), title.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "title": title,
            "url": url,
            "publisher": clean_text_scalar(rec.get("publisher")),
            "why_relevant": clean_text_scalar(rec.get("why_relevant")),
        })
    return out


def extract_api_web_sources(resp_obj: Any) -> list[dict[str, Any]]:
    data = response_obj_to_builtin(resp_obj)
    out: list[dict[str, Any]] = []

    def add_source(src: Any) -> None:
        if not isinstance(src, dict):
            return
        out.append({
            "title": clean_text_scalar(src.get("title") or src.get("name")),
            "url": clean_text_scalar(src.get("url") or src.get("href")),
            "publisher": clean_text_scalar(src.get("publisher") or src.get("site_name") or src.get("domain")),
            "why_relevant": clean_text_scalar(src.get("why_relevant")),
        })

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            if clean_text_scalar(x.get("type")) == "web_search_call":
                action = x.get("action")
                if isinstance(action, dict):
                    sources = action.get("sources")
                    if isinstance(sources, list):
                        for src in sources:
                            add_source(src)
            if clean_text_scalar(x.get("type")) == "url_citation":
                add_source(x)
            sources = x.get("sources")
            if isinstance(sources, list):
                for src in sources:
                    add_source(src)
            annotations = x.get("annotations")
            if isinstance(annotations, list):
                for ann in annotations:
                    if isinstance(ann, dict) and clean_text_scalar(ann.get("type")) == "url_citation":
                        add_source(ann)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for item in x:
                walk(item)

    walk(data)
    return dedupe_source_records(out)



def normalize_source_url(url: Any) -> str:
    txt = clean_text_scalar(url)
    if not txt:
        return ""
    try:
        parsed = urlparse(txt)
        scheme = clean_text_scalar(parsed.scheme).lower() or "https"
        netloc = clean_text_scalar(parsed.netloc).lower().replace("www.", "")
        path = re.sub(r"/+", "/", clean_text_scalar(parsed.path))
        path = path.rstrip("/")
        query = clean_text_scalar(parsed.query)
        if query:
            kept = []
            for part in query.split("&"):
                low = part.lower()
                if low.startswith("utm_") or low.startswith("fbclid=") or low.startswith("gclid="):
                    continue
                kept.append(part)
            query = "&".join(kept)
        out = f"{scheme}://{netloc}{path}"
        if query:
            out += f"?{query}"
        return out
    except Exception:
        return txt.strip()


def normalize_source_key(rec: dict[str, Any]) -> str:
    url = normalize_source_url(rec.get("url"))
    title = clean_text_scalar(rec.get("title")).lower()
    if url:
        return f"url::{url}"
    if title:
        return f"title::{title}"
    return ""


def infer_publisher_from_domain(domain: str) -> str:
    dom = clean_text_scalar(domain).lower().replace("www.", "")
    mapping = {
        "wikipedia.org": "Wikipedia",
        "wikimedia.org": "Wikimedia",
        "data.ipu.org": "Inter-Parliamentary Union (IPU)",
        "idi.org.il": "Israel Democracy Institute",
        "parlgov.fly.dev": "ParlGov",
        "boe.es": "Boletín Oficial del Estado (BOE)",
        "rtve.es": "RTVE",
        "eitb.eus": "EITB",
        "europapress.es": "Europa Press",
        "congress.gov": "Congress.gov",
        "juntaelectoralcentral.es": "Junta Electoral Central",
        "infoelectoral.interior.gob.es": "Ministerio del Interior (España)",
        "electionguide.org": "IFES Election Guide",
        "manifesto-project.wzb.eu": "Manifesto Project",
        "robert-schuman.eu": "Fondation Robert Schuman",
        "odihr.osce.org": "OSCE/ODIHR",
        "osce.org": "OSCE",
        "gov.il": "Gov.il",
    }
    for key, val in mapping.items():
        if dom == key or dom.endswith("." + key):
            return val
    if not dom:
        return ""
    return dom


HIGH_GRADE_SOURCE_TAGS: set[str] = {
    "official_public_source",
    "intergovernmental_reference",
    "institutional_reference",
}
INDEPENDENT_NONWIKI_SOURCE_TAGS: set[str] = set(HIGH_GRADE_SOURCE_TAGS) | {"news_media"}
SUPPORTIVE_NONWIKI_SOURCE_TAGS: set[str] = set(INDEPENDENT_NONWIKI_SOURCE_TAGS) | {"official_party_source"}


def _domain_matches(domain: str, candidates: Iterable[str]) -> bool:
    dom = clean_text_scalar(domain).lower().replace("www.", "")
    for cand in candidates:
        c = clean_text_scalar(cand).lower().replace("www.", "")
        if not c:
            continue
        if dom == c or dom.endswith("." + c):
            return True
    return False


def classify_source_quality(rec: dict[str, Any]) -> tuple[str, int]:
    domain = clean_text_scalar(rec.get("domain")).lower().replace("www.", "")
    publisher = clean_text_scalar(rec.get("publisher")).lower()
    title = clean_text_scalar(rec.get("title")).lower()
    why = clean_text_scalar(rec.get("why_relevant")).lower()
    merged = " | ".join([publisher, title, why])

    # Classify encyclopedia-like domains first so pages that mention
    # "official results", "parliament", or similar terms are not mis-tagged
    # as public or party sources.
    if _domain_matches(domain, {"wikipedia.org", "wikimedia.org"}):
        return "encyclopedia", 45
    if not domain and any(tok in publisher for tok in {"wikipedia", "wikimedia"}):
        return "encyclopedia", 45

    intergovernmental_domains = {"data.ipu.org", "ipu.org", "archive.ipu.org", "osce.org", "odihr.osce.org", "oscepa.org", "coe.int"}
    intergovernmental_tokens = [
        "inter-parliamentary union",
        "osce/odihr",
        "organization for security and co-operation in europe",
        "organization for security and cooperation in europe",
        "office for democratic institutions and human rights",
        "odihr",
        "council of europe",
        "ipu parline",
        "parline",
    ]
    if _domain_matches(domain, intergovernmental_domains) or any(tok in merged for tok in intergovernmental_tokens):
        return "intergovernmental_reference", 96

    institutional_domains = {
        "idi.org.il",
        "parlgov.fly.dev",
        "parlgov",
        "idea.int",
        "ifes.org",
        "electionguide.org",
        "israeled.org",
        "crsreports.congress.gov",
        "constitutionnet.org",
        "parline.org",
        "parties-and-elections.eu",
        "manifesto-project.wzb.eu",
        "wzb.eu",
        "robert-schuman.eu",
    }
    institutional_text_tokens = [
        "israel democracy institute",
        "parlgov",
        "manifesto project",
        "ifes election guide",
        "election guide",
        "international institute for democracy",
        "fondation robert schuman",
        "manifesto project dataset",
    ]
    if _domain_matches(domain, institutional_domains) or any(tok in merged for tok in institutional_text_tokens):
        return "institutional_reference", 88

    news_domains = [
        "reuters.com", "apnews.com", "bbc.", "dw.com", "elpais.com",
        "rtve.es", "eitb.eus", "europapress.es", "theguardian.com",
        "nytimes.com", "timesofisrael.com", "jpost.com", "axios.com",
        "sputnikportal.", "sputniknews.", "lemonde.fr", "politico.eu",
    ]
    if any(tok in domain for tok in news_domains):
        return "news_media", 66

    official_public_exact_domains = {
        "juntaelectoralcentral.es",
        "infoelectoral.interior.gob.es",
        "boe.es",
        "congress.gov",
        "gov.il",
        "knesset.gov.il",
        "main.knesset.gov.il",
        "bechirot.gov.il",
        "bechirot24.bechirot.gov.il",
        "senato.it",
        "camera.it",
        "kiesraad.nl",
        "english.kiesraad.nl",
        "dekamer.be",
        "ch.ch",
        "ge.ch",
        "bk.admin.ch",
        "swissstats.bfs.admin.ch",
        "bfs.admin.ch",
        "aboutswitzerland.eda.admin.ch",
        "eda.admin.ch",
        "arhiva.rik.parlament.gov.rs",
    }
    official_public_suffix_domains = {
        "admin.ch",
        "gov.il",
        "gov.rs",
        "parlament.gov.rs",
        "knesset.gov.il",
        "bechirot.gov.il",
    }
    official_public_domain_tokens = [
        ".gov", "gov.", ".gob", "gob.", "europa.eu", "europarl",
        "interior.gob", "juntaelectoralcentral", "electoralcommission",
        "electoral-commission", "electionresults", "parliament.uk",
        "senato.it", "camera.it", "kiesraad.nl", "dekamer.be", ".admin.ch",
    ]
    official_public_publisher_tokens = [
        "electoral commission",
        "election commission",
        "official election results",
        "official results",
        "junta electoral central",
        "ministry of interior",
        "ministerio del interior",
        "central election commission",
        "boletín oficial del estado",
        "boletin oficial del estado",
        "senato della repubblica",
        "camera dei deputati",
        "kiesraad",
        "chamber of representatives",
        "house of representatives",
        "federal public service interior",
        "federal public service",
        "federal chancellery",
        "schweizerische bundeskanzlei",
        "federal statistical office",
        "department of foreign affairs",
        "foreign affairs (about switzerland)",
        "swiss confederation",
        "république et canton",
        "republique et canton",
        "canton de genève",
        "canton de geneve",
        "belgian chamber of representatives",
    ]
    official_public_title_tokens = [
        "official election results",
        "official results",
        "the 24th knesset elections",
        "members of the twenty-fourth knesset",
        "how the national council is elected",
        "nationalratswahlen",
        "election du conseil national",
        "elections to the house of representatives",
        "source : service public fédéral intérieur",
        "source : service public federal interieur",
    ]

    has_public_domain = (
        _domain_matches(domain, official_public_exact_domains)
        or any(domain == s or domain.endswith("." + s) for s in official_public_suffix_domains)
        or any(tok in domain for tok in official_public_domain_tokens)
    )
    looks_public_by_text = (
        any(tok in merged for tok in official_public_publisher_tokens)
        or any(tok in merged for tok in official_public_title_tokens)
    )
    if has_public_domain or (looks_public_by_text and not domain):
        return "official_public_source", 100

    official_party_text_tokens = [
        "official party page",
        "official party website",
        "official page",
        "official website",
        "party website",
        "party page",
    ]
    if any(tok in merged for tok in official_party_text_tokens) or (
        "official" in merged and any(tok in merged for tok in ["party", "partido", "movement", "coalition", "list"])
    ):
        return "official_party_source", 76

    if publisher or title or why:
        return "other_web", 55
    return "other_web", 40

def merge_source_lists(parsed_sources: list[dict[str, Any]], api_sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    def upsert(src: Any, origin: str) -> None:
        if not isinstance(src, dict):
            return
        candidate = {
            "title": clean_text_scalar(src.get("title")),
            "url": normalize_source_url(src.get("url")),
            "publisher": clean_text_scalar(src.get("publisher")),
            "why_relevant": clean_text_scalar(src.get("why_relevant")),
        }
        key = normalize_source_key(candidate)
        if not key:
            return
        if key not in merged:
            domain = clean_text_scalar(urlparse(candidate["url"]).netloc).lower().replace("www.", "")
            candidate["domain"] = domain
            if not candidate["publisher"]:
                candidate["publisher"] = infer_publisher_from_domain(domain)
            candidate["_from_model"] = 1 if origin == "model" else 0
            candidate["_from_api"] = 1 if origin == "api" else 0
            tag, score = classify_source_quality(candidate)
            candidate["quality_tag"] = tag
            candidate["quality_score"] = int(score)
            candidate["_completeness"] = sum(bool(clean_text_scalar(candidate.get(k))) for k in ["title", "publisher", "why_relevant"])
            merged[key] = candidate
            order.append(key)
            return

        existing = merged[key]
        existing["_from_model"] = max(int(existing.get("_from_model", 0)), 1 if origin == "model" else 0)
        existing["_from_api"] = max(int(existing.get("_from_api", 0)), 1 if origin == "api" else 0)
        for field in ["title", "url", "publisher"]:
            if not clean_text_scalar(existing.get(field)) and clean_text_scalar(candidate.get(field)):
                existing[field] = candidate[field]
        old_why = clean_text_scalar(existing.get("why_relevant"))
        new_why = clean_text_scalar(candidate.get("why_relevant"))
        if new_why and (not old_why or len(new_why) > len(old_why)):
            existing["why_relevant"] = new_why
        if not clean_text_scalar(existing.get("publisher")):
            dom = clean_text_scalar(existing.get("domain"))
            existing["publisher"] = infer_publisher_from_domain(dom)
        tag, score = classify_source_quality(existing)
        existing["quality_tag"] = tag
        existing["quality_score"] = int(score)
        existing["_completeness"] = sum(bool(clean_text_scalar(existing.get(k))) for k in ["title", "publisher", "why_relevant"])

    for src in parsed_sources:
        upsert(src, origin="model")
    for src in api_sources:
        upsert(src, origin="api")

    records = [merged[k] for k in order]

    def sort_key(rec: dict[str, Any]) -> tuple[int, int, int, int, int]:
        from_model = int(rec.get("_from_model", 0))
        why_len = len(clean_text_scalar(rec.get("why_relevant")))
        has_title = 1 if clean_text_scalar(rec.get("title")) else 0
        completeness = int(rec.get("_completeness", 0))
        quality = int(rec.get("quality_score", 0))
        return (
            from_model,
            has_title,
            1 if why_len > 0 else 0,
            completeness,
            quality,
        )

    records = sorted(records, key=sort_key, reverse=True)

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    domain_counts: dict[str, int] = {}

    def maybe_add(rec: dict[str, Any], require_model: bool | None = None, require_context: bool = False) -> None:
        if len(selected) >= 4:
            return
        key = normalize_source_key(rec)
        if not key or key in seen:
            return
        dom = clean_text_scalar(rec.get("domain"))
        has_context = bool(clean_text_scalar(rec.get("title")) or clean_text_scalar(rec.get("why_relevant")))
        if require_model is True and not bool(rec.get("_from_model")):
            return
        if require_model is False and bool(rec.get("_from_model")):
            return
        if require_context and not has_context:
            return
        max_per_domain = 2 if bool(rec.get("_from_model")) else 1
        if dom and domain_counts.get(dom, 0) >= max_per_domain and len(records) > 4:
            return
        selected.append(rec)
        seen.add(key)
        if dom:
            domain_counts[dom] = domain_counts.get(dom, 0) + 1

    for rec in records:
        maybe_add(rec, require_model=True, require_context=True)
    for rec in records:
        maybe_add(rec, require_model=False, require_context=True)

    if len(selected) < 1:
        for rec in records:
            maybe_add(rec, require_model=None, require_context=False)

    cleaned: list[dict[str, Any]] = []
    for rec in selected[:4]:
        cleaned.append({
            "title": clean_text_scalar(rec.get("title")),
            "url": normalize_source_url(rec.get("url")),
            "publisher": clean_text_scalar(rec.get("publisher")),
            "why_relevant": clean_text_scalar(rec.get("why_relevant")),
            "domain": clean_text_scalar(rec.get("domain")),
            "quality_tag": clean_text_scalar(rec.get("quality_tag")),
            "quality_score": int(pd.to_numeric(pd.Series([rec.get("quality_score")]), errors="coerce").fillna(0).iloc[0]),
        })
    return cleaned


def choose_effective_sources(parsed_sources: list[dict[str, Any]], api_sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return merge_source_lists(parsed_sources, api_sources)


def create_response_with_tool_fallback(client: Any, row: pd.Series) -> tuple[Any, str]:
    last_exc: Exception | None = None
    tried: list[str] = []
    for web_tool_type in PREFERRED_WEB_SEARCH_TOOL_TYPES:
        body = build_response_request_body(row, web_tool_type=web_tool_type)
        for attempt in range(5):
            try:
                resp_obj = client.responses.create(**body)
                return resp_obj, web_tool_type
            except Exception as exc:
                last_exc = exc
                msg = clean_text_scalar(exc).lower()
                wrong_tool = ("tool" in msg and "web_search" in msg and "invalid" in msg) or ("unsupported" in msg and "web_search" in msg)
                if wrong_tool:
                    tried.append(web_tool_type)
                    break
                time.sleep(min(2 ** attempt, 20))
        else:
            tried.append(web_tool_type)
            continue
    raise RuntimeError(f"Responses API call failed after trying tool types {tried or PREFERRED_WEB_SEARCH_TOOL_TYPES}: {last_exc!r}")


def write_batch_jsonl(queue: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for _, row in queue.iterrows():
            if not bool(row.get("collect_flag", True)):
                continue
            req = {
                "custom_id": row["target_id"],
                "method": "POST",
                "url": "/v1/responses",
                "body": build_response_request_body(row),
            }
            f.write(json_dumps_compact(req) + "\n")



def collect_with_openai(queue: pd.DataFrame, out_jsonl: Path, out_csv: Path) -> None:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("The `openai` package is required. Install it with `pip install openai`.") from exc

    if not OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY is required for RUN_COLLECT=True.")

    client = OpenAI(api_key=OPENAI_API_KEY)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    completed: set[str] = set()
    flat_rows: list[dict[str, Any]] = []
    if out_csv.exists():
        try:
            existing = pd.read_csv(out_csv)
            if "target_id" in existing.columns:
                completed.update(existing["target_id"].astype(str).tolist())
                flat_rows = existing.to_dict(orient="records")
        except Exception:
            pass

    def flush() -> None:
        if flat_rows:
            pd.DataFrame(flat_rows).to_csv(out_csv, index=False)

    with out_jsonl.open("a", encoding="utf-8") as fout:
        for _, row in queue.iterrows():
            if not bool(row.get("collect_flag", True)):
                continue
            tid = str(row["target_id"])
            if tid in completed:
                continue

            try:
                resp_obj, used_web_tool_type = create_response_with_tool_fallback(client, row)
            except Exception as exc:
                rec = {"target_id": tid, "status": "error", "error": repr(exc)}
                fout.write(json_dumps_compact(rec) + "\n")
                fout.flush()
                continue

            output_text = extract_response_text(resp_obj)
            api_sources = extract_api_web_sources(resp_obj)
            usage = extract_response_usage(resp_obj)
            tool_calls_web_search = count_web_search_calls(resp_obj)

            if not output_text:
                rec = {
                    "target_id": tid,
                    "status": "no_output_text",
                    "usage": usage,
                    "tool_calls_web_search": tool_calls_web_search,
                    "api_sources_json": json.dumps(api_sources, ensure_ascii=False),
                    "used_web_tool_type": used_web_tool_type,
                }
                fout.write(json_dumps_compact(rec) + "\n")
                fout.flush()
                continue

            try:
                parsed = parse_json_from_text(output_text)
            except Exception as exc:
                rec = {
                    "target_id": tid,
                    "status": "invalid_json",
                    "error": repr(exc),
                    "output_text": output_text,
                    "usage": usage,
                    "tool_calls_web_search": tool_calls_web_search,
                    "api_sources_json": json.dumps(api_sources, ensure_ascii=False),
                    "used_web_tool_type": used_web_tool_type,
                }
                fout.write(json_dumps_compact(rec) + "\n")
                fout.flush()
                continue

            model_sources = parsed.get("sources", [])
            effective_sources = choose_effective_sources(model_sources, api_sources)

            flat = {
                "target_id": tid,
                "status": "ok",
                "resolved": parsed.get("resolved"),
                "resolution_status": clean_text_scalar(parsed.get("resolution_status")),
                "selected_mpd_party_id": parsed.get("selected_mpd_party_id"),
                "selected_party_name_mpd": clean_text_scalar(parsed.get("selected_party_name_mpd")),
                "selected_from_candidate_list": parsed.get("selected_from_candidate_list"),
                "candidate_rank_by_share": parsed.get("candidate_rank_by_share"),
                "model_confidence_0_100": parsed.get("confidence_0_100"),
                "match_basis": clean_text_scalar(parsed.get("match_basis")),
                "nonmatch_reason_type": clean_text_scalar(parsed.get("nonmatch_reason_type")),
                "identified_party_if_absent": clean_text_scalar(parsed.get("identified_party_if_absent")),
                "reason_summary": clean_text_scalar(parsed.get("reason_summary")),
                "uncertainty_note": clean_text_scalar(parsed.get("uncertainty_note")),
                "model_sources_json": json.dumps(model_sources, ensure_ascii=False),
                "api_sources_json": json.dumps(api_sources, ensure_ascii=False),
                "effective_sources_json": json.dumps(effective_sources, ensure_ascii=False),
                "tool_calls_web_search": tool_calls_web_search,
                "used_web_tool_type": used_web_tool_type,
                **usage,
                "iso3c": clean_text_scalar(row.get("iso3c")),
                "iso2c": clean_text_scalar(row.get("iso2c")),
                "country": clean_text_scalar(row.get("country")),
                "election_year": row.get("election_year"),
                "edate_mpd": clean_text_scalar(row.get("edate_mpd_str")),
                "ess_party_id": row.get("ess_party_id"),
                "partyfacts_id": row.get("partyfacts_id"),
                "partyname_ess": clean_text_scalar(row.get("partyname_ess")),
                "partyname": clean_text_scalar(row.get("partyname")),
                "partyname_eng": clean_text_scalar(row.get("partyname_eng")),
                "share_ess": row.get("share_ess"),
                "n_voters": row.get("n_voters"),
                "candidate_count": row.get("candidate_count"),
                "best_name_score": row.get("best_name_score"),
                "generic_skip": row.get("generic_skip"),
            }
            flat_rows.append(flat)
            rec = {
                "target_id": tid,
                "status": "ok",
                "parsed": parsed,
                "api_sources": api_sources,
                "effective_sources": effective_sources,
                "usage": usage,
                "tool_calls_web_search": tool_calls_web_search,
                "used_web_tool_type": used_web_tool_type,
            }
            fout.write(json_dumps_compact(rec) + "\n")
            fout.flush()
            flush()
            if SLEEP_SECONDS > 0:
                time.sleep(SLEEP_SECONDS)


# =============================================================================
# Review helpers

# =============================================================================
def parse_sources_json(blob: Any) -> list[dict[str, Any]]:
    if blob is None:
        return []
    if isinstance(blob, list):
        return [x for x in blob if isinstance(x, dict)]
    txt = clean_text_scalar(blob)
    if not txt:
        return []
    try:
        parsed = json.loads(txt)
    except Exception:
        return []
    if isinstance(parsed, list):
        return [x for x in parsed if isinstance(x, dict)]
    return []



def flatten_sources_columns(df: pd.DataFrame, col: str = "effective_sources_json", max_items: int = 4) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for blob in df[col].tolist():
        sources = parse_sources_json(blob)
        rec: dict[str, Any] = {}
        domains: list[str] = []
        quality_counts: dict[str, int] = {}
        official_count = 0
        official_or_reference_count = 0
        high_grade_count = 0
        high_grade_domains: set[str] = set()
        independent_non_wiki_count = 0
        independent_non_wiki_domains: set[str] = set()
        official_party_source_count = 0
        non_wiki_count = 0
        for i in range(max_items):
            src = sources[i] if i < len(sources) else {}
            url = normalize_source_url(src.get("url"))
            domain = clean_text_scalar(src.get("domain"))
            if not domain:
                domain = clean_text_scalar(urlparse(url).netloc).lower().replace("www.", "")
            provided_quality_tag = clean_text_scalar(src.get("quality_tag"))
            quality_tag = ""
            quality_score_guess = None
            if src:
                quality_tag, quality_score_guess = classify_source_quality({
                    "title": src.get("title"),
                    "url": url,
                    "publisher": src.get("publisher"),
                    "why_relevant": src.get("why_relevant"),
                    "domain": domain,
                })
            quality_score = pd.to_numeric(pd.Series([quality_score_guess]), errors="coerce").fillna(0).iloc[0]
            if domain:
                domains.append(domain)
            if src:
                normalized_quality = quality_tag or "other_web"
                quality_counts[normalized_quality] = quality_counts.get(normalized_quality, 0) + 1
                if quality_tag in {"official_public_source", "intergovernmental_reference"}:
                    official_count += 1
                if quality_tag in HIGH_GRADE_SOURCE_TAGS:
                    official_or_reference_count += 1
                    high_grade_count += 1
                    if domain:
                        high_grade_domains.add(domain)
                if quality_tag in INDEPENDENT_NONWIKI_SOURCE_TAGS:
                    independent_non_wiki_count += 1
                    if domain:
                        independent_non_wiki_domains.add(domain)
                if quality_tag == "official_party_source":
                    official_party_source_count += 1
                if quality_tag != "encyclopedia":
                    non_wiki_count += 1
            j = i + 1
            rec[f"source_{j}_title"] = clean_text_scalar(src.get("title"))
            rec[f"source_{j}_url"] = url
            rec[f"source_{j}_publisher"] = clean_text_scalar(src.get("publisher"))
            rec[f"source_{j}_why"] = clean_text_scalar(src.get("why_relevant"))
            rec[f"source_{j}_domain"] = domain
            rec[f"source_{j}_quality_tag"] = quality_tag
            rec[f"source_{j}_quality_score"] = int(quality_score) if not pd.isna(quality_score) else 0
            rec[f"source_{j}_provided_quality_tag"] = provided_quality_tag
        rec["source_count"] = len(sources)
        rec["source_domains"] = "; ".join([d for d in domains if d])
        rec["distinct_source_domain_count"] = len({d for d in domains if d})
        rec["official_source_count"] = int(official_count)
        rec["official_or_reference_source_count"] = int(official_or_reference_count)
        rec["high_grade_source_count"] = int(high_grade_count)
        rec["high_grade_distinct_domain_count"] = int(len(high_grade_domains))
        rec["independent_non_wiki_source_count"] = int(independent_non_wiki_count)
        rec["independent_non_wiki_distinct_domain_count"] = int(len(independent_non_wiki_domains))
        rec["official_party_source_count"] = int(official_party_source_count)
        rec["non_wiki_source_count"] = int(non_wiki_count)
        rec["source_quality_summary"] = "; ".join([f"{k}:{v}" for k, v in sorted(quality_counts.items()) if k])
        rows.append(rec)
    return pd.DataFrame(rows, index=df.index)

def candidate_id_allowed(row: pd.Series) -> bool:
    sel = row.get("selected_mpd_party_id")
    if pd.isna(sel):
        return False
    try:
        sel_int = int(float(sel))
    except Exception:
        return False
    blob = clean_text_scalar(row.get("cand_json"))
    if not blob:
        return False
    try:
        parsed = json.loads(blob)
    except Exception:
        return False
    allowed = {int(item["mpd_party_id"]) for item in parsed if item.get("mpd_party_id") is not None}
    return sel_int in allowed




def lookup_selected_candidate_fields(row: pd.Series) -> dict[str, Any]:
    sel = row.get("selected_mpd_party_id")
    if pd.isna(sel):
        return {}
    try:
        sel_int = int(float(sel))
    except Exception:
        return {}
    blob = clean_text_scalar(row.get("cand_json"))
    if not blob:
        return {}
    try:
        parsed = json.loads(blob)
    except Exception:
        return {}
    if not isinstance(parsed, list):
        return {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        mpd_id = item.get("mpd_party_id")
        try:
            mpd_int = int(float(mpd_id)) if mpd_id is not None else None
        except Exception:
            mpd_int = None
        if mpd_int == sel_int:
            return item
    return {}


def classify_vote_share_gap_band(abs_gap: Any) -> str:
    gap = pd.to_numeric(pd.Series([abs_gap]), errors="coerce").iloc[0]
    if pd.isna(gap):
        return ""
    gap = float(gap)
    if gap <= 2:
        return "within_2pp"
    if gap <= 5:
        return "within_5pp"
    if gap <= 10:
        return "within_10pp"
    return "over_10pp"


def build_vote_share_compare_note(row: pd.Series) -> str:
    resolved = bool(row.get("resolved")) if not pd.isna(row.get("resolved")) else False
    if not resolved:
        return ""
    ess = pd.to_numeric(pd.Series([row.get("share_ess")]), errors="coerce").iloc[0]
    mpd = pd.to_numeric(pd.Series([row.get("matched_share_mpd")]), errors="coerce").iloc[0]
    if pd.isna(ess) or pd.isna(mpd):
        return ""
    gap = float(ess) - float(mpd)
    sign = "+" if gap > 0 else ""
    band = clean_text_scalar(row.get("vote_share_gap_band"))
    band_text = f"; {band}" if band else ""
    return f"ESS {float(ess):.2f}% vs MPD {float(mpd):.2f}% ({sign}{gap:.2f} pp{band_text})"


def classify_vote_share_soft_alert(abs_gap: Any) -> str:
    gap = pd.to_numeric(pd.Series([abs_gap]), errors="coerce").iloc[0]
    if pd.isna(gap):
        return ""
    gap = float(gap)
    if gap > 10:
        return "very_large_gap_check"
    if gap > 5:
        return "large_gap_review"
    return ""


def build_vote_share_soft_alert_note(row: pd.Series) -> str:
    alert = clean_text_scalar(row.get("vote_share_gap_soft_alert"))
    if not alert:
        return ""
    ess = pd.to_numeric(pd.Series([row.get("share_ess")]), errors="coerce").iloc[0]
    mpd = pd.to_numeric(pd.Series([row.get("matched_share_mpd")]), errors="coerce").iloc[0]
    if pd.isna(ess) or pd.isna(mpd):
        return ""
    gap = float(ess) - float(mpd)
    sign = "+" if gap > 0 else ""
    base = f"ESS {float(ess):.2f}% vs MPD {float(mpd):.2f}% ({sign}{gap:.2f} pp)"
    if alert == "very_large_gap_check":
        return f"Very large ESS-MPD share gap; caution flag only, not an automatic rejection: {base}."
    return f"Large ESS-MPD share gap; review carefully before applying: {base}."


def compute_trust_cap_and_flags(row: pd.Series) -> tuple[int, list[str]]:
    flags: list[str] = []
    resolved = bool(row.get("resolved")) if not pd.isna(row.get("resolved")) else False
    status = clean_text_scalar(row.get("resolution_status"))
    if not resolved or status in {"generic_nonparty_option", "insufficient_evidence", "no_clear_party_match"}:
        return 0, ["no_positive_match"]

    if not bool(row.get("candidate_allowed")):
        return 0, ["selected_candidate_not_in_allowed_list"]

    if not bool(row.get("selected_from_candidate_list")):
        return 0, ["selected_from_candidate_list_false"]

    cap = 100

    if bool(row.get("generic_skip")):
        cap = min(cap, 25)
        flags.append("generic_option_flagged")

    source_count = int(pd.to_numeric(pd.Series([row.get("source_count")]), errors="coerce").fillna(0).iloc[0])
    distinct_domains = int(pd.to_numeric(pd.Series([row.get("distinct_source_domain_count")]), errors="coerce").fillna(0).iloc[0])
    official_count = int(pd.to_numeric(pd.Series([row.get("official_source_count")]), errors="coerce").fillna(0).iloc[0])
    official_or_reference_count = int(pd.to_numeric(pd.Series([row.get("official_or_reference_source_count")]), errors="coerce").fillna(0).iloc[0])
    high_grade_count = int(pd.to_numeric(pd.Series([row.get("high_grade_source_count")]), errors="coerce").fillna(official_or_reference_count).iloc[0])
    high_grade_domains = int(pd.to_numeric(pd.Series([row.get("high_grade_distinct_domain_count")]), errors="coerce").fillna(0).iloc[0])
    independent_non_wiki_count = int(pd.to_numeric(pd.Series([row.get("independent_non_wiki_source_count")]), errors="coerce").fillna(0).iloc[0])
    independent_non_wiki_domains = int(pd.to_numeric(pd.Series([row.get("independent_non_wiki_distinct_domain_count")]), errors="coerce").fillna(0).iloc[0])
    official_party_source_count = int(pd.to_numeric(pd.Series([row.get("official_party_source_count")]), errors="coerce").fillna(0).iloc[0])
    non_wiki_count = int(pd.to_numeric(pd.Series([row.get("non_wiki_source_count")]), errors="coerce").fillna(0).iloc[0])
    match_basis = clean_text_scalar(row.get("match_basis"))

    if source_count <= 0:
        cap = min(cap, 55)
        flags.append("no_sources")
    elif source_count == 1:
        cap = min(cap, 72)
        flags.append("only_one_source")

    if source_count >= 2 and distinct_domains <= 1:
        cap = min(cap, 88)
        flags.append("single_domain_source_set")

    if source_count > 0 and non_wiki_count <= 0:
        cap = min(cap, 72)
        flags.append("only_wiki_sources")
    elif source_count > 0 and independent_non_wiki_count <= 0:
        cap = min(cap, 82)
        flags.append("no_independent_nonwiki_source")
    elif source_count > 0 and official_or_reference_count <= 0:
        cap = min(cap, 84)
        flags.append("no_official_or_reference_source")

    if official_party_source_count > 0 and official_or_reference_count <= 0:
        cap = min(cap, 78)
        flags.append("only_party_official_support_without_independent_reference")

    best_name_score = pd.to_numeric(pd.Series([row.get("best_name_score")]), errors="coerce").iloc[0]
    if match_basis in {"exact_name", "abbreviation"}:
        if pd.isna(best_name_score):
            cap = min(cap, 88)
            flags.append("missing_name_similarity")
        else:
            score = float(best_name_score)
            if score < 0.35:
                cap = min(cap, 82)
                flags.append("low_name_similarity_for_exactish_match")
            elif score < 0.60:
                cap = min(cap, 92)
                flags.append("moderate_name_similarity_for_exactish_match")

    elif match_basis in {"translation", "transliteration"}:
        if pd.isna(best_name_score):
            cap = min(cap, 85)
            flags.append("missing_name_similarity")
        else:
            score = float(best_name_score)
            if score < 0.20:
                cap = min(cap, 74)
                flags.append("very_low_name_similarity_for_translation_match")
            elif score < 0.40:
                cap = min(cap, 84)
                flags.append("low_name_similarity_for_translation_match")
        if official_or_reference_count <= 0:
            cap = min(cap, 86)
            flags.append("translation_match_without_reference_source")
        if high_grade_count < 2 and independent_non_wiki_count < 2:
            cap = min(cap, 89)
            flags.append("translation_match_without_two_independent_nonwiki_sources")

    elif match_basis == "partyfacts_hint":
        cap = min(cap, 88)
        flags.append("partyfacts_hint_needs_independent_check")
        if pd.isna(best_name_score) or float(best_name_score) < 0.55:
            cap = min(cap, 82)
            flags.append("partyfacts_hint_with_weak_name_similarity")
        if official_or_reference_count <= 0:
            cap = min(cap, 78)
            flags.append("partyfacts_hint_without_reference_source")

    elif match_basis in {"alliance_or_joint_list", "merger_or_renaming"}:
        cap = min(cap, 88)
        flags.append("structurally_complex_match")
        if official_or_reference_count <= 0:
            cap = min(cap, 80)
            flags.append("complex_match_without_reference_source")
        if not pd.isna(best_name_score) and float(best_name_score) < 0.35:
            cap = min(cap, 72)
            flags.append("complex_match_with_weak_name_similarity")

    else:
        if pd.isna(best_name_score):
            cap = min(cap, 80)
            flags.append("missing_name_similarity")
        else:
            score = float(best_name_score)
            if score < 0.30:
                cap = min(cap, 70)
                flags.append("very_low_name_similarity")
            elif score < 0.45:
                cap = min(cap, 78)
                flags.append("low_name_similarity")

    if match_basis in {"exact_name", "abbreviation", "translation", "transliteration"}:
        if official_or_reference_count >= 1 and independent_non_wiki_count < 2:
            cap = min(cap, 89)
            flags.append("not_enough_independent_nonwiki_verification_for_auto")
        if high_grade_count >= 2 and high_grade_domains < 2:
            cap = min(cap, 93)
            flags.append("high_grade_sources_not_cross_domain")
        if independent_non_wiki_count >= 2 and independent_non_wiki_domains < 2:
            cap = min(cap, 91)
            flags.append("independent_nonwiki_sources_not_cross_domain")

    rank = pd.to_numeric(pd.Series([row.get("candidate_rank_by_share")]), errors="coerce").iloc[0]
    if not pd.isna(rank):
        r = int(rank)
        if match_basis in {"alliance_or_joint_list", "merger_or_renaming", "partyfacts_hint"}:
            if r > 15:
                cap = min(cap, 86)
                flags.append("candidate_far_down_share_ranking")
            elif r > 10:
                cap = min(cap, 92)
                flags.append("candidate_outside_top10_by_share")
        else:
            if r > 20:
                cap = min(cap, 88)
                flags.append("candidate_far_down_share_ranking")
            elif r > 12:
                cap = min(cap, 94)
                flags.append("candidate_outside_top12_by_share")

    web_calls = int(pd.to_numeric(pd.Series([row.get("tool_calls_web_search")]), errors="coerce").fillna(0).iloc[0])
    if web_calls <= 0 and source_count <= 0:
        cap = min(cap, 50)
        flags.append("no_external_verification")

    return int(cap), flags

def infer_nonmatch_reason_type(row: pd.Series) -> str:
    explicit = clean_text_scalar(row.get("nonmatch_reason_type"))
    if explicit:
        return explicit
    resolved = bool(row.get("resolved")) if not pd.isna(row.get("resolved")) else False
    if resolved:
        return "not_applicable"
    if bool(row.get("generic_skip")):
        return "generic_nonparty_option"
    status = clean_text_scalar(row.get("resolution_status"))
    if status == "generic_nonparty_option":
        return "generic_nonparty_option"
    if status == "insufficient_evidence":
        return "insufficient_evidence"
    text_bits = " | ".join(
        clean_text_scalar(row.get(k))
        for k in ["reason_summary", "uncertainty_note", "partyname_ess", "partyname", "partyname_eng"]
        if clean_text_scalar(row.get(k))
    ).lower()
    absent_markers = [
        "absent from the allowed candidate list",
        "absent from the provided mpd candidate list",
        "absent from the candidate list",
        "absent from the allowed mpd candidate list",
        "not present among the candidate mpd entries",
        "not present among the candidate mpd options",
        "not present in the allowed mpd candidate list",
        "not present among the provided candidates",
        "none of the allowed mpd candidates",
        "no candidate mpd option",
        "no corresponding mpd option",
        "no corresponding mpd candidate",
        "not in the candidate list",
        "candidate list includes other parties only",
        "only a component party appears",
        "component party rather than the full joint electoral list",
        "regional alliance/list that is not present among the candidate",
        "correct mpd party is absent",
        "true party is absent",
        "absent from the provided candidates",
    ]
    if any(marker in text_bits for marker in absent_markers):
        return "candidate_absent_from_allowed_list"
    if status:
        return status
    return "no_clear_party_match"


def infer_identified_party_if_absent(row: pd.Series) -> str:
    explicit = clean_text_scalar(row.get("identified_party_if_absent"))
    if explicit:
        return explicit
    if infer_nonmatch_reason_type(row) != "candidate_absent_from_allowed_list":
        return ""
    for key in ["partyname_ess", "partyname", "partyname_eng", "display_name"]:
        txt = clean_text_scalar(row.get(key))
        if txt:
            return txt
    return ""


def build_trust_note(row: pd.Series) -> str:
    if not bool(row.get("resolved")):
        reason_type = infer_nonmatch_reason_type(row)
        if reason_type == "candidate_absent_from_allowed_list":
            ident = infer_identified_party_if_absent(row)
            if ident:
                return f"No positive match; identifiable party/list appears absent from the allowed MPD candidates: {ident}."
            return "No positive match; identifiable party/list appears absent from the allowed MPD candidates."
        return "No positive match; left unmatched."
    parts: list[str] = []
    sc = int(pd.to_numeric(pd.Series([row.get("source_count")]), errors="coerce").fillna(0).iloc[0])
    doms = int(pd.to_numeric(pd.Series([row.get("distinct_source_domain_count")]), errors="coerce").fillna(0).iloc[0])
    high_grade = int(pd.to_numeric(pd.Series([row.get("high_grade_source_count")]), errors="coerce").fillna(0).iloc[0])
    indep_nonwiki = int(pd.to_numeric(pd.Series([row.get("independent_non_wiki_source_count")]), errors="coerce").fillna(0).iloc[0])
    party_official = int(pd.to_numeric(pd.Series([row.get("official_party_source_count")]), errors="coerce").fillna(0).iloc[0])
    if sc > 0:
        parts.append(f"{sc} source(s)")
    if doms > 0:
        parts.append(f"{doms} domain(s)")
    if high_grade > 0:
        parts.append(f"{high_grade} official/reference source(s)")
    if indep_nonwiki > 0:
        parts.append(f"{indep_nonwiki} independent non-wiki source(s)")
    if party_official > 0:
        parts.append(f"{party_official} official party source(s)")
    soft_alert = clean_text_scalar(row.get("vote_share_gap_soft_alert"))
    if soft_alert:
        parts.append(f"share-gap alert: {soft_alert}")
    flags = clean_text_scalar(row.get("trust_flags"))
    if flags:
        parts.append(f"flags: {flags}")
    return "; ".join(parts)

def compute_review_bucket(row: pd.Series) -> str:
    resolved = bool(row.get("resolved")) if not pd.isna(row.get("resolved")) else False
    if not resolved:
        reason_type = infer_nonmatch_reason_type(row)
        if reason_type == "candidate_absent_from_allowed_list":
            return "candidate_absent_from_allowed_list"
        return "no_match"
    trust = int(pd.to_numeric(pd.Series([row.get("trust_score_0_100")]), errors="coerce").fillna(0).iloc[0])
    source_count = int(pd.to_numeric(pd.Series([row.get("source_count")]), errors="coerce").fillna(0).iloc[0])
    allowed = bool(row.get("candidate_allowed"))
    offref = int(pd.to_numeric(pd.Series([row.get("official_or_reference_source_count")]), errors="coerce").fillna(0).iloc[0])
    high_grade_domains = int(pd.to_numeric(pd.Series([row.get("high_grade_distinct_domain_count")]), errors="coerce").fillna(0).iloc[0])
    indep_nonwiki = int(pd.to_numeric(pd.Series([row.get("independent_non_wiki_source_count")]), errors="coerce").fillna(0).iloc[0])
    indep_nonwiki_domains = int(pd.to_numeric(pd.Series([row.get("independent_non_wiki_distinct_domain_count")]), errors="coerce").fillna(0).iloc[0])
    basis = clean_text_scalar(row.get("match_basis"))

    basis_allows_auto = basis in {"exact_name", "abbreviation", "translation", "transliteration"}
    has_strong_evidence = (
        (offref >= 2 and high_grade_domains >= 2)
        or (offref >= 1 and indep_nonwiki >= 2 and indep_nonwiki_domains >= 2)
    )

    if allowed and trust >= 90 and source_count >= 2 and basis_allows_auto and has_strong_evidence:
        return "strong_candidate"
    if allowed and trust >= 70:
        return "needs_manual_review"
    return "hold_back"

def build_review_table(queue: pd.DataFrame, results: pd.DataFrame, review_csv: Path) -> pd.DataFrame:
    review = queue.merge(results, on="target_id", how="left", suffixes=("", "_ai"))
    review["candidate_allowed"] = review.apply(candidate_id_allowed, axis=1)
    review["model_source_count_raw"] = review.get("model_sources_json", pd.Series(index=review.index)).apply(lambda x: len(parse_sources_json(x)))
    review["api_source_count_raw"] = review.get("api_sources_json", pd.Series(index=review.index)).apply(lambda x: len(parse_sources_json(x)))

    merged_effective_sources: list[str] = []
    for _, row in review.iterrows():
        parsed_sources = parse_sources_json(row.get("model_sources_json"))
        api_sources = parse_sources_json(row.get("api_sources_json"))
        merged_effective_sources.append(json.dumps(choose_effective_sources(parsed_sources, api_sources), ensure_ascii=False))
    review["effective_sources_json"] = merged_effective_sources

    source_cols = flatten_sources_columns(review, "effective_sources_json", max_items=4)
    review = pd.concat([review, source_cols], axis=1)

    review["model_confidence_0_100"] = pd.to_numeric(review.get("model_confidence_0_100"), errors="coerce")
    review["decision_confidence_0_100"] = review["model_confidence_0_100"]
    review["share_ess"] = pd.to_numeric(review.get("share_ess"), errors="coerce")
    review["selected_mpd_party_id"] = pd.to_numeric(review.get("selected_mpd_party_id"), errors="coerce")
    selected_meta = review.apply(lookup_selected_candidate_fields, axis=1)
    review["matched_partyname_mpd_lookup"] = selected_meta.apply(lambda d: clean_text_scalar(d.get("partyname_mpd")))
    review["matched_partyfacts_id_cand"] = pd.to_numeric(selected_meta.apply(lambda d: d.get("partyfacts_id_cand")), errors="coerce")
    review["matched_share_mpd"] = pd.to_numeric(selected_meta.apply(lambda d: d.get("share_mpd")), errors="coerce")
    if "selected_party_name_mpd" in review.columns:
        _sel_existing = review["selected_party_name_mpd"].fillna("").astype(str).str.strip()
        review["selected_party_name_mpd"] = review["selected_party_name_mpd"].where(_sel_existing.ne(""), review["matched_partyname_mpd_lookup"])
    else:
        review["selected_party_name_mpd"] = review["matched_partyname_mpd_lookup"]
    resolved_bool = review["resolved"].astype("boolean").fillna(False)
    _matched_mask = resolved_bool & review["selected_mpd_party_id"].notna() & review["share_ess"].notna() & review["matched_share_mpd"].notna()
    review["ess_minus_mpd_share_gap_pp"] = np.where(_matched_mask, review["share_ess"] - review["matched_share_mpd"], np.nan)
    review["abs_share_gap_pp"] = pd.to_numeric(review["ess_minus_mpd_share_gap_pp"], errors="coerce").abs()
    review["vote_share_gap_band"] = review["abs_share_gap_pp"].apply(classify_vote_share_gap_band)
    review["vote_share_compare_note"] = review.apply(build_vote_share_compare_note, axis=1)
    review["vote_share_gap_soft_alert"] = review["abs_share_gap_pp"].apply(classify_vote_share_soft_alert)
    review["vote_share_gap_soft_alert_note"] = review.apply(build_vote_share_soft_alert_note, axis=1)
    review["nonmatch_reason_type"] = review.apply(infer_nonmatch_reason_type, axis=1)
    review["identified_party_if_absent"] = review.apply(infer_identified_party_if_absent, axis=1)
    review["candidate_absent_flag"] = review["nonmatch_reason_type"].eq("candidate_absent_from_allowed_list")
    review["trust_cap_0_100"] = 0
    review["trust_flags"] = ""

    caps: list[int] = []
    flags_list: list[str] = []
    for _, row in review.iterrows():
        cap, flags = compute_trust_cap_and_flags(row)
        caps.append(cap)
        flags_list.append("; ".join(flags))
    review["trust_cap_0_100"] = caps
    review["trust_flags"] = flags_list
    review["trust_score_0_100"] = np.where(
        resolved_bool,
        np.minimum(review["decision_confidence_0_100"].fillna(0), review["trust_cap_0_100"].fillna(0)),
        0,
    ).astype(int)
    review["trust_note"] = review.apply(build_trust_note, axis=1)
    review["review_bucket"] = review.apply(compute_review_bucket, axis=1)
    review["recommended_action"] = np.select(
        [
            review["review_bucket"].eq("strong_candidate"),
            review["review_bucket"].eq("needs_manual_review"),
            review["review_bucket"].eq("candidate_absent_from_allowed_list"),
            review["review_bucket"].eq("no_match"),
        ],
        [
            "can_apply_if_you_agree",
            "review_manually_before_apply",
            "keep_unmatched_candidate_absent_from_allowed_list",
            "keep_unmatched",
        ],
        default="hold_back",
    )

    # Preserve manual approval columns if a prior review file exists.
    if review_csv.exists():
        try:
            old = pd.read_csv(review_csv, usecols=lambda c: c in {"target_id", "approve_apply", "reviewer_note"})
            review = review.merge(old, on="target_id", how="left", suffixes=("", "_old"))
        except Exception:
            review["approve_apply"] = ""
            review["reviewer_note"] = ""
    else:
        review["approve_apply"] = ""
        review["reviewer_note"] = ""

    if "approve_apply" not in review.columns:
        review["approve_apply"] = ""
    if "reviewer_note" not in review.columns:
        review["reviewer_note"] = ""

    ordered = [
        "target_id", "country", "iso2c", "iso3c", "election_year", "edate_mpd_str",
        "ess_party_id", "partyfacts_id", "partyname_ess", "partyname", "partyname_eng",
        "n_voters", "share_ess", "matched_share_mpd", "ess_minus_mpd_share_gap_pp", "abs_share_gap_pp", "vote_share_gap_band", "vote_share_compare_note", "vote_share_gap_soft_alert", "vote_share_gap_soft_alert_note", "generic_skip",
        "candidate_count", "top5_mpd_ids", "best_name_score",
        "status", "resolved", "resolution_status", "nonmatch_reason_type", "identified_party_if_absent", "candidate_absent_flag",
        "selected_mpd_party_id", "selected_party_name_mpd", "matched_partyfacts_id_cand", "selected_from_candidate_list",
        "candidate_rank_by_share", "match_basis",
        "model_confidence_0_100", "decision_confidence_0_100", "trust_cap_0_100", "trust_score_0_100", "review_bucket",
        "recommended_action", "candidate_allowed", "trust_flags", "trust_note", "reason_summary", "uncertainty_note",
        "source_count", "source_domains", "distinct_source_domain_count",
        "official_source_count", "official_or_reference_source_count", "high_grade_source_count", "high_grade_distinct_domain_count", "independent_non_wiki_source_count", "independent_non_wiki_distinct_domain_count", "official_party_source_count", "non_wiki_source_count", "source_quality_summary",
        "model_source_count_raw", "api_source_count_raw",
        "source_1_title", "source_1_url", "source_1_publisher", "source_1_why", "source_1_quality_tag", "source_1_quality_score", "source_1_provided_quality_tag",
        "source_2_title", "source_2_url", "source_2_publisher", "source_2_why", "source_2_quality_tag", "source_2_quality_score", "source_2_provided_quality_tag",
        "source_3_title", "source_3_url", "source_3_publisher", "source_3_why", "source_3_quality_tag", "source_3_quality_score", "source_3_provided_quality_tag",
        "source_4_title", "source_4_url", "source_4_publisher", "source_4_why", "source_4_quality_tag", "source_4_quality_score", "source_4_provided_quality_tag",
        "tool_calls_web_search", "used_web_tool_type", "usage_input_tokens", "usage_output_tokens", "usage_total_tokens",
        "approve_apply", "reviewer_note", "effective_sources_json", "api_sources_json", "model_sources_json", "cand_json", "cand_text",
    ]
    ordered = [c for c in ordered if c in review.columns] + [c for c in review.columns if c not in ordered]
    return review[ordered].copy()


# =============================================================================
# Apply results back into respondent-level file
# =============================================================================
def build_pervars_table(mpd_path: Path) -> pd.DataFrame:
    mpd_cols = get_stata_columns(mpd_path)
    per_cols = [c for c in mpd_cols if c.startswith("per")]
    rile_block = cols_between(mpd_cols, "rile", "intpeace")
    keep_cols = ["countryname", "edate", "party", *per_cols, *rile_block, "parfam"]
    keep_cols = [c for i, c in enumerate(keep_cols) if c in mpd_cols and c not in keep_cols[:i]]
    mpd = pd.read_stata(mpd_path, columns=keep_cols, convert_categoricals=False)
    mpd["iso2c"] = mpd["countryname"].map(mpd_country_to_iso2)
    mpd = mpd.rename(columns={"edate": "edate_mpd", "party": "mpd_party_id"})
    cols = ["iso2c", "edate_mpd", "mpd_party_id", *[c for c in keep_cols if c not in {"countryname", "edate", "party"}]]
    return mpd[cols].copy()


def apply_results(validated_path: Path, mpd_path: Path, candidate_table: pd.DataFrame) -> None:
    queue = pd.read_csv(QUEUE_CSV)
    results = pd.read_csv(RESULTS_CSV)
    review = build_review_table(queue, results, REVIEW_CSV)
    review.to_csv(REVIEW_CSV, index=False)
    print(f"Saved review file: {REVIEW_CSV}")

    if REQUIRE_MANUAL_APPROVAL_TO_APPLY:
        approve = pd.to_numeric(review["approve_apply"], errors="coerce").fillna(0).astype(int)
        accepted = review[
            (approve.eq(1))
            & review["resolved"].fillna(False)
            & review["candidate_allowed"].fillna(False)
            & review["selected_mpd_party_id"].notna()
        ].copy()
        if accepted.empty:
            raise SystemExit(
                "No rows are approved yet. Edit the review CSV and set approve_apply=1 on the rows you want to use, then rerun with RUN_APPLY=True."
            )
    else:
        if not AUTO_APPLY_STRONG_MATCHES:
            raise SystemExit(
                "REQUIRE_MANUAL_APPROVAL_TO_APPLY=False but AUTO_APPLY_STRONG_MATCHES=False. "
                "Turn one of them on before RUN_APPLY=True."
            )
        accepted = review[
            review["resolved"].fillna(False)
            & review["candidate_allowed"].fillna(False)
            & review["selected_mpd_party_id"].notna()
            & review["trust_score_0_100"].ge(MIN_TRUST_SCORE_TO_APPLY)
            & review["review_bucket"].eq("strong_candidate")
        ].copy()
        if accepted.empty:
            raise SystemExit(
                "No AI suggestions met the auto-apply criteria. Review the CSV and either lower MIN_TRUST_SCORE_TO_APPLY or use manual approval."
            )

    accepted["selected_mpd_party_id"] = to_numeric_or_na(accepted["selected_mpd_party_id"])
    accepted["edate_mpd"] = pd.to_datetime(accepted["edate_mpd_str"], errors="coerce")
    cand_lookup = candidate_table.rename(columns={
        "mpd_party_id": "selected_mpd_party_id",
        "partyfacts_id_cand": "ai_partyfacts_id_cand",
        "partyname_mpd": "ai_partyname_mpd",
        "share_mpd": "ai_share_mpd",
    })
    accepted = accepted.merge(cand_lookup, on=["iso2c", "edate_mpd", "selected_mpd_party_id"], how="left")
    if "ai_partyname_mpd" in accepted.columns and "selected_party_name_mpd" in accepted.columns:
        _ai_name = accepted["ai_partyname_mpd"].fillna("").astype(str).str.strip()
        accepted["ai_partyname_mpd"] = accepted["ai_partyname_mpd"].where(_ai_name.ne(""), accepted["selected_party_name_mpd"])
    if "matched_partyfacts_id_cand" in accepted.columns and "ai_partyfacts_id_cand" in accepted.columns:
        accepted["matched_partyfacts_id_cand"] = accepted["matched_partyfacts_id_cand"].where(accepted["matched_partyfacts_id_cand"].notna(), accepted["ai_partyfacts_id_cand"])
    elif "ai_partyfacts_id_cand" in accepted.columns and "matched_partyfacts_id_cand" not in accepted.columns:
        accepted["matched_partyfacts_id_cand"] = accepted["ai_partyfacts_id_cand"]
    if "ai_share_mpd" in accepted.columns and "matched_share_mpd" in accepted.columns:
        accepted["ai_share_mpd"] = accepted["ai_share_mpd"].where(accepted["ai_share_mpd"].notna(), accepted["matched_share_mpd"])
    elif "matched_share_mpd" in accepted.columns and "ai_share_mpd" not in accepted.columns:
        accepted["ai_share_mpd"] = accepted["matched_share_mpd"]

    keep_cols = [
        "iso3c", "edate_mpd", "ess_party_id", "selected_mpd_party_id", "ai_partyname_mpd", "ai_share_mpd", "matched_partyfacts_id_cand", "matched_share_mpd", "ess_minus_mpd_share_gap_pp", "abs_share_gap_pp", "vote_share_gap_band", "vote_share_compare_note", "vote_share_gap_soft_alert", "vote_share_gap_soft_alert_note",
        "trust_score_0_100", "model_confidence_0_100", "review_bucket", "resolution_status", "match_basis",
        "reason_summary", "uncertainty_note", "effective_sources_json", "source_count", "source_domains", "distinct_source_domain_count",
    ]
    keep_cols = [c for c in keep_cols if c in accepted.columns]
    accepted = accepted[keep_cols].drop_duplicates(["iso3c", "edate_mpd", "ess_party_id"])
    accepted.to_csv(APPROVED_CSV, index=False)
    print(f"Saved approved matches: {APPROVED_CSV}")

    df = pd.read_stata(validated_path, convert_categoricals=False)
    df = df.merge(accepted, on=["iso3c", "edate_mpd", "ess_party_id"], how="left")

    ai_used_mask = df["mpd_party_id"].isna() & df["selected_mpd_party_id"].notna()
    df["ai_match_used"] = ai_used_mask.astype("int8")
    if "match_source" not in df.columns:
        df["match_source"] = np.where(df["mpd_party_id"].notna(), "pablo_validated", "")
    df.loc[ai_used_mask, "match_source"] = "ai_approved"

    df["mpd_party_id"] = df["mpd_party_id"].where(df["mpd_party_id"].notna(), df["selected_mpd_party_id"])
    if "partyname_mpd" in df.columns:
        df["partyname_mpd"] = df["partyname_mpd"].where(df["partyname_mpd"].notna(), df["ai_partyname_mpd"])
    else:
        df["partyname_mpd"] = df["ai_partyname_mpd"]
    if "share_mpd" in df.columns:
        df["share_mpd"] = df["share_mpd"].where(df["share_mpd"].notna(), df["ai_share_mpd"])
    else:
        df["share_mpd"] = df["ai_share_mpd"]

    df["ai_match_trust_score"] = df["trust_score_0_100"]
    df["ai_match_model_conf"] = df["model_confidence_0_100"]
    df["ai_match_bucket"] = df["review_bucket"]
    df["ai_match_status"] = df["resolution_status"]
    df["ai_match_basis"] = df["match_basis"]
    df["ai_match_reason"] = df["reason_summary"]
    df["ai_match_uncertainty"] = df["uncertainty_note"]
    df["ai_match_partyfacts_id_cand"] = df["matched_partyfacts_id_cand"]
    df["ai_match_share_mpd_compare"] = df["matched_share_mpd"]
    df["ai_match_ess_minus_mpd_gap_pp"] = df["ess_minus_mpd_share_gap_pp"]
    df["ai_match_abs_share_gap_pp"] = df["abs_share_gap_pp"]
    df["ai_match_vote_share_gap_band"] = df["vote_share_gap_band"]
    df["ai_match_vote_share_compare_note"] = df["vote_share_compare_note"]
    df["ai_match_vote_share_gap_soft_alert"] = df["vote_share_gap_soft_alert"]
    df["ai_match_vote_share_gap_soft_alert_note"] = df["vote_share_gap_soft_alert_note"]
    df["ai_match_sources_json"] = df["effective_sources_json"]
    df["ai_match_source_count"] = df["source_count"]
    df["ai_match_source_domains"] = df["source_domains"]
    df["ai_match_source_domain_count"] = df["distinct_source_domain_count"]

    pervars = build_pervars_table(mpd_path)
    ai_per = pervars.rename(columns={c: f"{c}_ai" for c in pervars.columns if c not in {"iso2c", "edate_mpd", "mpd_party_id"}})
    ai_per = ai_per.rename(columns={"mpd_party_id": "selected_mpd_party_id"})
    df = df.merge(ai_per, on=["iso2c", "edate_mpd", "selected_mpd_party_id"], how="left")

    for col in pervars.columns:
        if col in {"iso2c", "edate_mpd", "mpd_party_id"}:
            continue
        ai_col = f"{col}_ai"
        if ai_col not in df.columns:
            continue
        if col not in df.columns:
            df[col] = df[ai_col]
        else:
            df[col] = df[col].where(df[col].notna(), df[ai_col])
        df = df.drop(columns=[ai_col])

    drop_aux = [
        "selected_mpd_party_id", "ai_partyname_mpd", "ai_share_mpd", "ai_partyfacts_id_cand", "matched_partyfacts_id_cand", "matched_share_mpd", "ess_minus_mpd_share_gap_pp", "abs_share_gap_pp", "vote_share_gap_band", "vote_share_compare_note", "vote_share_gap_soft_alert", "vote_share_gap_soft_alert_note",
        "trust_score_0_100", "model_confidence_0_100", "review_bucket",
        "resolution_status", "match_basis", "reason_summary", "uncertainty_note",
        "effective_sources_json", "source_count", "source_domains", "distinct_source_domain_count",
    ]
    drop_aux = [c for c in drop_aux if c in df.columns]
    if drop_aux:
        df = df.drop(columns=drop_aux)

    varlabs = get_stata_variable_labels(validated_path)
    varlabs.update({
        "match_source": "Source of final ESS-MPD match",
        "ai_match_used": "AI added a new ESS-MPD match",
        "ai_match_trust_score": "Conservative AI trust score 0-100",
        "ai_match_model_conf": "Model confidence 0-100",
        "ai_match_bucket": "AI review bucket",
        "ai_match_status": "AI match status",
        "ai_match_basis": "AI match basis",
        "ai_match_reason": "AI concise justification",
        "ai_match_uncertainty": "AI uncertainty note",
        "ai_match_partyfacts_id_cand": "PartyFacts id for AI-selected MPD candidate",
        "ai_match_share_mpd_compare": "MPD vote share (%) for AI-selected match",
        "ai_match_ess_minus_mpd_gap_pp": "ESS minus MPD vote share gap (percentage points) for AI-selected match",
        "ai_match_abs_share_gap_pp": "Absolute ESS-MPD vote share gap (percentage points) for AI-selected match",
        "ai_match_vote_share_gap_band": "ESS-MPD vote share gap band for AI-selected match",
        "ai_match_vote_share_compare_note": "ESS vs MPD vote share diagnostic note for AI-selected match",
        "ai_match_vote_share_gap_soft_alert": "Soft caution flag for large ESS-MPD vote share gap on AI-selected match",
        "ai_match_vote_share_gap_soft_alert_note": "Soft caution note for large ESS-MPD vote share gap on AI-selected match",
        "ai_match_sources_json": "AI source list JSON",
        "ai_match_source_count": "AI source count",
        "ai_match_source_domains": "AI source domains",
        "ai_match_source_domain_count": "AI distinct source-domain count",
    })

    print(f"Writing final CSV: {FINAL_CSV}")
    df.to_csv(FINAL_CSV, index=False)
    print(f"Writing final DTA: {FINAL_DTA}")
    stata_rename_map = save_stata(
        df,
        FINAL_DTA,
        variable_labels=varlabs,
        data_label="Validated ESS-MPD match with rigorous AI extensions",
        daily_date_cols=[c for c in ["edate_ess", "edate_mpd", "first_date_int", "last_date_int"] if c in df.columns],
    )
    map_path = write_stata_name_map(stata_rename_map, FINAL_DTA)
    print(f"Saved {FINAL_CSV}")
    print(f"Saved {FINAL_DTA}")
    if map_path is not None:
        print(f"Saved {map_path}")


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    print("ESS-MPD rigorous AI extension")
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"SOURCE_VALIDATED_PATH: {SOURCE_VALIDATED_PATH}")
    mpd_path = resolve_mpd_input(BASE_DIR)
    pf_external_path = resolve_partyfacts_external_input(BASE_DIR)
    print(f"MPD_PATH: {mpd_path}")
    print(f"PARTYFACTS_EXTERNAL_PATH: {pf_external_path}")

    if PREFLIGHT_ONLY:
        print("PREFLIGHT_ONLY=True, so no files were written.")
        return

    candidate_table = load_manifesto_candidate_table(mpd_path, pf_external_path)

    if RUN_PREP:
        queue = build_unmatched_queue(SOURCE_VALIDATED_PATH, candidate_table)
        queue.to_csv(QUEUE_CSV, index=False)
        qlabs = {
            "target_id": "Unique unmatched ESS party-election key",
            "n_voters": "Number of voter respondents in this key",
            "share_ess": "ESS vote share (%)",
            "candidate_count": "Number of MPD candidates in same election",
            "best_name_score": "Best simple name similarity score",
            "generic_skip": "Heuristic generic non-party option flag",
            "missing_display_name": "Display party label is empty after ESS name fallback",
            "low_value_display_skip": "Exact low-value ballot-option label flag",
            "collect_skip_reason": "Why this queue row is excluded from AI collection",
            "collect_flag": "Should be sent to AI collection",
        }
        save_stata(
            queue,
            QUEUE_DTA,
            variable_labels=qlabs,
            data_label="Rigorous AI queue for unmatched ESS-MPD cases",
            daily_date_cols=["edate_mpd"],
        )
        write_batch_jsonl(queue, REQUESTS_JSONL)
        print(f"Saved {QUEUE_CSV}")
        print(f"Saved {QUEUE_DTA}")
        print(f"Saved {REQUESTS_JSONL}")
        print(f"Queue rows: {len(queue):,}")
        print(f"Excluded missing display_name rows: {int((queue['collect_skip_reason'] == 'missing_display_name').sum()):,}")
        print(f"Excluded low-value ballot-option rows: {int((queue['collect_skip_reason'] == 'low_value_ballot_option').sum()):,}")
        print(f"Excluded generic/non-party rows: {int((queue['collect_skip_reason'] == 'generic_nonparty_option').sum()):,}")
        print(f"Collectable rows: {int(queue['collect_flag'].sum()):,}")

    if RUN_COLLECT:
        queue = pd.read_csv(QUEUE_CSV)
        collect_with_openai(queue, RESULTS_JSONL, RESULTS_CSV)
        print(f"Saved/updated {RESULTS_JSONL}")
        print(f"Saved/updated {RESULTS_CSV}")

    if RUN_BUILD_REVIEW:
        if not QUEUE_CSV.exists():
            raise SystemExit(f"Missing queue file: {QUEUE_CSV}")
        if not RESULTS_CSV.exists():
            raise SystemExit(f"Missing results file: {RESULTS_CSV}")
        queue = pd.read_csv(QUEUE_CSV)
        results = pd.read_csv(RESULTS_CSV)
        review = build_review_table(queue, results, REVIEW_CSV)
        review.to_csv(REVIEW_CSV, index=False)
        print(f"Saved {REVIEW_CSV}")

    if RUN_APPLY:
        if not QUEUE_CSV.exists():
            raise SystemExit(f"Missing queue file: {QUEUE_CSV}")
        if not RESULTS_CSV.exists():
            raise SystemExit(f"Missing results file: {RESULTS_CSV}")
        apply_results(SOURCE_VALIDATED_PATH, mpd_path, candidate_table)

    print("Done.")


if __name__ == "__main__":
    main()
