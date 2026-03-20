#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Spyder-ready PyBLP runner for the new backbone data, using:
- common_support_complete_case bundle construction
- full-MPD family leave-out instrument by default (old way)
- the same demand system as before
- a new party-election-level supply stage that estimates multi-anchor cost models

Supply microfoundation (per dimension g):

    V_jt D_jtg = kI_g (x_jtg - mu_jg)
               + kL_g (x_jtg - x_{j,t-1,g})
               + kF_g (x_jtg - a_{f,-j,t,g})

where:
- x_jtg        = party position in election t, dimension g
- D_jtg        = aggregate own derivative from demand
- mu_jg        = party fixed ideological ideal point (captured by party FE)
- x_{j,t-1,g}  = lagged party position
- a_{f,-j,t,g} = leave-one-country-out family anchor

Estimated supply specifications:
1) ideal_only          : x = FE + beta*D
2) lag_only            : x - lag = beta*D
3) family_only         : x - fam = beta*D
4) ideal_lag           : x = FE + rho_L*lag + beta*D
5) ideal_family        : x = FE + rho_F*fam + beta*D
6) lag_family          : x - fam = rho_L*(lag - fam) + beta*D
7) ideal_lag_family    : x = FE + rho_L*lag + rho_F*fam + beta*D

For specs with FE, the implied ideal-point weight is recovered as:
    rho_I = 1 - rho_L - rho_F
with missing terms set to zero.

For each spec, both OLS and IV are run. The endogenous regressor is D.
Excluded instruments are aggregated market demographic means built from the demo file.

Main outputs
------------
- demand_common_support_complete_case_multicost.csv
- supply_multicost_specs.csv
- supply_multicost_sample_summary.csv
- party_election_supply_panel.csv
- product_data_used_in_blp_multicost.csv
- bundle_definition_common_support_complete_case.csv
- preflight_diagnostics_from_backbone.csv

This v4 version z-scores BOTH the economic and cultural indices (and the corresponding
family leave-out instruments) before demand estimation.
"""

from __future__ import annotations

from pathlib import Path
import math
import warnings

import numpy as np
import pandas as pd

cwd = Path.cwd()
if (cwd / "pyblp.py").exists():
    raise RuntimeError(
        f"Found local file {cwd/'pyblp.py'} which shadows the pyblp package. "
        "Rename it, delete __pycache__, and restart."
    )

import pyblp
import statsmodels.api as sm
from numpy.linalg import pinv

# =============================================================================
# PATHS / SPYDER SETTINGS
# =============================================================================
BASE_DIR = Path("/Users/zl279/Downloads/Poli-IO/raw_data")
DATA0_DIR = BASE_DIR / "0_data"
BACKBONE_DIR = BASE_DIR / "pyblp_backbone"
OUT_DIR = BACKBONE_DIR / "pyblp_multicost_supply_out_v4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AUTO_CANDIDATES = [
    BACKBONE_DIR / "auto_regression_pyblp_ready_with_char_patched.dta",
    BACKBONE_DIR / "auto_regression_pyblp_ready_with_char.dta",
    BACKBONE_DIR / "auto_regression_pyblp_backbone.dta",
]
DEMO_CANDIDATES = [BACKBONE_DIR / "demo_file_pyblp_backbone.dta"]
MPD_CANDIDATES = [
    DATA0_DIR / "MPD" / "MPDataset_MPDS2025a_stata14.dta",
    DATA0_DIR / "MPD" / "MPDataset_MPDS2024a_stata14.dta",
    Path("/Users/zl279/Downloads/OneDrive_1_7-31-2025/MPDataset_MPDS2024a_stata14.dta"),
]
SPREADSHEET_CANDIDATES = [
    BASE_DIR / "MPD_items_codebook_v1a.xlsx",
    DATA0_DIR / "MPD_items_codebook_v1a.xlsx",
    Path("/Users/zl279/Downloads/MPD_items_codebook_v1a.xlsx"),
]

AUTO_REG_PATH: Path | None = None
DEMO_PATH: Path | None = None
MPD_PATH: Path | None = None
SPREADSHEET_PATH: Path | None = None

# =============================================================================
# SETTINGS
# =============================================================================
RUN_NAME = "CULT_NO_ANTI_IMPERIALISM_2D_COMMON_SUPPORT_COMPLETE_CASE_MULTICOST_ZBOTH"

ECON_RIGHT_MULT = -1.0
CULT_RIGHT_MULT = +1.0
DROP_FROM_CULT = ["per103"]
Z_SCORE_ECONOMIC = True
Z_SCORE_CULTURE = True

# Sparse CEE/transitional-democracy sub-codes dropped in the common-support bundle
SPARSE_ECON_CODES = {"per4012", "per4123", "per4124", "per5041", "per5061"}
SPARSE_CULT_CODES = {"per2022", "per2023", "per7062"}

INTEGRATION_SIZE = 200
OPT_MAXITER = 15000
PRINT_PYBLP_RESULTS = True
SAVE_PRODUCT_DATA = True
SAVE_PREFLIGHT = True

# Leave-out scope for the demand instruments.
# "full_mpd" reproduces the old/main way.
# You can switch to "ess_sample_country_year" later if desired.
LEAVEOUT_SCOPE = "full_mpd"

SUPPLY_THRESHOLDS = [
    ("no restriction", None),
    ("ever ≥10%", 0.10),
]
RICH_MOMENTS = ["college", "age", "lrscale", "female", "young", "middle", "old", "boomer"]

SUPPLY_SPECS = [
    "ideal_only",
    "lag_only",
    "family_only",
    "ideal_lag",
    "ideal_family",
    "lag_family",
    "ideal_lag_family",
]

# =============================================================================
# HELPERS
# =============================================================================
def first_existing(candidates: list[Path]) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "None of the candidate paths exist:\n  - " + "\n  - ".join(str(p) for p in candidates)
    )


def resolve_paths() -> tuple[Path, Path, Path, Path]:
    auto = AUTO_REG_PATH if AUTO_REG_PATH is not None else first_existing(AUTO_CANDIDATES)
    demo = DEMO_PATH if DEMO_PATH is not None else first_existing(DEMO_CANDIDATES)
    mpd = MPD_PATH if MPD_PATH is not None else first_existing(MPD_CANDIDATES)
    sheet = SPREADSHEET_PATH if SPREADSHEET_PATH is not None else first_existing(SPREADSHEET_CANDIDATES)
    return auto, demo, mpd, sheet


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def ensure_cols(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA


def normalize_election(region4: pd.Series, election_year: pd.Series) -> pd.Series:
    reg4 = region4.astype(str).str[:4]
    ey = pd.to_numeric(election_year, errors="coerce").round().astype("Int64").astype(str)
    return (reg4 + ey).str.lower().str.replace(" ", "", regex=False)


def weighted_mean(group: pd.DataFrame, value_cols: list[str], wcol: str) -> pd.Series:
    w = pd.to_numeric(group[wcol], errors="coerce").to_numpy(dtype=float)
    out = {}
    for c in value_cols:
        x = pd.to_numeric(group[c], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(w) & np.isfinite(x)
        if m.sum() == 0:
            out[c] = np.nan
        else:
            denom = float(np.sum(w[m]))
            if (not np.isfinite(denom)) or abs(denom) <= 1e-12:
                out[c] = np.nan
            else:
                out[c] = float(np.sum(w[m] * x[m]) / denom)
    return pd.Series(out)


def weighted_avg_from_arrays(x: np.ndarray, w: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(w)
    if m.sum() == 0:
        return np.nan
    denom = float(np.sum(w[m]))
    if (not np.isfinite(denom)) or abs(denom) <= 1e-12:
        return np.nan
    return float(np.sum(x[m] * w[m]) / denom)


def safe_weighted_average(x: pd.Series | np.ndarray, w: pd.Series | np.ndarray) -> float:
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(pd.Series(w), errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(x) & np.isfinite(w)
    if m.sum() == 0:
        return np.nan
    x = x[m]
    w = w[m]
    denom = float(np.sum(w))
    if (not np.isfinite(denom)) or abs(denom) <= 1e-12:
        return np.nan
    return float(np.average(x, weights=w))


def zscore_params(x: pd.Series) -> tuple[float, float]:
    x = pd.to_numeric(x, errors="coerce")
    mu = float(x.mean(skipna=True))
    sd = float(x.std(skipna=True, ddof=0))
    if (not np.isfinite(sd)) or sd <= 1e-12:
        sd = np.nan
    return mu, sd


def apply_z(x: pd.Series, mu: float, sd: float) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    if (not np.isfinite(mu)) or (not np.isfinite(sd)) or sd <= 1e-12:
        return pd.Series(np.nan, index=x.index)
    return (x - mu) / sd


def normal_pval_from_t(t: float) -> float:
    if not np.isfinite(t):
        return np.nan
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2.0))))


def stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def safe_int(x):
    try:
        if x is None:
            return pd.NA
        if isinstance(x, (float, np.floating)) and np.isnan(x):
            return pd.NA
        return int(x)
    except Exception:
        return pd.NA


def prune_collinear_columns(X: np.ndarray, colnames: list[str], atol: float = 1e-10) -> list[str]:
    keep = []
    current = np.zeros((X.shape[0], 0))
    current_rank = 0
    for j, name in enumerate(colnames):
        v = X[:, [j]]
        trial = np.hstack([current, v])
        r = np.linalg.matrix_rank(trial, tol=atol)
        if r > current_rank:
            keep.append(name)
            current = trial
            current_rank = r
    return keep


def build_equal_weight_index(df: pd.DataFrame, signed_list: list[tuple[str, int]]) -> pd.Series:
    parts = []
    for pc, sgn in signed_list:
        if pc in df.columns:
            parts.append(pd.to_numeric(df[pc], errors="coerce") * float(sgn))
    if not parts:
        return pd.Series(np.nan, index=df.index)
    M = pd.concat(parts, axis=1)
    return M.mean(axis=1, skipna=True)


def pick_sign_columns(sheet: pd.DataFrame):
    code_col = None
    for c in sheet.columns:
        if sheet[c].astype(str).str.contains(r"\bper\d+\b", regex=True, na=False).any():
            code_col = c
            break
    if code_col is None:
        raise ValueError("Cannot find MPD code column (needs per###) in spreadsheet.")
    sheet = sheet.copy()
    sheet["_percode"] = sheet[code_col].astype(str).str.extract(r"(per\d+)", expand=False)

    sign_candidates = []
    for c in sheet.columns:
        if c == code_col or c == "_percode":
            continue
        vals = pd.to_numeric(sheet[c], errors="coerce")
        u = set(vals.dropna().unique().tolist())
        if u and u.issubset({-1, 1}):
            sign_candidates.append(c)

    econ_col = None
    cult_col = None
    for c in sign_candidates:
        lc = str(c).lower()
        if econ_col is None and ("economic" in lc or "econ" in lc):
            econ_col = c
        if cult_col is None and ("cultural" in lc or "culture" in lc):
            cult_col = c

    if econ_col is None or cult_col is None:
        raise ValueError(
            f"Could not identify econ/cultural sign columns. Candidates: {sign_candidates}"
        )
    return sheet, code_col, econ_col, cult_col


def get_signed_percode_list(sheet: pd.DataFrame, sign_col: str) -> list[tuple[str, int]]:
    out = []
    tmp = sheet[["_percode", sign_col]].copy().dropna(subset=["_percode"]) 
    tmp[sign_col] = pd.to_numeric(tmp[sign_col], errors="coerce")
    tmp = tmp.dropna(subset=[sign_col])
    for _, r in tmp.iterrows():
        out.append((str(r["_percode"]), int(r[sign_col])))
    seen = set()
    uniq = []
    for pc, s in out:
        if pc not in seen:
            seen.add(pc)
            uniq.append((pc, s))
    return uniq


def compute_leaveout_family(mpd_df: pd.DataFrame, idx_col: str, out_prefix: str) -> pd.DataFrame:
    tmp = mpd_df.dropna(subset=["country", "parfam", "election_year", "pervote", idx_col]).copy()

    g = (
        tmp.groupby(["country", "parfam", "election_year"], as_index=False)
           .apply(lambda h: pd.Series({
               "fam_country_mean": safe_weighted_average(h[idx_col], h["pervote"]),
               "pervote_sum": float(pd.to_numeric(h["pervote"], errors="coerce").sum(skipna=True)),
               "n_rows_group": int(len(h)),
           }), include_groups=False)
           .reset_index(drop=True)
    )

    zero_or_bad = g["fam_country_mean"].isna()
    n_bad = int(zero_or_bad.sum())
    if n_bad > 0:
        warnings.warn(
            f"{out_prefix}: dropped {n_bad} country-family-election groups with missing/zero-sum pervote when building leave-out.",
            RuntimeWarning,
        )
        g = g.loc[~zero_or_bad].copy()

    grp = g.groupby(["parfam", "election_year"])["fam_country_mean"]
    g["sum_all"] = grp.transform("sum")
    g["n_all"] = grp.transform("count")
    g[out_prefix] = np.where(
        g["n_all"] > 1,
        (g["sum_all"] - g["fam_country_mean"]) / (g["n_all"] - 1),
        np.nan,
    )
    g[f"{out_prefix}2"] = (g[out_prefix] ** 2) / 100.0
    return g[["country", "parfam", "election_year", out_prefix, f"{out_prefix}2"]].drop_duplicates()


def make_demand_instruments(prod_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = prod_df.copy()
    out["const"] = 1.0
    baseZ = [
        "lo_fam_econ", "lo_fam_cult", "lo_fam_econ2", "lo_fam_cult2",
        "candidate_age", "candidate_gender",
    ]
    for z in baseZ:
        out[f"ageX_{z}"] = out["candidate_age"] * out[z]
        out[f"genderX_{z}"] = out["candidate_gender"] * out[z]

    Z_cols = ["const"] + baseZ + [f"ageX_{z}" for z in baseZ] + [f"genderX_{z}" for z in baseZ]
    Zmat = np.column_stack([pd.to_numeric(out[c], errors="coerce").to_numpy(dtype=float) for c in Z_cols])
    ok = np.isfinite(Zmat).all(axis=1)
    Z_keep = prune_collinear_columns(Zmat[ok, :], Z_cols, atol=1e-10)

    for j, c in enumerate(Z_keep):
        out[f"demand_instruments{j}"] = out[c]
    return out, [f"demand_instruments{j}" for j in range(len(Z_keep))]


def own_derivative_vector(res, name: str) -> np.ndarray:
    J = res.compute_demand_jacobians(name=name)
    d = res.extract_diagonals(J)
    return np.asarray(d, float).ravel()


def add_market_rich_Z(prod_df: pd.DataFrame, demo_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    use = [c for c in RICH_MOMENTS if c in demo_df.columns]
    W = demo_df.groupby("market_ids", as_index=False)[use].mean()
    W = W.rename(columns={c: f"{c}_bar" for c in use})
    out = prod_df.merge(W, on="market_ids", how="left", validate="m:1")
    cols = [f"{c}_bar" for c in use]
    return out, cols


def within_demean(df: pd.DataFrame, cols: list[str], fe_col: str) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = out[c] - out.groupby(fe_col)[c].transform("mean")
    return out


def build_preflight(auto: pd.DataFrame, demo: pd.DataFrame) -> pd.DataFrame:
    out = []
    auto_req = [
        "country", "election_year", "region", "vote", "anweight", "female", "educ_cat", "age", "lrscale",
        "mpd_party_id", "parfam", "partyname", "partyname_eng", "share_mpd", "candidate_age", "candidate_gender",
    ]
    demo_req = ["election", "college", "age", "lrscale", "female", "young", "middle", "old", "boomer"]

    for c in auto_req:
        out.append({"check": f"auto_has_{c}", "value": int(c in auto.columns)})
    for c in demo_req:
        out.append({"check": f"demo_has_{c}", "value": int(c in demo.columns)})

    auto_tmp = auto.copy()
    auto_tmp["region4"] = auto_tmp["region"].astype(str).str[:4]
    auto_tmp["market_check"] = normalize_election(auto_tmp["region4"], auto_tmp["election_year"])
    demo_tmp = demo.copy()
    demo_tmp["market_check"] = demo_tmp["election"].astype(str).str.lower().str.replace(" ", "", regex=False)

    auto_markets = set(auto_tmp["market_check"].dropna().unique())
    demo_markets = set(demo_tmp["market_check"].dropna().unique())
    out.append({"check": "n_auto_markets", "value": len(auto_markets)})
    out.append({"check": "n_demo_markets", "value": len(demo_markets)})
    out.append({"check": "auto_markets_missing_from_demo", "value": len(auto_markets - demo_markets)})
    out.append({"check": "demo_markets_not_used_by_auto", "value": len(demo_markets - auto_markets)})

    vote12 = auto_tmp["vote"].isin([1, 2])
    matched_party = auto_tmp["mpd_party_id"].notna()
    out.append({
        "check": "missing_candidate_age_vote12_matchedparty",
        "value": int((vote12 & matched_party & auto_tmp["candidate_age"].isna()).sum()),
    })
    out.append({
        "check": "missing_candidate_gender_vote12_matchedparty",
        "value": int((vote12 & matched_party & auto_tmp["candidate_gender"].isna()).sum()),
    })
    return pd.DataFrame(out)


def complete_mask(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    if not cols:
        return pd.Series(True, index=df.index)
    miss = [c for c in cols if c not in df.columns]
    if miss:
        return pd.Series(False, index=df.index)
    return df[cols].notna().all(axis=1)


def aggregate_auto_to_products(auto: pd.DataFrame, all_percodes: list[str]) -> pd.DataFrame:
    group_keys = ["election_year", "country", "region4", "parfam", "mpd_party_id", "election"]
    value_cols = ["vote_share", "candidate_age", "candidate_gender"] + all_percodes
    try:
        prod_base = (
            auto.groupby(group_keys, as_index=False)
                .apply(lambda g: weighted_mean(g, value_cols, "anweight"), include_groups=False)
                .reset_index(drop=True)
        )
    except TypeError:
        prod_base = (
            auto.groupby(group_keys, as_index=False)
                .apply(lambda g: weighted_mean(g, value_cols, "anweight"))
                .reset_index(drop=True)
        )

    elections = sorted(prod_base["election"].dropna().unique().tolist())
    election_to_mid = {e: i + 1 for i, e in enumerate(elections)}
    prod_base["market_ids"] = prod_base["election"].map(election_to_mid).astype(int)
    prod_base = prod_base.rename(columns={"vote_share": "shares"})
    prod_base["prices"] = 0.0
    prod_base["product_ids"] = (
        prod_base["market_ids"].astype(str) + "_" + prod_base["mpd_party_id"].astype("Int64").astype(str)
    )

    sum_shares = prod_base.groupby("market_ids")["shares"].transform("sum")
    bad = sum_shares >= 1 - 1e-12
    if bad.any():
        prod_base.loc[bad, "shares"] = prod_base.loc[bad, "shares"] * (0.99 / sum_shares[bad])

    return prod_base


def party_keep_mask_products(prod_df: pd.DataFrame, thresh: float | None) -> np.ndarray:
    if thresh is None:
        return np.ones(len(prod_df), dtype=bool)
    max_share = prod_df.groupby(["country", "mpd_party_id"])["shares"].transform("max")
    return (max_share >= float(thresh)).to_numpy()


def fit_ols(y: np.ndarray, d: np.ndarray, W: np.ndarray, add_const: bool, param_names: list[str]):
    y = np.asarray(y, float).reshape(-1)
    d = np.asarray(d, float).reshape(-1)
    W = np.asarray(W, float)
    if W.ndim == 1:
        W = W.reshape(-1, 1)
    if W.size == 0:
        W = np.zeros((len(y), 0))

    mask = np.isfinite(y) & np.isfinite(d)
    if W.shape[1] > 0:
        mask &= np.isfinite(W).all(axis=1)
    y = y[mask]
    d = d[mask]
    W = W[mask, :]

    cols = []
    X_parts = []
    if add_const:
        X_parts.append(np.ones((len(y), 1)))
        cols.append("const")
    if W.shape[1] > 0:
        X_parts.append(W)
        cols.extend(param_names[:-1])
    X_parts.append(d.reshape(-1, 1))
    cols.append(param_names[-1])
    X = np.hstack(X_parts)

    res = sm.OLS(y, X).fit(cov_type="HC1")
    params = pd.Series(res.params, index=cols)
    cov = pd.DataFrame(res.cov_params(), index=cols, columns=cols)
    return {
        "params": params,
        "cov": cov,
        "n": int(len(y)),
        "method_obj": res,
    }


def fit_iv_one_endog(y: np.ndarray, d: np.ndarray, W: np.ndarray, Zx: np.ndarray, add_const: bool, param_names: list[str]):
    y = np.asarray(y, float).reshape(-1, 1)
    d = np.asarray(d, float).reshape(-1, 1)
    W = np.asarray(W, float)
    Zx = np.asarray(Zx, float)
    if W.ndim == 1:
        W = W.reshape(-1, 1)
    if Zx.ndim == 1:
        Zx = Zx.reshape(-1, 1)
    if W.size == 0:
        W = np.zeros((len(y), 0))
    if Zx.size == 0:
        Zx = np.zeros((len(y), 0))

    mask = np.isfinite(y).ravel() & np.isfinite(d).ravel()
    if W.shape[1] > 0:
        mask &= np.isfinite(W).all(axis=1)
    if Zx.shape[1] > 0:
        mask &= np.isfinite(Zx).all(axis=1)
    y = y[mask]
    d = d[mask]
    W = W[mask, :]
    Zx = Zx[mask, :]

    X_parts = []
    Q_parts = []
    cols = []
    if add_const:
        c = np.ones((len(y), 1))
        X_parts.append(c)
        Q_parts.append(c)
        cols.append("const")
    if W.shape[1] > 0:
        X_parts.append(W)
        Q_parts.append(W)
        cols.extend(param_names[:-1])
    X_parts.append(d)
    cols.append(param_names[-1])
    X = np.hstack(X_parts)
    if Zx.shape[1] > 0:
        Q_parts.append(Zx)
    Q = np.hstack(Q_parts)

    n = X.shape[0]
    if n <= X.shape[1] + 2:
        raise ValueError("Too few usable observations for IV.")

    QQ_inv = pinv(Q.T @ Q)
    XPQX = X.T @ Q @ QQ_inv @ Q.T @ X
    XPQy = X.T @ Q @ QQ_inv @ Q.T @ y
    b = pinv(XPQX) @ XPQy
    u = y - X @ b

    meat = np.zeros((Q.shape[1], Q.shape[1]))
    for i in range(n):
        qi = Q[i:i+1, :].T
        meat += float(u[i, 0] ** 2) * (qi @ qi.T)
    A = XPQX
    B = X.T @ Q @ QQ_inv @ meat @ QQ_inv @ Q.T @ X
    V = pinv(A) @ B @ pinv(A)

    params = pd.Series(b.ravel(), index=cols)
    cov = pd.DataFrame(V, index=cols, columns=cols)

    # First-stage F for excluded instruments conditional on included exogenous controls
    fs_F = np.nan
    if Zx.shape[1] > 0:
        Xr_parts = []
        Xu_parts = []
        if add_const:
            c = np.ones((len(y), 1))
            Xr_parts.append(c)
            Xu_parts.append(c)
        if W.shape[1] > 0:
            Xr_parts.append(W)
            Xu_parts.append(W)
        Xu_parts.append(Zx)
        Xr = np.hstack(Xr_parts) if Xr_parts else np.zeros((len(y), 0))
        Xu = np.hstack(Xu_parts)
        if Xr.shape[1] > 0:
            rr = sm.OLS(d.ravel(), Xr).fit()
            ur = sm.OLS(d.ravel(), Xu).fit()
            q = Zx.shape[1]
            df2 = max(int(ur.nobs - ur.df_model - 1), 1)
            ssr_r = rr.ssr
            ssr_u = ur.ssr
            fs_F = ((ssr_r - ssr_u) / q) / max(ssr_u / df2, 1e-18)
        else:
            ur = sm.OLS(d.ravel(), Xu).fit()
            q = Zx.shape[1]
            df2 = max(int(ur.nobs - ur.df_model - 1), 1)
            ssr_r = float(np.sum((d.ravel() - d.ravel().mean()) ** 2))
            ssr_u = ur.ssr
            fs_F = ((ssr_r - ssr_u) / q) / max(ssr_u / df2, 1e-18)

    return {
        "params": params,
        "cov": cov,
        "n": int(n),
        "fs_F": float(fs_F) if np.isfinite(fs_F) else np.nan,
    }


def theta_to_kappas(spec: str, params: pd.Series) -> dict[str, float]:
    beta = float(params.get("beta_d", np.nan))
    rho_l = float(params.get("rho_lag", 0.0))
    rho_f = float(params.get("rho_family", 0.0))

    out = {
        "rho_ideal_implied": np.nan,
        "rho_lag": np.nan,
        "rho_family": np.nan,
        "kappa_ideal": np.nan,
        "kappa_lag": np.nan,
        "kappa_family": np.nan,
    }

    if spec == "ideal_only":
        out["rho_ideal_implied"] = 1.0
        out["rho_lag"] = 0.0
        out["rho_family"] = 0.0
        out["kappa_ideal"] = np.nan if beta == 0 or not np.isfinite(beta) else 1.0 / beta
    elif spec == "lag_only":
        out["rho_ideal_implied"] = 0.0
        out["rho_lag"] = 1.0
        out["rho_family"] = 0.0
        out["kappa_lag"] = np.nan if beta == 0 or not np.isfinite(beta) else 1.0 / beta
    elif spec == "family_only":
        out["rho_ideal_implied"] = 0.0
        out["rho_lag"] = 0.0
        out["rho_family"] = 1.0
        out["kappa_family"] = np.nan if beta == 0 or not np.isfinite(beta) else 1.0 / beta
    elif spec == "ideal_lag":
        out["rho_lag"] = rho_l
        out["rho_family"] = 0.0
        out["rho_ideal_implied"] = 1.0 - rho_l
        if beta != 0 and np.isfinite(beta):
            out["kappa_lag"] = rho_l / beta
            out["kappa_ideal"] = (1.0 - rho_l) / beta
    elif spec == "ideal_family":
        out["rho_lag"] = 0.0
        out["rho_family"] = rho_f
        out["rho_ideal_implied"] = 1.0 - rho_f
        if beta != 0 and np.isfinite(beta):
            out["kappa_family"] = rho_f / beta
            out["kappa_ideal"] = (1.0 - rho_f) / beta
    elif spec == "lag_family":
        out["rho_lag"] = rho_l
        out["rho_family"] = 1.0 - rho_l
        out["rho_ideal_implied"] = 0.0
        if beta != 0 and np.isfinite(beta):
            out["kappa_lag"] = rho_l / beta
            out["kappa_family"] = (1.0 - rho_l) / beta
    elif spec == "ideal_lag_family":
        out["rho_lag"] = rho_l
        out["rho_family"] = rho_f
        out["rho_ideal_implied"] = 1.0 - rho_l - rho_f
        if beta != 0 and np.isfinite(beta):
            out["kappa_lag"] = rho_l / beta
            out["kappa_family"] = rho_f / beta
            out["kappa_ideal"] = (1.0 - rho_l - rho_f) / beta
    return out


def delta_method_for_kappa(spec: str, params: pd.Series, cov: pd.DataFrame) -> dict[str, float]:
    p = params.copy()
    names = list(p.index)
    theta0 = p.to_numpy(dtype=float)
    V = cov.loc[names, names].to_numpy(dtype=float)

    out = {}
    base = theta_to_kappas(spec, p)
    for target in ["kappa_ideal", "kappa_lag", "kappa_family"]:
        g0 = base[target]
        if not np.isfinite(g0):
            out[f"se_{target}"] = np.nan
            continue
        grad = np.zeros(len(theta0))
        for j in range(len(theta0)):
            h = 1e-6 * max(1.0, abs(theta0[j]))
            tp = theta0.copy(); tp[j] += h
            tm = theta0.copy(); tm[j] -= h
            pp = pd.Series(tp, index=names)
            pm = pd.Series(tm, index=names)
            gp = theta_to_kappas(spec, pp)[target]
            gm = theta_to_kappas(spec, pm)[target]
            if np.isfinite(gp) and np.isfinite(gm):
                grad[j] = (gp - gm) / (2 * h)
            else:
                grad[j] = np.nan
        if not np.isfinite(grad).all():
            out[f"se_{target}"] = np.nan
        else:
            var = float(grad @ V @ grad)
            out[f"se_{target}"] = math.sqrt(max(var, 0.0))
    return out


def build_party_election_panel(prodZ: pd.DataFrame, d_econ: np.ndarray, d_cult: np.ndarray, Zcols: list[str]) -> pd.DataFrame:
    df = prodZ.copy()
    df["deriv_econ"] = d_econ
    df["deriv_cult"] = d_cult
    if "market_weight" not in df.columns:
        df["market_weight"] = 1.0

    group_cols = ["country", "election_year", "mpd_party_id", "parfam"]

    def agg_fun(g: pd.DataFrame) -> pd.Series:
        # In some pandas versions, include_groups=False strips grouping columns from g.
        # Recover from g.name if needed.
        if all(col in g.columns for col in group_cols):
            country0 = g["country"].iloc[0]
            election_year0 = g["election_year"].iloc[0]
            mpd_party_id0 = g["mpd_party_id"].iloc[0]
            parfam0 = g["parfam"].iloc[0]
        else:
            # g.name can be a tuple in the same order as group_cols
            name = getattr(g, "name", None)
            if isinstance(name, tuple) and len(name) == len(group_cols):
                country0, election_year0, mpd_party_id0, parfam0 = name
            else:
                country0 = election_year0 = mpd_party_id0 = parfam0 = pd.NA

        w = pd.to_numeric(g["market_weight"], errors="coerce").to_numpy(dtype=float)
        party_key = pd.NA
        if pd.notna(country0) and pd.notna(mpd_party_id0):
            try:
                party_key = f"{country0}|{int(float(mpd_party_id0))}"
            except Exception:
                party_key = f"{country0}|{mpd_party_id0}"

        out = {
            "country": country0,
            "election_year": election_year0,
            "mpd_party_id": mpd_party_id0,
            "parfam": parfam0,
            "party_key": party_key,
            "x_econ": weighted_avg_from_arrays(pd.to_numeric(g["idx_economic"], errors="coerce").to_numpy(dtype=float), w),
            "x_cult": weighted_avg_from_arrays(pd.to_numeric(g["idx_cultural"], errors="coerce").to_numpy(dtype=float), w),
            "fam_econ": weighted_avg_from_arrays(pd.to_numeric(g["lo_fam_econ"], errors="coerce").to_numpy(dtype=float), w),
            "fam_cult": weighted_avg_from_arrays(pd.to_numeric(g["lo_fam_cult"], errors="coerce").to_numpy(dtype=float), w),
            "D_econ": weighted_avg_from_arrays(pd.to_numeric(g["deriv_econ"], errors="coerce").to_numpy(dtype=float), w),
            "D_cult": weighted_avg_from_arrays(pd.to_numeric(g["deriv_cult"], errors="coerce").to_numpy(dtype=float), w),
            "n_markets_party_election": int(g["market_ids"].nunique()),
            "max_share_this_election": float(pd.to_numeric(g["shares"], errors="coerce").max()),
        }
        for z in Zcols:
            out[z] = weighted_avg_from_arrays(pd.to_numeric(g[z], errors="coerce").to_numpy(dtype=float), w)
        return pd.Series(out)

    try:
        pe = (
            df.groupby(group_cols, as_index=False)
              .apply(agg_fun, include_groups=False)
              .reset_index(drop=True)
        )
    except TypeError:
        pe = (
            df.groupby(group_cols, as_index=False)
              .apply(agg_fun)
              .reset_index(drop=True)
        )

    pe = pe.sort_values(["party_key", "election_year"]).reset_index(drop=True)
    pe["lag_econ"] = pe.groupby("party_key")["x_econ"].shift(1)
    pe["lag_cult"] = pe.groupby("party_key")["x_cult"].shift(1)
    pe["party_max_share_any_market"] = pe.groupby("party_key")["max_share_this_election"].transform("max")
    return pe
def build_spec_matrices(df: pd.DataFrame, dim: str, spec: str, Zcols: list[str]):
    x = df[f"x_{dim}"].to_numpy(dtype=float)
    d = df[f"D_{dim}"].to_numpy(dtype=float)
    lag = df[f"lag_{dim}"].to_numpy(dtype=float)
    fam = df[f"fam_{dim}"].to_numpy(dtype=float)
    Zx = df[Zcols].to_numpy(dtype=float)

    if spec == "ideal_only":
        y = x.copy()
        W = np.zeros((len(df), 0))
        add_const = False
        param_names = ["beta_d"]
        fe = True
        needed = np.isfinite(x) & np.isfinite(d) & np.isfinite(Zx).all(axis=1)
    elif spec == "lag_only":
        y = x - lag
        W = np.zeros((len(df), 0))
        add_const = False
        param_names = ["beta_d"]
        fe = False
        needed = np.isfinite(x) & np.isfinite(d) & np.isfinite(lag) & np.isfinite(Zx).all(axis=1)
    elif spec == "family_only":
        y = x - fam
        W = np.zeros((len(df), 0))
        add_const = False
        param_names = ["beta_d"]
        fe = False
        needed = np.isfinite(x) & np.isfinite(d) & np.isfinite(fam) & np.isfinite(Zx).all(axis=1)
    elif spec == "ideal_lag":
        y = x.copy()
        W = lag.reshape(-1, 1)
        add_const = False
        param_names = ["rho_lag", "beta_d"]
        fe = True
        needed = np.isfinite(x) & np.isfinite(d) & np.isfinite(lag) & np.isfinite(Zx).all(axis=1)
    elif spec == "ideal_family":
        y = x.copy()
        W = fam.reshape(-1, 1)
        add_const = False
        param_names = ["rho_family", "beta_d"]
        fe = True
        needed = np.isfinite(x) & np.isfinite(d) & np.isfinite(fam) & np.isfinite(Zx).all(axis=1)
    elif spec == "lag_family":
        y = x - fam
        W = (lag - fam).reshape(-1, 1)
        add_const = False
        param_names = ["rho_lag", "beta_d"]
        fe = False
        needed = np.isfinite(x) & np.isfinite(d) & np.isfinite(lag) & np.isfinite(fam) & np.isfinite(Zx).all(axis=1)
    elif spec == "ideal_lag_family":
        y = x.copy()
        W = np.column_stack([lag, fam])
        add_const = False
        param_names = ["rho_lag", "rho_family", "beta_d"]
        fe = True
        needed = np.isfinite(x) & np.isfinite(d) & np.isfinite(lag) & np.isfinite(fam) & np.isfinite(Zx).all(axis=1)
    else:
        raise ValueError(f"Unknown spec: {spec}")

    out = df.loc[needed, ["party_key", "country", "election_year", "mpd_party_id"]].copy()
    out["y"] = y[needed]
    out["d"] = d[needed]
    out["dim"] = dim
    out["spec"] = spec

    Wsub = W[needed, :] if W.size > 0 else np.zeros((needed.sum(), 0))
    Zsub = Zx[needed, :]

    if fe:
        tmp = out.copy()
        tmp["d"] = d[needed]
        for j, name in enumerate(param_names[:-1]):
            tmp[name] = Wsub[:, j]
        for j, z in enumerate(Zcols):
            tmp[z] = Zsub[:, j]
        demean_cols = ["y", "d"] + param_names[:-1] + Zcols
        tmp = within_demean(tmp, cols=demean_cols, fe_col="party_key")
        y_use = tmp["y"].to_numpy(dtype=float)
        d_use = tmp["d"].to_numpy(dtype=float)
        W_use = tmp[param_names[:-1]].to_numpy(dtype=float) if param_names[:-1] else np.zeros((len(tmp), 0))
        Z_use = tmp[Zcols].to_numpy(dtype=float)
        out = tmp
    else:
        y_use = out["y"].to_numpy(dtype=float)
        d_use = d[needed]
        W_use = Wsub
        Z_use = Zsub

    return out, y_use, d_use, W_use, Z_use, add_const, param_names


def add_result_rows(rows: list[dict], run_name: str, sample_label: str, dim_label: str, spec: str, method: str,
                    est, n_parties: int, kZ: int):
    params = est["params"]
    cov = est["cov"]
    beta = float(params.get("beta_d", np.nan))
    se_beta = math.sqrt(max(float(cov.loc["beta_d", "beta_d"]), 0.0)) if "beta_d" in cov.index else np.nan
    t_beta = beta / se_beta if np.isfinite(se_beta) and se_beta > 0 else np.nan
    p_beta = normal_pval_from_t(t_beta)

    rho_lag = float(params.get("rho_lag", np.nan)) if "rho_lag" in params.index else np.nan
    rho_family = float(params.get("rho_family", np.nan)) if "rho_family" in params.index else np.nan
    se_rho_lag = math.sqrt(max(float(cov.loc["rho_lag", "rho_lag"]), 0.0)) if "rho_lag" in cov.index else np.nan
    se_rho_family = math.sqrt(max(float(cov.loc["rho_family", "rho_family"]), 0.0)) if "rho_family" in cov.index else np.nan

    implied = theta_to_kappas(spec, params)
    implied_se = delta_method_for_kappa(spec, params, cov)

    for target in ["kappa_ideal", "kappa_lag", "kappa_family"]:
        kval = implied.get(target, np.nan)
        kse = implied_se.get(f"se_{target}", np.nan)
        kt = kval / kse if np.isfinite(kval) and np.isfinite(kse) and kse > 0 else np.nan
        kp = normal_pval_from_t(kt)
        implied[f"t_{target}"] = kt
        implied[f"p_{target}"] = kp
        implied[f"star_{target}"] = stars(kp)

    rows.append({
        "run": run_name,
        "sample": sample_label,
        "dimension": dim_label,
        "spec": spec,
        "method": method,
        "beta_d": beta,
        "se_beta_d": se_beta,
        "t_beta_d": t_beta,
        "p_beta_d": p_beta,
        "star_beta_d": stars(p_beta),
        "rho_lag": rho_lag,
        "se_rho_lag": se_rho_lag,
        "rho_family": rho_family,
        "se_rho_family": se_rho_family,
        "rho_ideal_implied": implied.get("rho_ideal_implied", np.nan),
        "kappa_ideal": implied.get("kappa_ideal", np.nan),
        "se_kappa_ideal": implied_se.get("se_kappa_ideal", np.nan),
        "t_kappa_ideal": implied.get("t_kappa_ideal", np.nan),
        "p_kappa_ideal": implied.get("p_kappa_ideal", np.nan),
        "star_kappa_ideal": implied.get("star_kappa_ideal", ""),
        "kappa_lag": implied.get("kappa_lag", np.nan),
        "se_kappa_lag": implied_se.get("se_kappa_lag", np.nan),
        "t_kappa_lag": implied.get("t_kappa_lag", np.nan),
        "p_kappa_lag": implied.get("p_kappa_lag", np.nan),
        "star_kappa_lag": implied.get("star_kappa_lag", ""),
        "kappa_family": implied.get("kappa_family", np.nan),
        "se_kappa_family": implied_se.get("se_kappa_family", np.nan),
        "t_kappa_family": implied.get("t_kappa_family", np.nan),
        "p_kappa_family": implied.get("p_kappa_family", np.nan),
        "star_kappa_family": implied.get("star_kappa_family", ""),
        "rho_sum_implied": np.nansum([implied.get("rho_ideal_implied", np.nan), implied.get("rho_lag", np.nan), implied.get("rho_family", np.nan)]),
        "negative_any_kappa": int(any(np.isfinite(implied.get(k, np.nan)) and implied.get(k, np.nan) < 0 for k in ["kappa_ideal", "kappa_lag", "kappa_family"])),
        "fs_F": float(est.get("fs_F", np.nan)) if "fs_F" in est else np.nan,
        "n": int(est["n"]),
        "n_parties": int(n_parties),
        "kZ": int(kZ),
    })


def main():
    auto_path, demo_path, mpd_path, spreadsheet_path = resolve_paths()

    print("Starting PyBLP multicost supply estimator...")
    print(f"AUTO input: {auto_path}")
    print(f"DEMO input: {demo_path}")
    print(f"MPD input:  {mpd_path}")
    print(f"Sign sheet: {spreadsheet_path}")
    print(f"Output dir: {OUT_DIR}")
    print(f"Leave-out scope: {LEAVEOUT_SCOPE}")

    # ---------------------------------------------------------------------
    # Spreadsheet / bundle definitions
    # ---------------------------------------------------------------------
    sheet0 = pd.read_excel(spreadsheet_path, sheet_name=0)
    sheet, code_col, econ_sign_col, cult_sign_col = pick_sign_columns(sheet0)

    full_econ = get_signed_percode_list(sheet, econ_sign_col)
    full_cult = get_signed_percode_list(sheet, cult_sign_col)
    econ_codes = [(pc, s) for (pc, s) in full_econ if pc not in SPARSE_ECON_CODES]
    cult_codes = [
        (pc, s) for (pc, s) in full_cult
        if (pc not in set(DROP_FROM_CULT)) and (pc not in SPARSE_CULT_CODES)
    ]
    dropped_econ = sorted(SPARSE_ECON_CODES)
    dropped_cult = sorted(set(DROP_FROM_CULT).union(SPARSE_CULT_CODES))

    econ_pcs = [pc for pc, _ in econ_codes]
    cult_pcs = [pc for pc, _ in cult_codes]
    all_percodes = sorted(set(econ_pcs + cult_pcs))

    bundle_df = pd.DataFrame([
        {
            "bundle": "economic",
            "k_used": len(econ_codes),
            "codes_used": ",".join(econ_pcs),
            "codes_dropped": ",".join(dropped_econ),
            "rule": "common_support_complete_case",
        },
        {
            "bundle": "cultural",
            "k_used": len(cult_codes),
            "codes_used": ",".join(cult_pcs),
            "codes_dropped": ",".join(dropped_cult),
            "rule": "common_support_complete_case",
        },
    ])
    bundle_df.to_csv(OUT_DIR / "bundle_definition_common_support_complete_case.csv", index=False)

    print("Spreadsheet sign columns used:")
    print(f"  ECON sign column: {econ_sign_col}  (k={len(econ_codes)} after dropping sparse codes)")
    print(f"  CULT sign column: {cult_sign_col}  (k={len(cult_codes)} after dropping per103 + sparse codes)")
    print(f"  Dropped ECON sparse codes: {dropped_econ}")
    print(f"  Dropped CULT codes: {dropped_cult}")
    print("")

    # ---------------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------------
    auto = pd.read_stata(auto_path, convert_categoricals=False).copy()
    demo = pd.read_stata(demo_path, convert_categoricals=False).copy()
    mpd = pd.read_stata(mpd_path, convert_categoricals=False).copy()

    if SAVE_PREFLIGHT:
        preflight = build_preflight(auto, demo)
        preflight.to_csv(OUT_DIR / "preflight_diagnostics_from_backbone.csv", index=False)
        print("Saved preflight diagnostics:", OUT_DIR / "preflight_diagnostics_from_backbone.csv")

    # ---------------------------------------------------------------------
    # AUTO prep
    # ---------------------------------------------------------------------
    need = [
        "female", "educ_cat", "age", "anweight", "lrscale", "election_year", "vote", "region",
        "country", "parfam", "mpd_party_id", "partyname", "partyname_eng", "share_mpd",
        "candidate_age", "candidate_gender",
    ]
    ensure_cols(auto, need)
    for pc in all_percodes:
        if pc not in auto.columns:
            auto[pc] = pd.NA

    auto["anweight"] = to_num(auto["anweight"])
    auto["election_year"] = to_num(auto["election_year"]).astype("Int64")
    auto["candidate_age"] = to_num(auto["candidate_age"])
    auto["candidate_gender"] = to_num(auto["candidate_gender"])
    auto["mpd_party_id"] = to_num(auto["mpd_party_id"])

    auto = auto.dropna(subset=["female", "educ_cat", "age", "anweight", "lrscale", "election_year"]).copy()
    auto = auto[auto["vote"].isin([1, 2])].copy()
    auto["region4"] = auto["region"].astype(str).str[:4]
    auto = auto[(auto["region4"].notna()) & (auto["region4"] != "")].copy()
    auto["election"] = normalize_election(auto["region4"], auto["election_year"])

    # market weight for aggregation later
    market_w = (
        auto.groupby(["region4", "election_year", "election"], as_index=False)["anweight"]
            .sum()
            .rename(columns={"anweight": "market_weight"})
    )

    auto["sum_weight"] = auto.groupby(["region4", "election_year"])["anweight"].transform("sum")
    auto["sum_party_weight"] = auto.groupby(["region4", "election_year", "mpd_party_id"])["anweight"].transform("sum")
    auto["vote_share"] = auto["sum_party_weight"] / auto["sum_weight"]
    auto = auto.dropna(subset=["country", "parfam", "mpd_party_id", "vote_share"]).copy()

    prod_base_all = aggregate_auto_to_products(auto, all_percodes)
    prod_base_all = prod_base_all.merge(
        market_w[["election", "market_weight"]].drop_duplicates(),
        on="election", how="left", validate="m:1"
    )
    election_to_mid = dict(zip(prod_base_all["election"], prod_base_all["market_ids"]))

    keep_bundle = complete_mask(prod_base_all, econ_pcs) & complete_mask(prod_base_all, cult_pcs)
    prod = prod_base_all.loc[keep_bundle].copy().reset_index(drop=True)

    ess_country_years = (
        prod[["country", "election_year"]]
            .dropna()
            .drop_duplicates()
            .sort_values(["country", "election_year"])
            .reset_index(drop=True)
    )

    # ---------------------------------------------------------------------
    # DEMO prep
    # ---------------------------------------------------------------------
    ensure_cols(demo, ["election", "college", "age", "lrscale", "female", "young", "middle", "old", "boomer"])
    demo["election"] = demo["election"].astype(str).str.lower().str.replace(" ", "", regex=False)
    demo["market_ids"] = demo["election"].map(election_to_mid)
    demo = demo.dropna(subset=["market_ids", "college", "age"]).copy()
    demo["market_ids"] = demo["market_ids"].astype(int)
    demo = demo[demo["market_ids"].isin(prod["market_ids"].unique())].copy()

    # ---------------------------------------------------------------------
    # MPD prep
    # ---------------------------------------------------------------------
    ensure_cols(mpd, ["countryname", "parfam", "date", "pervote"])
    mpd["country"] = mpd["countryname"].astype(str).str.strip()
    mpd["election_year"] = (to_num(mpd["date"]) // 100).astype("Int64")
    mpd["pervote"] = to_num(mpd["pervote"])
    for pc in all_percodes:
        if pc not in mpd.columns:
            mpd[pc] = pd.NA

    keep_mpd_bundle = complete_mask(mpd, econ_pcs) & complete_mask(mpd, cult_pcs)
    mpd_cc = mpd.loc[keep_mpd_bundle].copy()
    mpd_cc["idx_economic_raw"] = build_equal_weight_index(mpd_cc, econ_codes) * float(ECON_RIGHT_MULT)
    mpd_cc["idx_cultural_raw"] = build_equal_weight_index(mpd_cc, cult_codes) * float(CULT_RIGHT_MULT)

    # ---------------------------------------------------------------------
    # Product-level indices and leave-outs
    # ---------------------------------------------------------------------
    prod["idx_economic_raw"] = build_equal_weight_index(prod, econ_codes) * float(ECON_RIGHT_MULT)
    prod["idx_cultural_raw"] = build_equal_weight_index(prod, cult_codes) * float(CULT_RIGHT_MULT)
    mu_e, sd_e = zscore_params(prod["idx_economic_raw"])
    mu_c, sd_c = zscore_params(prod["idx_cultural_raw"])
    prod["idx_economic"] = apply_z(prod["idx_economic_raw"], mu_e, sd_e) if Z_SCORE_ECONOMIC else prod["idx_economic_raw"]
    prod["idx_cultural"] = apply_z(prod["idx_cultural_raw"], mu_c, sd_c) if Z_SCORE_CULTURE else prod["idx_cultural_raw"]

    if LEAVEOUT_SCOPE == "full_mpd":
        mpd_for_lo = mpd_cc.copy()
    elif LEAVEOUT_SCOPE == "ess_sample_country_year":
        mpd_for_lo = mpd_cc.merge(
            ess_country_years,
            on=["country", "election_year"],
            how="inner",
            validate="m:1",
        )
    else:
        raise ValueError(f"Unknown LEAVEOUT_SCOPE: {LEAVEOUT_SCOPE}")

    mpd_min = mpd_for_lo[["country", "parfam", "election_year", "pervote", "idx_economic_raw", "idx_cultural_raw"]].copy()
    mpd_min = mpd_min.dropna(subset=["country", "parfam", "election_year", "pervote"]).copy()

    lo_econ = compute_leaveout_family(
        mpd_min.dropna(subset=["idx_economic_raw"]),
        idx_col="idx_economic_raw",
        out_prefix="lo_fam_econ",
    )
    lo_cult = compute_leaveout_family(
        mpd_min.dropna(subset=["idx_cultural_raw"]),
        idx_col="idx_cultural_raw",
        out_prefix="lo_fam_cult",
    )

    prod = prod.merge(lo_econ, on=["country", "parfam", "election_year"], how="left", validate="m:1")
    prod = prod.merge(lo_cult, on=["country", "parfam", "election_year"], how="left", validate="m:1")

    if Z_SCORE_ECONOMIC:
        prod["lo_fam_econ"] = apply_z(prod["lo_fam_econ"], mu_e, sd_e)
        prod["lo_fam_econ2"] = (prod["lo_fam_econ"] ** 2) / 100.0
    if Z_SCORE_CULTURE:
        prod["lo_fam_cult"] = apply_z(prod["lo_fam_cult"], mu_c, sd_c)
        prod["lo_fam_cult2"] = (prod["lo_fam_cult"] ** 2) / 100.0

    n_before_core_drop = len(prod)
    prod = prod.dropna(
        subset=[
            "shares", "candidate_age", "candidate_gender", "idx_economic", "idx_cultural",
            "lo_fam_econ", "lo_fam_econ2", "lo_fam_cult", "lo_fam_cult2",
        ]
    ).copy().reset_index(drop=True)
    n_after_core_drop = len(prod)

    prod, demand_instr_cols = make_demand_instruments(prod)
    demo_use = demo[demo["market_ids"].isin(prod["market_ids"].unique())].copy()

    if SAVE_PRODUCT_DATA:
        prod.to_csv(OUT_DIR / "product_data_used_in_blp_multicost.csv", index=False)

    # ---------------------------------------------------------------------
    # Demand estimation
    # ---------------------------------------------------------------------
    X1 = pyblp.Formulation("1 + prices + candidate_age + candidate_gender + idx_economic + idx_cultural")
    X2 = pyblp.Formulation("0 + idx_economic + idx_cultural")
    
    #demo_use["age_z"] = (demo_use["age"] - demo_use["age"].mean()) / demo_use["age"].std(ddof=0)
    agent_form = pyblp.Formulation("0 + college + young + old")
    integration = pyblp.Integration("halton", size=INTEGRATION_SIZE)

    problem = pyblp.Problem(
        product_formulations=(X1, X2),
        product_data=prod,
        agent_formulation=agent_form,
        agent_data=demo_use,
        integration=integration,
        add_exogenous=False,
    )

    sigma_start = np.eye(problem.K2) * 1e-4
    #sigma_start = np.array([
    #    [1e-4, 0.0],
    #    [1e-4, 1e-4],
    #], dtype=float)
    #pi_start = np.zeros((problem.K2, problem.D))
    pi_start = np.array([
        [1e-4,1e-4,1e-4],
        [1e-4,1e-4,1e-4],
    ], dtype=float)
    beta = [None, 0.0, None, None] + [None] * problem.K2

    opt = pyblp.Optimization(
        "l-bfgs-b",
        {"gtol": 1e-8, "maxiter": OPT_MAXITER, "maxls": 50, "ftol": 1e-12},
    )

    results = problem.solve(
        sigma_start, pi_start,
        beta=beta,
        method="2s",
        fp_type="nonlinear",
        W_type="unadjusted",
        optimization=opt,
    )

    if PRINT_PYBLP_RESULTS:
        print(results)

    b = results.beta.ravel()
    se = results.beta_se.ravel()
    t_econ = b[4] / se[4] if np.isfinite(se[4]) and se[4] > 0 else np.nan
    p_econ = normal_pval_from_t(t_econ)
    t_cult = b[5] / se[5] if np.isfinite(se[5]) and se[5] > 0 else np.nan
    p_cult = normal_pval_from_t(t_cult)

    demand_df = pd.DataFrame([dict(
        run=RUN_NAME,
        leaveout_scope=LEAVEOUT_SCOPE,
        auto_input=str(auto_path),
        demo_input=str(demo_path),
        mpd_input=str(mpd_path),
        signsheet_input=str(spreadsheet_path),
        bundle_rule="common_support_complete_case",
        dropped_econ_codes=",".join(dropped_econ),
        dropped_cult_codes=",".join(dropped_cult),
        Z_SCORE_ECONOMIC=bool(Z_SCORE_ECONOMIC),
        Z_SCORE_CULTURE=bool(Z_SCORE_CULTURE),
        k_econ=int(len(econ_codes)),
        k_cult=int(len(cult_codes)),
        N_products=int(len(prod)),
        N_markets=int(prod["market_ids"].nunique()),
        N_agents=int(len(demo_use)),
        beta_idx_economic=float(b[4]),
        se_idx_economic=float(se[4]),
        t_idx_economic=float(t_econ),
        p_idx_economic=float(p_econ),
        star_idx_economic=stars(p_econ),
        beta_idx_cultural=float(b[5]),
        se_idx_cultural=float(se[5]),
        t_idx_cultural=float(t_cult),
        p_idx_cultural=float(p_cult),
        star_idx_cultural=stars(p_cult),
        sigma_econ=float(results.sigma[0, 0]) if results.sigma.size else np.nan,
        sigma_cult=float(results.sigma[1, 1]) if results.sigma.size > 1 else np.nan,
        econ_mu=float(mu_e),
        econ_sd=float(sd_e) if np.isfinite(sd_e) else np.nan,
        cult_mu=float(mu_c),
        cult_sd=float(sd_c) if np.isfinite(sd_c) else np.nan,
        n_products_pre_core_drop=int(n_before_core_drop),
        n_products_post_core_drop=int(n_after_core_drop),
        n_demand_instr=int(len(demand_instr_cols)),
        n_mpd_country_years_used_for_lo=int(mpd_for_lo[["country", "election_year"]].drop_duplicates().shape[0]),
        n_mpd_party_rows_used_for_lo=int(len(mpd_for_lo)),
    )])
    demand_df.to_csv(OUT_DIR / "demand_common_support_complete_case_multicost.csv", index=False)

    # ---------------------------------------------------------------------
    # Derivatives and supply panel
    # ---------------------------------------------------------------------
    d_econ = own_derivative_vector(results, "idx_economic")
    d_cult = own_derivative_vector(results, "idx_cultural")

    prodZ, Zcols = add_market_rich_Z(prod, demo_use)
    pe = build_party_election_panel(prodZ, d_econ, d_cult, Zcols)
    pe.to_csv(OUT_DIR / "party_election_supply_panel.csv", index=False)

    # ---------------------------------------------------------------------
    # Supply estimation: all 7 specs, OLS and IV, 2 dimensions, 2 samples
    # ---------------------------------------------------------------------
    supply_rows = []
    sample_rows = []

    for sample_label, thresh in SUPPLY_THRESHOLDS:
        if thresh is None:
            pe_s = pe.copy()
        else:
            pe_s = pe.loc[pe["party_max_share_any_market"] >= float(thresh)].copy()

        sample_rows.append({
            "run": RUN_NAME,
            "leaveout_scope": LEAVEOUT_SCOPE,
            "sample": sample_label,
            "n_party_elections": int(len(pe_s)),
            "n_parties": int(pe_s["party_key"].nunique()),
            "kZ": int(len(Zcols)),
            "n_products_final": int(len(prod)),
            "n_markets_final": int(prod["market_ids"].nunique()),
            "n_agents_final": int(len(demo_use)),
            "n_mpd_country_years_used_for_lo": int(mpd_for_lo[["country", "election_year"]].drop_duplicates().shape[0]),
            "n_mpd_party_rows_used_for_lo": int(len(mpd_for_lo)),
        })

        for dim_label, dim in [("Economic", "econ"), ("Cultural", "cult")]:
            for spec in SUPPLY_SPECS:
                panel, y, d, W, Zx, add_const, param_names = build_spec_matrices(pe_s, dim, spec, Zcols)
                if len(panel) == 0:
                    continue

                # OLS
                try:
                    est_ols = fit_ols(y, d, W, add_const=add_const, param_names=param_names)
                    add_result_rows(
                        supply_rows, RUN_NAME, sample_label, dim_label, spec, "OLS",
                        est_ols, n_parties=panel["party_key"].nunique(), kZ=len(Zcols)
                    )
                except Exception as e:
                    warnings.warn(f"OLS failed for {sample_label} / {dim_label} / {spec}: {e}")

                # IV
                try:
                    est_iv = fit_iv_one_endog(y, d, W, Zx, add_const=add_const, param_names=param_names)
                    add_result_rows(
                        supply_rows, RUN_NAME, sample_label, dim_label, spec, "IV",
                        est_iv, n_parties=panel["party_key"].nunique(), kZ=len(Zcols)
                    )
                except Exception as e:
                    warnings.warn(f"IV failed for {sample_label} / {dim_label} / {spec}: {e}")

    supply_df = pd.DataFrame(supply_rows)
    sample_df = pd.DataFrame(sample_rows)
    supply_df.to_csv(OUT_DIR / "supply_multicost_specs.csv", index=False)
    sample_df.to_csv(OUT_DIR / "supply_multicost_sample_summary.csv", index=False)

    # ---------------------------------------------------------------------
    # Console output
    # ---------------------------------------------------------------------
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 60)

    print("\n===================== DEMAND SUMMARY =====================")
    print(demand_df)

    print("\n===================== SUPPLY MULTICOST RESULTS (HEAD) =====================")
    show_cols = [
        "sample", "dimension", "spec", "method", "beta_d", "rho_ideal_implied", "rho_lag", "rho_family",
        "kappa_ideal", "kappa_lag", "kappa_family", "p_beta_d", "fs_F", "n", "n_parties"
    ]
    print(supply_df[show_cols].head(40))

    print("\nSaved outputs to:")
    print("  ", OUT_DIR / "demand_common_support_complete_case_multicost.csv")
    print("  ", OUT_DIR / "supply_multicost_specs.csv")
    print("  ", OUT_DIR / "supply_multicost_sample_summary.csv")
    print("  ", OUT_DIR / "party_election_supply_panel.csv")
    print("  ", OUT_DIR / "product_data_used_in_blp_multicost.csv")
    print("  ", OUT_DIR / "bundle_definition_common_support_complete_case.csv")
    if SAVE_PREFLIGHT:
        print("  ", OUT_DIR / "preflight_diagnostics_from_backbone.csv")
    print("\nDONE.")


if __name__ == "__main__":
    main()