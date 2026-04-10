/*-----------------------------------------------------------------------------
Project: TR 2025-26 Chapter 4
Autor: Pablo Garcia Guzman
This do-file: Mirror the EXACT order of operations in a3_reg_age_cohorts_idx_TR.do
              and report N at every step. Pinpoint where the obs drop happens.
-----------------------------------------------------------------------------*/

**# Set paths
	global root "/Users/pablogguz_/Documents/GitHub/poli-io-new/pablo/"
	global data "/Users/pablogguz_/Library/CloudStorage/Dropbox/ess-mpd-paper/0_data/"

	clear all
	set maxvar 120000

	log using "$root/log/verify_obs_drop.log", replace

	cap program drop step
	program define step
	    args label
	    qui count
	    di as txt %-65s "  `label'" as result %12.0fc r(N)
	end

	di "================================================================"
	di "  Mirror a3_reg_age_cohorts_idx_TR.do step by step on NEW file"
	di "================================================================"

	use "$data/Zhaosheng_proc/ess_mpd_matched_validated_plus_ai_rigorous_v6_2.dta", clear
	step "0. raw load"

	// --- Variable construction (matching a3 script exactly) ---
	g dob = yrbrn if !missing(yrbrn) & yrbrn > 0
	step "after: g dob = yrbrn if !missing(yrbrn) & yrbrn > 0"

	g female = (gndr == 2) if !missing(gndr)

	g _educ = .
	replace _educ = edulvla if !missing(edulvla) & edulvla < .
	replace _educ = 1 if edulvlb == 0   & missing(_educ) & !missing(edulvlb)
	replace _educ = 1 if edulvlb == 113 & missing(_educ) & !missing(edulvlb)
	replace _educ = 2 if inrange(edulvlb, 129, 229) & missing(_educ) & !missing(edulvlb)
	replace _educ = 2 if inrange(edulvlb, 212, 213) & missing(_educ) & !missing(edulvlb)
	replace _educ = 3 if inrange(edulvlb, 311, 323) & missing(_educ) & !missing(edulvlb)
	replace _educ = 4 if inrange(edulvlb, 412, 423) & missing(_educ) & !missing(edulvlb)
	replace _educ = 5 if inrange(edulvlb, 510, 800) & missing(_educ) & !missing(edulvlb)
	g educ_cat = 1 if inlist(_educ, 1, 2)
	replace educ_cat = 2 if inlist(_educ, 3, 4)
	replace educ_cat = 3 if _educ == 5
	drop _educ

	g employed = (pdwrk == 1) if pdwrk >= 0 & !missing(pdwrk)

	g age_at_election = election_year - dob
	g age_cat_el = 1 if inrange(age_at_election, 18, 34)
	replace age_cat_el = 2 if inrange(age_at_election, 35, 49)
	replace age_cat_el = 3 if inrange(age_at_election, 50, 64)
	replace age_cat_el = 4 if age_at_election >= 65 & !missing(age_at_election)

	g cohort = .
	replace cohort = 1 if dob <= 1945
	replace cohort = 2 if dob >= 1946 & dob <= 1964
	replace cohort = 3 if dob >= 1965 & dob <= 1980
	replace cohort = 4 if dob >= 1981 & !missing(dob)

	g ebrd = 0
	foreach c in ALB BGR BIH CZE EST GEO HRV HUN KAZ KGZ LTU LVA MDA ///
	             MKD MNE MNG POL ROU SRB SVK SVN TJK TKM TUR UKR UZB XKX {
	    replace ebrd = 1 if iso3c == "`c'"
	}

	// --- Index construction (matching a3 script - uses cap g, no filter) ---
	foreach v in per403 per405 per412 per413 per415 per503 per504 per506 per701 {
	    cap g _es_`v' = `v'
	}
	foreach v in per401 per414 per505 per507 per702 {
	    cap g _es_`v' = -`v'
	}
	egen idx_econ = rowmean(_es_*)

	foreach v in per109 per110 per601 per603 per608 {
	    cap g _cs_`v' = `v'
	}
	foreach v in per107 per108 per602 per604 per607 {
	    cap g _cs_`v' = -`v'
	}
	egen idx_cult = rowmean(_cs_*)

	step "1. after constructing idx_econ / idx_cult (rowmean of _es_*)"

	qui count if !missing(idx_econ) & !missing(idx_cult)
	di as txt "    of which idx_econ AND idx_cult non-missing: " as result %12.0fc r(N)

	qui count if !missing(idx_econ)
	di as txt "    of which idx_econ non-missing: " as result %12.0fc r(N)

	// --- The EBRD/adv_europe sample filter ---
	g adv_europe = 0
	replace adv_europe = 1 if iso3c == "AUT"
	replace adv_europe = 1 if iso3c == "BEL"
	replace adv_europe = 1 if iso3c == "DNK"
	replace adv_europe = 1 if iso3c == "FIN"
	replace adv_europe = 1 if iso3c == "FRA"
	replace adv_europe = 1 if iso3c == "DEU"
	replace adv_europe = 1 if iso3c == "GRC"
	replace adv_europe = 1 if iso3c == "IRL"
	replace adv_europe = 1 if iso3c == "ITA"
	replace adv_europe = 1 if iso3c == "LUX"
	replace adv_europe = 1 if iso3c == "NLD"
	replace adv_europe = 1 if iso3c == "PRT"
	replace adv_europe = 1 if iso3c == "ESP"
	replace adv_europe = 1 if iso3c == "SWE"
	replace adv_europe = 1 if iso3c == "GBR"
	replace adv_europe = 1 if iso3c == "NOR"
	replace adv_europe = 1 if iso3c == "CHE"

	g ebrd_cat = 1 if ebrd == 1
	replace ebrd_cat = 2 if adv_europe == 1
	replace ebrd_cat = 1 if iso3c == "GRC"

	drop if missing(ebrd_cat)
	step "2. drop if missing(ebrd_cat)"

	drop if missing(dob)
	step "3. drop if missing(dob)"

	// reghdfe will then use only obs with non-missing outcome and all controls
	di _n "  Reghdfe sample diagnostic (mimicking what reghdfe does):"
	count if !missing(idx_econ) & !missing(female) & !missing(educ_cat) & ///
	         !missing(employed) & !missing(domicil) & !missing(age_cat_el)
	di "    ^^ idx_econ + age_cat_el + 4 controls all non-missing"

	count if !missing(idx_econ) & !missing(age_cat_el)
	di "    ^^ idx_econ + age_cat_el non-missing (no-controls model 1)"

	count if !missing(idx_cult) & !missing(age_cat_el)
	di "    ^^ idx_cult + age_cat_el non-missing"

	di _n "  Per-code missingness in current sample:"
	foreach v in per401 per403 per405 per412 per413 per414 per415 per503 per504 per505 per506 per507 per701 per702 {
	    qui count if missing(`v')
	    di as txt "    `v': " as result %10.0fc r(N) as txt " missing"
	}

	di _n "  Per-code missingness for cultural codes:"
	foreach v in per107 per108 per109 per110 per601 per602 per603 per604 per607 per608 {
	    qui count if missing(`v')
	    di as txt "    `v': " as result %10.0fc r(N) as txt " missing"
	}

	log close
