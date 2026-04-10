/*-----------------------------------------------------------------------------
Project: TR 2025-26 Chapter 4
Autor: Pablo Garcia Guzman
This do-file: Compare old (clean_reg_TR) vs new (Zhaosheng AI) datasets
              in terms of obs, countries, election years, match coverage
-----------------------------------------------------------------------------*/

**# Set paths
	global root "/Users/pablogguz_/Documents/GitHub/poli-io-new/pablo/"
	global data "/Users/pablogguz_/Library/CloudStorage/Dropbox/ess-mpd-paper/0_data/"

	global proc "$data/proc/"

	clear all
	set maxvar 120000

	log using "$root/log/compare_datasets.log", replace

	di "============================================================"
	di "  COMPARISON: old clean_reg_TR vs Zhaosheng AI-validated"
	di "============================================================"

**# ---- OLD dataset --------------------------------------------------------
	di _n "============================================================"
	di "  OLD: ess_mpd_matched_clean_reg_TR.dta"
	di "============================================================"

	use "$proc/ess_mpd_matched_clean_reg_TR.dta", clear

	di _n "--- Total observations ---"
	count

	di _n "--- Election years ---"
	tab election_year

	di _n "--- Countries ---"
	tab iso3c

	di _n "--- Unique countries ---"
	distinct iso3c

	di _n "--- Obs by country x election year ---"
	bys iso3c election_year: gen _n1 = (_n == 1)
	count if _n1
	drop _n1

	di _n "--- Missing rates for key variables ---"
	foreach v in anweight female educ_cat employed domicil age_cat_el cohort dob {
		capture confirm variable `v'
		if _rc == 0 {
			qui count if missing(`v')
			local nmiss = r(N)
			qui count
			local ntot = r(N)
			di "`v': `nmiss' missing of `ntot' (" %4.1f 100*`nmiss'/`ntot' "%)"
		}
		else {
			di "`v': NOT IN DATASET"
		}
	}

	di _n "--- Per-code availability (non-missing share) ---"
	foreach v in per401 per403 per405 per412 per413 per414 per415 ///
	             per503 per504 per505 per506 per507 per701 per702 ///
	             per107 per108 per109 per110 ///
	             per601 per602 per603 per604 per607 per608 {
		qui count if !missing(`v')
		local nhave = r(N)
		qui count
		local ntot = r(N)
		di "`v': `nhave' / `ntot' (" %4.1f 100*`nhave'/`ntot' "%)"
	}

	// Save country-year obs counts for comparison
	preserve
		collapse (count) n_old = anweight, by(iso3c election_year)
		tempfile old_counts
		save `old_counts'
	restore

**# ---- NEW dataset --------------------------------------------------------
	di _n _n "============================================================"
	di "  NEW: ess_mpd_matched_validated_plus_ai_rigorous_v6_2.dta"
	di "============================================================"

	use "$data/Zhaosheng_proc/ess_mpd_matched_validated_plus_ai_rigorous_v6_2.dta", clear

	di _n "--- Total observations ---"
	count

	di _n "--- Election years ---"
	tab election_year

	di _n "--- Countries ---"
	tab iso3c

	di _n "--- Unique countries ---"
	distinct iso3c

	di _n "--- Obs by country x election year ---"
	bys iso3c election_year: gen _n1 = (_n == 1)
	count if _n1
	drop _n1

	di _n "--- Missing rates for key variables ---"
	foreach v in anweight gndr edulvla pdwrk domicil age yrbrn {
		capture confirm variable `v'
		if _rc == 0 {
			qui count if missing(`v')
			local nmiss = r(N)
			qui count
			local ntot = r(N)
			di "`v': `nmiss' missing of `ntot' (" %4.1f 100*`nmiss'/`ntot' "%)"
		}
		else {
			di "`v': NOT IN DATASET"
		}
	}

	di _n "--- Per-code availability (non-missing share) ---"
	foreach v in per401 per403 per405 per412 per413 per414 per415 ///
	             per503 per504 per505 per506 per507 per701 per702 ///
	             per107 per108 per109 per110 ///
	             per601 per602 per603 per604 per607 per608 {
		qui count if !missing(`v')
		local nhave = r(N)
		qui count
		local ntot = r(N)
		di "`v': `nhave' / `ntot' (" %4.1f 100*`nhave'/`ntot' "%)"
	}

	// Save country-year obs counts for comparison
	preserve
		collapse (count) n_new = anweight, by(iso3c election_year)
		tempfile new_counts
		save `new_counts'
	restore

**# ---- Side-by-side comparison --------------------------------------------
	di _n _n "============================================================"
	di "  SIDE-BY-SIDE: obs per country x election year"
	di "============================================================"

	use `old_counts', clear
	merge 1:1 iso3c election_year using `new_counts'

	g diff = n_new - n_old
	g pct_diff = 100 * diff / n_old

	di _n "--- Merge result ---"
	tab _merge

	di _n "--- Country x year cells: obs comparison ---"
	list iso3c election_year n_old n_new diff pct_diff if _merge == 3, sep(0) noobs

	di _n "--- Only in OLD ---"
	list iso3c election_year n_old if _merge == 1, sep(0) noobs

	di _n "--- Only in NEW ---"
	list iso3c election_year n_new if _merge == 2, sep(0) noobs

	di _n "--- Summary of differences (matched cells) ---"
	su diff pct_diff if _merge == 3

	di _n "--- Total obs ---"
	su n_old n_new

	log close
