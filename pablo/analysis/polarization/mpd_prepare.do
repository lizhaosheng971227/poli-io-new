/*-----------------------------------------------------------------------------
Project: ESS-MPD paper
Author: Pablo Garcia Guzman
This do-file: prepares MPD data for polarization charts over time
	- Constructs economic and cultural dimension indices (as in clean_for_reg)
	- Computes p80-p20 differences by election period

	ESS file w/ partyfacts ids: https://github.com/hdigital/partyfactsdata/tree/main/import/essprtv
	Link Partyfacts-MPD: https://partyfacts.herokuapp.com/download/
-----------------------------------------------------------------------------*/

**# Set paths
	global root "/Users/pablogguz_/Documents/GitHub/poli-io-new/pablo/"
	global data "/Users/pablogguz_/Library/CloudStorage/Dropbox/ess-mpd-paper/0_data/"

	global raw  "$data/raw/"
	global proc "$data/proc/"

	global fig "$root/output/figures/"
	global tab "$root/output/tables/"

	clear all
	set maxvar 120000

	log using "$root/log/mpd_prepare.log", replace

**# Load countries in ESS
    use "$proc/ess_mpd_matched_clean_reg_TR.dta", clear

    bys iso2c: keep if _n == 1
	keep iso2c
    tempfile ess_countries
    save `ess_countries'

**# Load MPD data
	use "$raw/MPD/MPDataset_MPDS2024a_stata14.dta", clear

	isocodes countryname, gen(iso2c)

	merge m:1 iso2c using `ess_countries'
	keep if _m == 3
	drop _m

	g election_year = year(edate)
    ebrdify iso2c

    g election_period = 1 if inrange(election_year, 1995, 1999)
    replace election_period = 2 if inrange(election_year, 2000, 2004)
    replace election_period = 3 if inrange(election_year, 2005, 2009)
    replace election_period = 4 if inrange(election_year, 2010, 2014)
    replace election_period = 5 if inrange(election_year, 2015, 2019)
    drop if missing(election_period)
    lab define election_period ///
        1 "1995-1999" 2 "2000-2004" 3 "2005-2009" ///
        4 "2010-2014" 5 "2015-2019", replace
    lab values election_period election_period

	// drop Russia, Cyprus, Iceland
	drop if iso3c == "RUS"
	drop if iso3c == "CYP"
    drop if iso3c == "ISL"

**# Construct economic and cultural dimension indices

    // --- Economic index: higher = more left-wing (pro-welfare, pro-regulation)
    foreach v in per403 per405 per412 per413 per415 ///
                 per503 per504 per506 per701 {
        cap g _es_`v' = `v'
    }
    foreach v in per401 per414 per505 per507 per702 {
        cap g _es_`v' = -`v'
    }

    egen idx_econ = rowmean(_es_*)
    lab var idx_econ "Economic dimension"

    // --- Cultural index: higher = more traditional / anti-globalisation / sovereignty
    foreach v in per109 per110 per601 per603 per608 {
        cap g _cs_`v' = `v'
    }
    foreach v in per107 per108 per602 per604 per607 {
        cap g _cs_`v' = -`v'
    }

    egen idx_cult = rowmean(_cs_*)
    lab var idx_cult "Cultural dimension"

    drop _es_* _cs_*

**# Collapse to p80-p20 by election period

	// within each election period, for each country, keep the latest election
	egen max_year = max(election_year), by(iso3c election_period)
	keep if election_year == max_year

	gcollapse (p80) p80_econ = idx_econ p80_cult = idx_cult ///
	          (p20) p20_econ = idx_econ p20_cult = idx_cult ///
			  [aw=pervote], by(iso3c election_period)

	g p82_idx_econ = p80_econ - p20_econ
	g p82_idx_cult = p80_cult - p20_cult
	lab var p82_idx_econ "Economic dimension"
	lab var p82_idx_cult "Cultural dimension"

	gcollapse (mean) p82_idx_econ p82_idx_cult, by(election_period)

	foreach var of varlist p82_* {
		local lab : variable label `var'
		local lab = subinstr("`lab'", "(mean) ", "", .)
		lab var `var' "`lab'"
	}

	compress
	lab drop election_period
	save "$tab/MPD_overtime_charts_pol.dta", replace

	log close
