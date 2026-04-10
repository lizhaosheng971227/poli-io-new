/*-----------------------------------------------------------------------------
Project: TR 2025-26 Chapter 4
Autor: Pablo Garcia Guzman
This do-file: Check MPD per-code availability across election years
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

	log using "$root/log/a_coverage_check.log", replace

**# Load raw MPD data
    // Try raw MPD file first; if not available locally, fall back to matched data
    // but collapse to unique party-elections (not individual respondents)
    capture confirm file "$raw/MPD/MPDataset_MPDS2025a_stata14.dta"
    if _rc == 0 {
        di "Loading raw MPD file..."
        use "$raw/MPD/MPDataset_MPDS2025a_stata14.dta", clear

        // Generate election year from date variable (YYYYMM format)
        g election_year = floor(date / 100)
        keep if election_year >= 1995

        // Keep only ESS countries
        isocodes countryname, gen(iso2c)
        local ess_countries "AL AT BE BG CH CY CZ DE DK EE ES FI FR GB GR HR HU IE IL IS IT LT LU LV ME MK MN NL NO PL PT RO RS SE SI SK UA XK"
        g ess_country = 0
        foreach c of local ess_countries {
            replace ess_country = 1 if iso2c == "`c'"
        }
        keep if ess_country == 1
        drop ess_country

        // Each observation is a party-election
        keep party election_year iso2c ///
            per401 per403 per405 per412 per413 per414 per415 ///
            per4012 per4123 per4124 ///
            per503 per504 per505 per506 per507 ///
            per701 per702 ///
            per5041 per5061 ///
            per107 per108 per109 per110 ///
            per601 per602 per603 per604 per607 per608 ///
            per2022 per2023 per7062
    }
    else {
        di "Raw MPD not available locally, using matched data at party-election level..."
        use "$proc/ess_mpd_matched_validated.dta", clear

        // Keep only observations with a valid MPD party match
        keep if !missing(mpd_party_id)

        // Collapse to unique party-elections (one obs per party-election)
        bys mpd_party_id election_year: keep if _n == 1

        keep mpd_party_id election_year ///
            per401 per403 per405 per412 per413 per414 per415 ///
            per4012 per4123 per4124 ///
            per503 per504 per505 per506 per507 ///
            per701 per702 ///
            per5041 per5061 ///
            per107 per108 per109 per110 ///
            per601 per602 per603 per604 per607 per608 ///
            per2022 per2023 per7062
    }

**# Generate availability indicators for each per-code
    foreach v in per401 per403 per405 per412 per413 per414 per415 ///
        per4012 per4123 per4124 ///
        per503 per504 per505 per506 per507 ///
        per701 per702 ///
        per5041 per5061 ///
        per107 per108 per109 per110 ///
        per601 per602 per603 per604 per607 per608 ///
        per2022 per2023 per7062 {
        g avail_`v' = !missing(`v')
    }

**# Collapse to election_year level: share of party-elections with data
    collapse (mean) avail_*, by(election_year)

**# Export for R visualization
    export delimited using "$tab/mpd_code_availability.csv", replace

    log close
