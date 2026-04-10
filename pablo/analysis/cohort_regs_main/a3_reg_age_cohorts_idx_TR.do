/*-----------------------------------------------------------------------------
Project: TR 2025-26 Chapter 4
Autor: Pablo Garcia Guzman
This do-file: Age-cohort regressions on composite policy indices (raw, std, PCA)
              using the Zhaosheng AI-validated matched data
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
    cap log close
	log using "$root/log/a3_reg_age_cohorts_idx_TR.log", replace

**# Load
	use "$data/Zhaosheng_proc/ess_mpd_matched_validated_plus_ai_rigorous_v6_2.dta", clear
    tab election_year
    distinct iso3c 
    
    gen anweight_rounds1_8 = pspwght*pweight 
    replace anweight = anweight_rounds1_8 if anweight == . & essround < 9
    g miss = missing(anweight)
    tab iso3c if miss == 1 // no missings
    cap drop miss
    lab var anweight "Analytical weight"
    
**# Construct variables missing from this dataset ----------------------------
    // (Following logic from 2d. clean_for_reg.do)

    // --- dob (date of birth = birth year)
    g dob = yrbrn if !missing(yrbrn) & yrbrn > 0

    // --- female
    g female = (gndr == 2) if !missing(gndr)
    lab var female "Female"
    lab define female 1 "Female" 0 "Male", replace
    lab values female female

    // --- educ_cat (from edulvla, fallback edulvlb)
    g educ = .
    replace educ = edulvla if !missing(edulvla) & edulvla < .
    // edulvlb fallback mapping to ISCED
    replace educ = 1 if edulvlb == 0   & missing(educ) & !missing(edulvlb)
    replace educ = 1 if edulvlb == 113 & missing(educ) & !missing(edulvlb)
    replace educ = 2 if inrange(edulvlb, 129, 229) & missing(educ) & !missing(edulvlb)
    replace educ = 2 if inrange(edulvlb, 212, 213) & missing(educ) & !missing(edulvlb)
    replace educ = 3 if inrange(edulvlb, 311, 323) & missing(educ) & !missing(edulvlb)
    replace educ = 4 if inrange(edulvlb, 412, 423) & missing(educ) & !missing(edulvlb)
    replace educ = 5 if inrange(edulvlb, 510, 800) & missing(educ) & !missing(edulvlb)

    g educ_cat = 1 if inlist(educ, 1, 2)
    replace educ_cat = 2 if inlist(educ, 3, 4)
    replace educ_cat = 3 if educ == 5
    lab define educ_cat ///
        1 "Lower secondary or below" ///
        2 "Upper secondary or Post-secondary non-tertiary" ///
        3 "Tertiary education", replace
    lab values educ_cat educ_cat
    drop educ

    // --- employed
    g employed = (pdwrk == 1) if pdwrk >= 0 & !missing(pdwrk)
    lab define employed 0 "Not employed" 1 "Employed", replace
    lab values employed employed

    // --- age_at_election and age_cat_el
    g age_at_election = election_year - dob
    g age_cat_el = 1 if inrange(age_at_election, 18, 34)
    replace age_cat_el = 2 if inrange(age_at_election, 35, 49)
    replace age_cat_el = 3 if inrange(age_at_election, 50, 64)
    replace age_cat_el = 4 if age_at_election >= 65 & !missing(age_at_election)
    lab define age_cat_el 1 "18-34" 2 "35-49" 3 "50-64" 4 "65+", replace
    lab values age_cat_el age_cat_el
    lab var age_cat_el "Age category (at election)"

    // --- cohort
    g cohort = .
    replace cohort = 1 if dob <= 1945
    replace cohort = 2 if dob >= 1946 & dob <= 1964
    replace cohort = 3 if dob >= 1965 & dob <= 1980
    replace cohort = 4 if dob >= 1981 & !missing(dob)
    lab define cohort ///
        1 "Silent Generation or before (pre-1945)" ///
        2 "Baby Boomers (1946-1964)" ///
        3 "Generation X (1965-1980)" ///
        4 "Millennials and Gen Z (post-1981)", replace
    lab values cohort cohort
    lab var cohort "Generation category"

    // --- ebrd (EBRD region indicator, constructed manually)
    g ebrd = 0
    foreach c in ALB BGR BIH CZE EST GEO HRV HUN KAZ KGZ LTU LVA MDA ///
                 MKD MNE MNG POL ROU SRB SVK SVN TJK TKM TUR UKR UZB XKX {
        replace ebrd = 1 if iso3c == "`c'"
    }

**# Construct composite policy indices ----------------------------------------
    // CEE sub-codes dropped (0% availability from ~2014)

    // --- Economic index: higher = more left-wing (pro-welfare, pro-regulation)
    foreach v in per403 per405 per412 per413 per415 ///
                 per503 per504 per506 per701 {
        cap g _es_`v' = `v'
    }
    foreach v in per401 per414 per505 per507 per702 {
        cap g _es_`v' = -`v'
    }

    egen idx_econ = rowmean(_es_*)
    lab var idx_econ "Economic index (higher = more left-wing)"

    // --- Cultural index: higher = more traditional / anti-globalisation / sovereignty
    foreach v in per109 per110 per601 per603 per608 {
        cap g _cs_`v' = `v'
    }
    foreach v in per107 per108 per602 per604 per607 {
        cap g _cs_`v' = -`v'
    }

    egen idx_cult = rowmean(_cs_*)
    lab var idx_cult "Cultural index (higher = more traditional/sovereignty)"

    // --- Standardized indices (z-score across all observations)
    foreach idx in econ cult {
        qui su idx_`idx' [aw=anweight]
        g idx_`idx'_std = (idx_`idx' - r(mean)) / r(sd)
    }
    lab var idx_econ_std "Economic index (z-scored)"
    lab var idx_cult_std "Cultural index (z-scored)"

    // --- PCA indices (PC1 within each bundle)
    qui pca _es_*, components(1)
    predict idx_econ_pca, score
    qui corr idx_econ_pca idx_econ
    if r(rho) < 0 {
        replace idx_econ_pca = -idx_econ_pca
    }
    lab var idx_econ_pca "Economic index (PCA, PC1)"

    qui pca _cs_*, components(1)
    predict idx_cult_pca, score
    qui corr idx_cult_pca idx_cult
    if r(rho) < 0 {
        replace idx_cult_pca = -idx_cult_pca
    }
    lab var idx_cult_pca "Cultural index (PCA, PC1)"

    drop _es_* _cs_*

**# Keep EU-EBRD + adv_europe countries (same as a2. reg_age_cohorts_TR.do)
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

    tab iso3c if missing(ebrd_cat)
    drop if missing(ebrd_cat)

**# Vars for reg -------------------------------------------
    gegen ctryxelecyear = group(iso3c election_year)

    drop if missing(dob)

    // drop if election_year == 2022

**# Regression setup
    global controls "i.female i.educ_cat i.employed i.domicil"
    global ctrls_keep "*.age_cat_el *.cohort"

**# Regressions: three versions of the composite indices -------------------
    foreach version in "" "_std" "_pca" {

        local vlab "raw"
        if "`version'" == "_std" local vlab "std"
        if "`version'" == "_pca" local vlab "pca"

        estimates clear

        foreach dim in econ cult {
            local outcome "idx_`dim'`version'"

            **# No controls
                eststo `dim'1: reghdfe `outcome' i.age_cat_el [aw=anweight], absorb(ctryxelecyear) vce(cluster iso3c)

                estadd local ctryxelecyear_fes "$\times$"
                estadd local controls ""
                estadd local genfes ""

                qui: distinct iso3c if e(sample)
                estadd scalar ncont = r(ndistinct)

                qui: sum `outcome' if e(sample)
                estadd scalar mout = r(mean)
                estadd scalar sdout = r(sd)

            **# With controls
                eststo `dim'2: reghdfe `outcome' i.age_cat_el [aw=anweight], absorb($controls ctryxelecyear) vce(cluster iso3c)

                estadd local ctryxelecyear_fes "$\times$"
                estadd local controls "$\times$"
                estadd local genfes ""

                qui: distinct iso3c if e(sample)
                estadd scalar ncont = r(ndistinct)

                qui: sum `outcome' if e(sample)
                estadd scalar mout = r(mean)
                estadd scalar sdout = r(sd)

            **# Controls + generational cohorts
                eststo `dim'3: reghdfe `outcome' i.age_cat_el i.cohort [aw=anweight], absorb($controls ctryxelecyear) vce(cluster iso3c)

                estadd local ctryxelecyear_fes "$\times$"
                estadd local controls "$\times$"
                estadd local genfes "$\times$"

                qui: distinct iso3c if e(sample)
                estadd scalar ncont = r(ndistinct)

                qui: sum `outcome' if e(sample)
                estadd scalar mout = r(mean)
                estadd scalar sdout = r(sd)
        }

        // Export combined table (economic + cultural side by side)
        #d;
        esttab econ1 econ2 econ3 cult1 cult2 cult3 using "$tab/tabfinal_idx_`vlab'_TR.tex", replace
            style(tex) booktabs
            mgroups("Economic index" "Cultural index",
                pattern(1 0 0 1 0 0)
                prefix(\multicolumn{@span}{c}{) suffix(})
                span erepeat(\cmidrule(lr){@span}))
            cells(b(fmt(3) star) se(fmt(3) par)) star(* 0.10 ** 0.05 *** 0.01)
            stats(
                r2 N ctryxelecyear_fes genfes controls mout sdout,
                fmt(3 0 0 0 0 3 3)
                labels(
                    "R-squared"
                    "Observations"
                    "Country $\times$ election year FEs"
                    "Generational cohort FEs"
                    "Controls"
                    "Mean of outcome"
                    "SD of outcome"
                )
            )
            nomtitles
            nobaselevels nonotes nogaps se
            collabels(none) label keep($ctrls_keep)
            order($ctrls_keep)
            substitute(\_ _, \% \%);
        #d cr
    }

    log close
