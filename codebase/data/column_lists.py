# ── NUMERICAL (39) ────────────────────────────────────────────────────────────
numerical_cols = [

    # Ignore nothing
    "ActDuur",        # V – Activity duration at destination (min) (0-9999, <missing> missing if not a displacement)
    "Leeftijd",       # P – Age of respondent (6‥98, 99 = 99 yrs or older)
    "AfstS",          # V – Serie displacement distance NL (hectometres) (1-9999, <missing> missing if not a displacement or if a serial displacement)
    "AantVpl",        # P – Regular trips in NL (0 None, 1–25)
    "AantOVVpl",      # P – Regular public-transport trips (0 None, 1–25)
    "AantSVpl",       # P – Series trips (0 None, 1–25)
    "FactorH",        # W – Household weight factor
    "FactorP",        # W – Person weight factor
    "FactorV",        # W – Trip weight factor

    #### ----> Floris: these should ignore 11
    "HHPers",         # P – Persons in household (1‥9, 10 = ≥10, 11 = Unknown)
    "HHLft1",         # P – Household members < 6 yrs  (0‥9, 10 = ≥10, 11 = Unknown)
    "HHLft2",         # P – Household members 6–11 yrs (0‥9, 10 = ≥10, 11 = Unknown)
    "HHLft3",         # P – Household members 12–17 yrs (0‥9, 10 = ≥10, 11 = Unknown)
    "HHLft4",         # P – Household members ≥ 18 yrs (0‥9, 10 = ≥10, 11 = Unknown)
    #### ----> Floris: end of ignoring 11
            

    #### ----> Floris: we ignore >= 10
    "HHRijbewijsAu",  # P – Car licences in household (0‥8, 9 = ≥9, 10 = Unknown)
    "HHRijbewijsMo",  # P – Motorcycle licences in household (0‥8, 9 = ≥9, 10 = Unknown)
    "HHRijbewijsBr",  # P – Moped licences in household (0‥8, 9 = ≥9, 10 = Unknown)
    "HHFiets",        # P – Bicycles in household (0‥8, 9 = ≥9, 10 = Unknown)
    "HHAuto",         # P – Passenger cars in household (0‥8, 9 = ≥9, 10 = Unknown)
    "HHAutoL",        # P – Lease / company cars in household (0‥8, 9 = ≥9, 10 = Unknown)
    "OPAuto",         # P – Cars on respondent’s name (0‥8, 9 = ≥9, 10 = Unknown)
    "HHMotor",        # P – Motorcycles in household (0‥8, 9 = ≥9, 10 = Unknown)
    "OPMotor",        # P – Motorcycles on respondent’s name (0‥8, 9 = ≥9, 10 = Unknown)
    "HHBrom",         # P – Mopeds in household (0‥8, 9 = ≥9, 10 = Unknown)
    "OPBrom",         # P – Mopeds on respondent’s name (0‥8, 9 = ≥9, 10 = Unknown)
    "HHSnor",         # P – Light-mopeds in household (0‥8, 9 = ≥9, 10 = Unknown)
    "OPSnor",         # P – Light-mopeds on respondent’s name (0‥8, 9 = ≥9, 10 = Unknown)
    #### ----> Floris: end of ignoring >= 10


    #### ----> Floris: we ignore 9994 and 9995
    # ── Vehicle characteristics ────────────────────────────────────────
    "BouwjaarPa1",    # P – Model-year newest car (1885‥2022, 9994 Unknown, 9995 N/A)
    "BouwjaarPa2",    # P – Model-year 2nd car   (1885‥2022, 9994 Unknown, 9995 N/A)
    "BouwjaarPaL",    # P – Model-year lease/company car (1885‥2022, 9994 Unknown, 9995 N/A)
    #### ----> Floris: end of ignoring 9994 and 9995


    #### ----> Floris: we should ignore 0
    "AfstandOP",      # P – Total distance regular displacements (hectometres) (0 = no displacements, 1-999999)
    "AfstandSOP",     # P – Total distance series displacements (hectometres) (0 = no displacements, 1-999999)
    "AfstV",          # V – Displacement distance NL (hectometres) (0 No regular displacements in NL, 1..99999, <missing> missing if not a displacement or if a serial displacement)
    "AfstR",          # R – Trip distance NL (hectometres)  (0 = not a trip in NL, 1-999979 = distance in hectometers, <missing> missing if not a displacement)
    "AfstRBL",        # R – Trip distance abroad (hectometres) (0 = not a trip abroad, 1-999979 = distance in hectometers, <missing> missing if not a displacement)
    #### ----> Floris: end of ignoring 0
]


# ── BINARY (36) ────────────────────────────────────────────────────────────────
binary_cols = [
    ####-----> Floris: These should only include 0 or 1. 
    "WrkVerg",          # P – Any travel cost reimbursement (0 No, 1 Yes, 2 Unknown, 3 N/A)
    "MeerWink",         # V – Multiple shops visited (0 No, 1 Yes, 2 Unknown, 3 N/A – no shopping, <missing> missing if not a displacement)
    "OPRijbewijsAu",    # P – Respondent holds car licence (0 = No, 1 = Yes, 2 = Unknown)
    "OPRijbewijsMo",    # P – Respondent holds motorcycle licence (0 = No, 1 = Yes, 2 = Unknown)
    "OPRijbewijsBr",    # P – Respondent holds moped licence (0 = No, 1 = Yes, 2 = Unknown)
    "HHEFiets",         # P – E-bike present in household (0 = No, 1 = Yes, 2 = Unknown)
    "Kind6",            # V – Child(ren) under 6 travelling along (0 = No, 1 = Yes, 2 = Unknown, <missing> missing if not a displacement or if serial displacement)
    "CorrVerpl",        # V – Trip split into rides by editor (0 = No, 1 = Yes)
    "SDezPlts",         # V – All series addresses in same place (0 = No, 1 = Yes)
    "Toer",             # V – Departure point is arrival point? (0 = No, 1 = Yes, <missing> if not a displacement or if a serial displacement)
    "VergVast", "VergKm", "VergBrSt", "VergOV",                             # P – Type of reimbursement: flat amount / per-km / fuel / public transport (0 No, 1 Yes, 2 N/A)
    "VergAans", "VergVoer", "VergBudg", "VergPark", "VergStal", "VergAnd",  # P – Reimbursement for purchase vehicle / lease vehicle / mobility budget / parking / bicycle storage / other (0 No, 1 Yes, 2 N/A)
    #### -----> Floris: end of what should be 0 or 1


    #### -----> Floris: these should be 1 or 2
    # ── Car-usage source flags ──
    "AutoEig",          # P – Used own car (0 = N/A (person has no displacements), 1 = Yes, 2 = No, 3 = Unknown, 4 = Not asked)
    "AutoHhl",          # P – Used household member’s car (0 = N/A (person has no displacements), 1 = Yes, 2 = No, 3 = Unknown, 4 = Not asked)
    "AutoLWg",          # P – Used employer lease-car (0 = N/A (person has no displacements), 1 = Yes, 2 = No, 3 = Unknown, 4 = Not asked)
    "AutoLPl",          # P – Used private-lease car (0 = N/A (person has no displacements), 1 = Yes, 2 = No, 3 = Unknown, 4 = Not asked)
    "AutoBed",          # P – Used company-registered car (0 = N/A (person has no displacements), 1 = Yes, 2 = No, 3 = Unknown, 4 = Not asked)
    "AutoDOrg",         # P – Used organisation car-share (0 = N/A (person has no displacements), 1 = Yes, 2 = No, 3 = Unknown, 4 = Not asked)
    "AutoDPart",        # P – Used peer-to-peer (online) shared car (0 = N/A (person has no displacements), 1 = Yes, 2 = No, 3 = Unknown, 4 = Not asked)
    "AutoDBek",         # P - Used share car with friends (0 = N/A (person has no displacements), 1 = Yes, 2 = No, 3 = Unknown, 4 = Not asked)
    "AutoLeen",         # P - Used a borrowed car (0 = N/A (person has no displacements), 1 = Yes, 2 = No, 3 = Unknown, 4 = Not asked)
    "AutoHuur",         # P - Used a rental car (0 = N/A (person has no displacements), 1 = Yes, 2 = No, 3 = Unknown, 4 = Not asked)
    "AutoAnd",          # P - Used a different kind of car (0 = N/A (person has no displacements), 1 = Yes, 2 = No, 3 = Unknown, 4 = Not asked)
    "ByzAdr",    # P – Special: other addresses visited? (0 N/A (no displacements), 1 Yes, 2 No, 3 Not asked)
    "ByzVvm",    # P – Special: other modes used? (0 N/A (no displacements), 1 Yes, 2 No, 3 Not asked)
    "ByzTyd",    # P – Special: other times travelled? (0 N/A (no displacements), 1 Yes, 2 No, 3 Not asked)
    "ByzDuur",   # P – Special: other travel durations? (0 N/A (no displacements), 1 Yes, 2 No, 3 Not asked)
    "ByzRoute",  # P – Special: other routes taken? (0 N/A (no displacements), 1 Yes, 2 No, 3 Not asked)
]


# ── ORDINAL (37) ──────────────────────────────────────────────────────────────
ordinal_cols = [

    # Ignore nothing 
    "Sted",         # P – Urbanisation class !!! for municipality !!! (1 Very strongly urban, 2 Strongly urban, 3 Moderately urban, 4 Slightly urban, 5 Not urban)
    "FqLopen",      # P – Walk outdoors (1 Daily, 2 Few/week, 3 Few/month, 4 Few/year, 5 Never)
    "FqNEFiets",    # P – Conventional bicycle (1 Daily, 2 Few/week, 3 Few/month, 4 Few/year, 5 Never)
    "FqEFiets",     # P – E-bike / speed-pedelec (1 Daily, 2 Few/week, 3 Few/month, 4 Few/year, 5 Never)
    "FqBTM",        # P – Bus-tram-metro (1 Daily, 2 Few/week, 3 Few/month, 4 Few/year, 5 Never)
    "FqTrein",      # P – Train (1 Daily, 2 Few/week, 3 Few/month, 4 Few/year, 5 Never)
    "FqAutoB",      # P – Car as driver (1 Daily, 2 Few/week, 3 Few/month, 4 Few/year, 5 Never)
    "FqAutoP",      # P – Car as passenger (1 Daily, 2 Few/week, 3 Few/month, 4 Few/year, 5 Never)
    "FqMotor",      # P – Motorcycle (1 Daily, 2 Few/week, 3 Few/month, 4 Few/year, 5 Never)
    "FqBrSnor",     # P – Moped / light-moped (1 Daily, 2 Few/week, 3 Few/month, 4 Few/year, 5 Never)
    "GemGr",        # P – Municipality size class (1 = < 5 k, 2 = 5–10 k, 3 = 10–20 k, 4 = 20–50 k, 5 = 50–100 k, 6 = 100–150 k, 7 = 150–250 k, 8 = ≥250 k)

    #### ----> Floris: ignore 0 
    "KAfstV",   # V – Trip distance class NL (0 None (Not NL), 1 0.1-0.5 km, 2 0.5-1 km, 3 1-2.5 km, 4 2.5-3.7 km, 5 3.7-5 km, 6 5-7.5 km, 7 7.5-10 km, 8 10-15 km, 9 15-20 km, 10 20-30 km, 11 30-40 km, 12 40-50 km, 13 50-75 km, 14 75-100 km, 15 ≥ 100 km, <missing> if not a displacement or if a serial displacement)
    "KAfstR",   # R – Ride distance class NL (0 None, 1 0.1-0.5 km … 15 ≥ 100 km)
    #### ----> Floris: end of ignoring 0

    #### ----> Floris: ignore 4 and 5 
    "BetWerk",      # P – Paid work (0 None, 1 < 12 h, 2 12–30 h, 3 ≥ 30 h, 4 Unknown, 5 Not asked (<15 y))
    #### ----> Floris: ignore 4 and 5

    #### ----> Floris: ignore 8 and 9
    "KBouwjaarPa1", "KBouwjaarPa2", "KBouwjaarPaL",  # P – Car model-year class (1 ≤ 2010, 2 2011-13, 3 2014-16, 4 2017-19, 5 2020, 6 2021, 7 2022, 8 Unknown, 9 N/A)
    #### ----> Floris: end of ignoring 8 and 9

    #### ----> Floris: ignore >= 6  
    "KGewichtPa1", "KGewichtPa2", "KGewichtPaL",     # P – Car weight class (1 < 951 kg, 2 951-1150, 3 1151-1350, 4 1351-1550, 5 > 1550, 6 Unknown, 7 N/A)
    "BerHalte",  # P – Bus / tram / metro stop: how often reachable? (1 Always, 2 Often, 3 Sometimes, 4 Seldom, 5 Never, 6 Unknown, 7 Not applicable – never needs to go, 8 N/A < 15 y)
    "BerFam",    # P – Family/friends location: how often reachable? (1 Always, 2 Often, 3 Sometimes, 4 Seldom, 5 Never, 6 Unknown, 7 Not applicable – never needs to go, 8 N/A < 15 y)
    "BerSport",  # P – Sport or hobby venue: how often reachable? (1 Always, 2 Often, 3 Sometimes, 4 Seldom, 5 Never, 6 Unknown, 7 N/A (no need), 8 N/A < 15 y)
    "BerWrk",  # P – Workplace: how often reachable? (1 Always, 2 Often, 3 Sometimes, 4 Seldom, 5 Never, 6 Unknown, 7 N/A (no need), 8 N/A < 15 y)
    "BerOnd",  # P – Education location: how often reachable? (1 Always, 2 Often, 3 Sometimes, 4 Seldom, 5 Never, 6 Unknown, 7 N/A (no need), 8 N/A < 15 y)
    "BerSup",  # P – Supermarket: how often reachable? (1 Always, 2 Often, 3 Sometimes, 4 Seldom, 5 Never, 6 Unknown, 7 N/A (no need), 8 N/A < 15 y)
    "BerZiek", # P – Hospital: how often reachable? (1 Always, 2 Often, 3 Sometimes, 4 Seldom, 5 Never, 6 Unknown, 7 N/A (no need), 8 N/A < 15 y)
    "BerArts", # P – GP (doctor): how often reachable? (1 Always, 2 Often, 3 Sometimes, 4 Seldom, 5 Never, 6 Unknown, 7 N/A (no need), 8 N/A < 15 y)
    "BerStat", # P – Railway station: how often reachable? (1 Always, 2 Often, 3 Sometimes, 4 Seldom, 5 Never, 6 Unknown, 7 N/A (no need), 8 N/A < 15 y)
    #### ----> Floris: end of ignoring >=6

    #### ----> Floris: ignore 7
    "SAantAdr",     # V – Addresses visited in series (1 = 3, 2 = 4, 3 = 5, 4 = 6–10, 5 = 11–20, 6 = ≥21, 7 = Unknown)
    #### ----> Floris: end of ignoring 7

    #### ----> Floris: ignore 9
    "HHLaagInk",    # P - Deviation from minimal-income lower bound (1 = Income until 80% of minimal income, 2 = Income from 80-85% of minimal income, ... 7 = Income of 105-110% of minimal income, 8 = Income of 110% or higher of minimal income, 9 = Income unknown)
    #### ----> Floris: end of ignoring 9

    #### ----> Floris: ignore 10
    "HHSocInk",     # P - Deviation from social minimum income (1 = Income until 101% of minimal income, 2 = Income from 101-105% of minimal income, ... 8 = Income of 140-150% of minimal income, 8 = Income of 150% or higher of minimal income, 10 = Income unknown)
    #### ----> Floris: end of ignoring 10

    #### ----> Floris: ignore 11
    "HHBestInkG",   # P - Available income of househould (10% groups) (1 = First 10% ... 10 = Tenth 10%, 11 = Unknown income)
    "HHGestInkG",   # P - Standardized available income of household (10% groups) (1 = First 10% ... 10 = Tenth 10%, 11 = Unknown income)
    "HHWelvG",      # P - Welfare of household (10% groups) (1 = First 10% group ... 10 = Tenth 10% group, 11 = welfare unknown)
    #### ----> Floris: end of ignoring 11


]

# ── CATEGORICAL (82) ───────────────────────────────────────────────────────────
categorical_cols = [
    "Geslacht"              # P - Sexuality (1= Man, 2 = Woman)
    "Herkomst",             # P – Migration background (1 Dutch, 2 Western, 3 Non-Western, 4 Unknown)
    "KLeeft",               # P – Age class (2 = 6–11 y, 3 = 12–14 y, 4 = 15–17 y, 5 = 18–19 y, 6 = 20–24 y, 7 = 25–29 y, 8 = 30–34 y, 9 = 35–39 y, 10 = 40–44 y, 11 = 45–49 y)
    "HHSam",                # P – Household composition (1 Single, 2 Couple, 3 Couple + child(ren), 4 Couple + child(ren) + others, 5 Couple + others, 6 Single-parent + child(ren), 7 Single-parent + child(ren) + others, 8 Other, 9 Unknown)
    "HHPlOP",               # P – Respondent’s position in household (1 Single, 2 Household core, 3 Partner, 4 Child, 5 Other member, 6 Unknown)
    "OnbBez",               # P – Unpaid activity (0 None, 1 Homemaker, 2 Retired, 3 Student, 4 Disabled, 5 Unemployed, 6 Unpaid-worker, 7 Other, 8 Unknown, 9 Not asked (<15y))
    "MaatsPart",            # P – Social-participation class (1 12–30 h work, 2 ≥ 30 h work, 3 Own household, 4 Student, 5 Unemployed, 6 Disabled, 7 Retired, 8 Other, 9 Unknown)
    "Opleiding",            # P – Highest completed education (0 None, 1 Primary, 2 Lower vocational, 3 Upper sec / VET, 4 Higher ed/ university, 5 Other, 6 Unknown, 7 Not asked (<15y))
    "HHBezitVm",            # P – Household vehicle possession (0 None, 1 ≥ 3 cars, 2 2 cars, 3 1 car, 4 ≥ 1 motor, 5 ≥ 1 moped, 6 ≥ 1 light-moped, 7 ≥ 1 e-bike, 8 Unknown)
    "OPBezitVm",            # P – Respondent’s vehicle possession (0 None, 1 ≥ 3 cars, 2 2 cars, 3 1 car, 4 ≥ 1 motor, 5 ≥ 1 moped, 6 ≥ 1 light-moped, 7 Unknown)
    "OVStKaart",            # P – Holds student OV-chip card (0 No, 1 Week, 2 Weekend, 3 Unknown, 4 N/A < 15 y or > 40 y)
    "Verpl",                # V – Displacement flag (<missing> missing if not a displacment, 0 No, 1 New, 6 Series, 7 Work-truck, 8 Series work-truck)
    "Doel",                 # V – Displacement destination purpose (1 Home, 2 Work, 3 Business visit, 4 Occupational, 5 Pick-up/bring persons, 6 Pick-up/bring goods, 7 Education, 8 Shopping, 9 Visit, 10 Touring/walk, 11 Sport/hobby, 12 Other leisure, 13 Personal services, 14 Other, <missing> if not a displacement or if a serial displacement)
    "MotiefV",              # V – Displacement motive (1 Commute, 2 Business visit, 3 Occupational, 4 Pick-up persons, 5 Goods, 6 Education, 7 Shopping, 8 Visit, 9 Touring, 10 Sport/hobby, 11 Other leisure, 12 Personal services, 13 Other, <missing> if not a displacement or if a serial displacement)
    "KMotiefV",             # V – Class division displacement motive group (1 Commute, 2 Business, 3 Services/personal, 4 Shopping, 5 Education, 6 Visit, 7 Other social/leisure, 8 Touring/walking, 9 Other, <missing> if not a displacement or if a serial displacement)
    "VertLoc",              # V – Departure location type (1 Home, 2 Other home, 3 Work, 4 Other, <missing> if not a displacement)
    "AardWerk",             # V – Nature of work (1 Construction, 2 Service, 3 Delivery, 4 Goods-transport, 5 Passenger-transport, 6 Care, 7 Emergency, 8 Business, 9 Collection, 10 Other, 11 Unknown, 12 N/A, <missing> missing if not a displacement)
    "Weggeweest",           # P – Was away yesterday? (0 No, 1 Yes, 6 Series trip, 7 Work truck trip, 8 Work-truck series)
    "VolgWerk",             # V – Sequence of work trips (1 Stand-alone work trip, 2 1st of 2, 3 2nd of 2, 4 1st of 3, 5 2nd of 3, 6 3rd of 3, 7 1st of series-followed trips, 8 Series work trip, 9 N/A – not work)
    "KVertTijd",      # V – Departure-time class (1 00:00-04:00, 2 04:00-07:00, 3 07:00-08:00, 4 08:00-09:00, 5 09:00-12:00, 6 12:00-13:00, 7 13:00-14:00, 8 14:00-16:00, 9 16:00-17:00, 10 17:00-18:00, 11 18:00-19:00, 12 19:00-20:00, 13 20:00-24:00)
    "Prov",                 # P – Province of residence (1 Groningen … 12 Limburg)
]

"Rvm" # Is target variable, so we ignore it
# Dropped cols (24)
drop_cols = [

    # Confounds with target variable:
    "Hvm",            # R/V – Detailed ride / main trip mode (1 Car, 2 Train, … 24 Other-no-motor)
    "KHvm",                 # V – Trip mode class (1 Car-driver, 2 Car-passenger, 3 Train, 4 Bus/Tram/Metro, 5 Bicycle, 6 Walk, 7 Other, <missing> missing if not a displacement)
    "EFiets",               # P – Type of e-bike used (0 N/A, 1 E-bike, 2 Speed-pedelec, 3 Both, 4 Unknown, 5 Not asked/no e-bike)
    "WrkVervw",             # P – Transport mode to work with most km's (1 Walk, 2 Bicycle/e-bike, 3 Moped, 4 Car, 5 Van, 6 Motorcycle, 7 Train, 8 Bus/Tram/Metro, 9 Other, 10 Unknown, 11 N/A works from home, 12 N/A No paid work, 13 N/A < 15 y)
    "HvmRol", "RvmRol",     # R/V – Role in mode (1 Driver, 2 Passenger, 3 Unknown, 4 N/A)
    "Reisduur",     # V – Trip travel-time NL (min) (0 Not a regular displacement in NL, 1 = 1 to 5 minutes, 2 = 5 to 10 minutes ... 10 = 90 to 120 minutes, 11 = 120 minutes or more)
    "KReisduur",# V – Travel-time class NL (0 None, 1 1-5 min, 2 5-10, 3 10-15, 4 15-20, 5 20-25, 6 25-30, 7 30-45, 8 45-60, 9 60-90, 10 90-120, 11 ≥ 120)
    #### ----> Floris: ignore 12 and 13
    "RTSamen",      # R – Train party size (1‥8, 9 = 9-12, 10 = 12-20, 11 = ≥20, 12 = Unknown, 13 = N/A)
    #### ----> Floris: end of ignoring 12 and 13
    "RAantIn",        # R – Occupants in car (1‥8, 9 = ≥9, 10 Unknown, 11 N/A)  
    # 
    #     # Removed as they cannot be generalised:
    "RdHalteA",             # P – Reason A stop not always reachable (1 No own transport, 2 Cannot/will-not cycle, 3 Cannot/will-not use Public Transport, 4 Cannot/will-not use taxi, 5 Cannot travel alone, 6 Health, 7 Journey too long, 8 Too expensive, 9 Traffic too busy, 10 Feels unsafe, 11 Other, 12 Unknown, 13 N/A (always reachable), 14 N/A (other reason))
    "RdHalteB",             # P – Reason B stop not always reachable (same code list; 12 = No second reason)
    "RdFamA", "RdFamB",     # P – Reasons A/B family/friends not always reachable (1 No own transport, 2 Cannot/will-not cycle, 3 Cannot/will-not use Public Transport, 4 Cannot/will-not use taxi, 5 Cannot travel alone, 6 Health, 7 Journey too long, 8 Too expensive, 9 Traffic too busy, 10 Feels unsafe, 11 Other, 12 Unknown, 13 N/A (always reachable), 14 N/A (other reason))
    "RdSportA", "RdSportB", # P – Reasons A/B sport venue not always reachable (1 No own transport, 2 Cannot/will-not cycle, 3 Cannot/will-not use Public Transport, 4 Cannot/will-not use taxi, 5 Cannot travel alone, 6 Health, 7 Journey too long, 8 Too expensive, 9 Traffic too busy, 10 Feels unsafe, 11 Other, 12 Unknown, 13 N/A (always reachable), 14 N/A (other reason))
    "RdWrkA", "RdWrkB",     # P – Reasons A/B workplace not always reachable (1 No own transport, 2 Cannot/will-not cycle, 3 Cannot/will-not use Public Transport, 4 Cannot/will-not use taxi, 5 Cannot travel alone, 6 Health, 7 Journey too long, 8 Too expensive, 9 Traffic too busy, 10 Feels unsafe, 11 Other, 12 Unknown, 13 N/A (always reachable), 14 N/A (other reason))
    "RdOndA", "RdOndB",     # P – Reasons A/B education location not always reachable (1 No own transport, 2 Cannot/will-not cycle, 3 Cannot/will-not use Public Transport, 4 Cannot/will-not use taxi, 5 Cannot travel alone, 6 Health, 7 Journey too long, 8 Too expensive, 9 Traffic too busy, 10 Feels unsafe, 11 Other, 12 Unknown, 13 N/A (always reachable), 14 N/A (other reason))
    "RdSupA", "RdSupB",     # P – Reasons A/B supermarket not always reachable (1 No own transport, 2 Cannot/will-not cycle, 3 Cannot/will-not use Public Transport, 4 Cannot/will-not use taxi, 5 Cannot travel alone, 6 Health, 7 Journey too long, 8 Too expensive, 9 Traffic too busy, 10 Feels unsafe, 11 Other, 12 Unknown, 13 N/A (always reachable), 14 N/A (other reason))
    "RdZiekA", "RdZiekB",   # P – Reasons A/B hospital not always reachable (1 No own transport, 2 Cannot/will-not cycle, 3 Cannot/will-not use Public Transport, 4 Cannot/will-not use taxi, 5 Cannot travel alone, 6 Health, 7 Journey too long, 8 Too expensive, 9 Traffic too busy, 10 Feels unsafe, 11 Other, 12 Unknown, 13 N/A (always reachable), 14 N/A (other reason))
    "RdArtsA", "RdArtsB",   # P – Reasons A/B GP not always reachable (1 No own transport, 2 Cannot/will-not cycle, 3 Cannot/will-not use Public Transport, 4 Cannot/will-not use taxi, 5 Cannot travel alone, 6 Health, 7 Journey too long, 8 Too expensive, 9 Traffic too busy, 10 Feels unsafe, 11 Other, 12 Unknown, 13 N/A (always reachable), 14 N/A (other reason))
    "RdStatA", "RdStatB",   # P – Reasons A/B station not always reachable (1 No own transport, 2 Cannot/will-not cycle, 3 Cannot/will-not use Public Transport, 4 Cannot/will-not use taxi, 5 Cannot travel alone, 6 Health, 7 Journey too long, 8 Too expensive, 9 Traffic too busy, 10 Feels unsafe, 11 Other, 12 Unknown, 13 N/A (always reachable), 14 N/A (other reason))

    # Removed to make the dataset smaller:
    "AankPC",         # V – Arrival postcode NL (1000‥9999; 0 abroad; 0000 unk., <empty> empty in case of no displacement)
    "VertPC",         # V – Departure postcode NL (1000‥9999; 0 abroad; 0000 unk., <empty> empty in case of no displacement)
    "VertGem", "AankGem",   # V – Municipality code (14–1991 list, 9999 Unknown, <missing> if not a displacement)
    "RedenNWZ",             # P – Illness duration (1 1–6 days, 2 7 days–4 weeks, 3 > 4 weeks, 4 Unknown, 0 N/A, 5 N/A other reason)
    "ByzDag",               # P – Diary day had special circumstances? (0 N/A (no displacements), 1 Yes, 2 No specific events, 3 No – this weekday is always different, 4 Unknown)
    "ByzReden",             # P – Reason different travel pattern (0 N/A (no displacements), 1 Day off, 2 Illness, 3 Traffic, 4 Appointments, 5 Ill household-member, 6 Travel together, 7 Luggage, 8 Weather, 9 Other, 10 Unknown, 11 Not asked)
    "Rit",                  # R – Ride flag (1 New ride, 3 Foreign ride, 7 Work-truck ride) (type of ride)
    "VertProv","AankProv",  # V – Province code (0 Abroad/none, 1–12 NL, 99 Unknown, <missing> if not a displacement)
    "WoPC",           # P – Home postal-code (1000‥9999 = Dutch PC)
    "KRvm",                 # R – Ride mode class (1 Car-driver, 2 Car-passenger, 3 Train, 4 Bus/Tram/Metro, 5 Bicycle, 6 Walk, 7 Other, <missing> missing if not a displacement)
    "BrandstofPa1",         # P – Primary fuel youngest car (1 Petrol, 2 Diesel, 3 LPG, 4 Electric, 5 Other, 6 Unknown, 7 N/A)
    "XBrandstofPa1",        # P – Secondary fuel youngest car (0 None, 1 Petrol, 2 Diesel, 3 LPG, 4 Electric, 5 Other, 6 Unknown, 7 N/A)
    "BrandstofEPa1",        # P – Electric-drive type youngest car (0 Not electric, 1 Full-EV, 2 Plug-in hybrid, 3 Hybrid charge-in-use, 4 Other, 5 Unknown, 6 N/A)
    "TenaamPa1",            # P – Registered owner youngest car (1 Own name, 2 Other household member, 3 N/A)
    "BrandstofPa2",         # P – Primary fuel 2nd car (1 Petrol … 6 Unknown, 7 N/A)
    "XBrandstofPa2",        # P – Secondary fuel 2nd car (0 None … 6 Unknown, 7 N/A)
    "BrandstofEPa2",        # P – Electric drive type 2nd car (0 Not electric … 6 N/A)
    "TenaamPa2",            # P – Registered owner 2nd car (1 Own, 2 Other household member, 3 N/A)
    "BrandstofPaL",         # P – Primary fuel lease / company car (1 Petrol … 6 Unknown, 7 N/A)
    "XBrandstofPaL",        # P – Secondary fuel lease / company car (0 None … 6 Unknown, 7 N/A)
    "BrandstofEPaL",        # P – Electric drive type lease / company car (0 Not electric … 6 N/A)
    "RVliegVer",            # R – Flight leg removed (0 No, 1 Before, 2 After, 3 Both)
    "VertGeb", "AankGeb",   # V – Country/area code (0 NL, 1–99 foreign area list, <missing> if not a displacement)
    "MRA",                  # P – Amsterdam metro-region zone (1 Centre, 2 North, 3 East … 21 Rest NL)
    "MRDH",                 # P – Rotterdam/The Hague metro-region zone (1 The Hague-Centre … 18 Rest NL)
    "Utr",                  # P – Utrecht provincial sub-region (1 De Ronde Venen & Stichtse Vecht … 21 Rest NL)
    "AankMRA",              # V – Arrival in MRA (0 = No, 1..21 are areas in Amsterdam, 99 = Unknown, <missing> missing if not a displacement)
    "AankMRDH",             # V – Arrival in MRDH (0 = No, 1..18 are areas in Rotterdam Den Haag, 99 = Unknown, <missing> missing if not a displacement)
    "AankUtr",              # V – Arrival in Utrecht region (0 = No, 1..21 are areas in Utrecht Region, 99 = Unknown, <missing> missing if not a displacement)
    "VertMRA",              # V – Departure in MRA (0 = No, 1..21 are areas in Amsterdam, 99 = Unknown, <missing> missing if not a displacement)
    "VertMRDH",             # V – Departure in MRDH (0 = No, 1..18 are areas in Rotterdam Den Haag, 99 = Unknown, <missing> missing if not a displacement)
    "VertUtr",              # V – Departure in Utrecht region (0 = No, 1..21 are areas in Utrecht Region, 99 = Unknown, <missing> missing if not a displacement)
    "RedenNW",              # P – Reason for not travelling (0 N/A (person has displacements), 1 Illness/injury, 2 Physical limitation/handicap, 3 Weather, 4 Working at home, 5 Home study, 6 Care duties, 7 No out-of-home activity, 8 Transport was too expensive, 9 No fitting transport, 10 Stay abroad, 11 Other, 12 Unknown)
    "RedenNWW",             # P – Reason for not travelling: Type of bad weather (1 Cold, 2 Heat, 3 Precipitation, 4 Wind/storm, 5 Ice, 6 Fog, 7 Changeable, 8 Other, 9 Unknown, 0 N/A, 10 N/A other reason (person has displacements))
    "RedenNWB",             # P – Reason for not travelling: Purpose of stay abroad (1 Leisure/holiday, 2 Work, 3 Study, 4 Family visit, 5 Other, 6 Unknown, 0 N/A, 7 N/A other reason (person has displacements))
    "SVvm1","SVvm2","SVvm3","SVvm4",  # V – 1st–4th mode in series (1 Car, 2 Train, … 24 Other-no-motor, 25 N/A (except for SVvm1))
    "VertCorop","AankCorop",# V – COROP region code (0 Abroad/none, 1–40 list, 99 Unknown, <missing> if not a displacement)
    
    # Removed for other reasons:
    "OP",           # P – New-person row flag (0 = No new person, 1 = New person)
    "OPID",         # P – Unique ID for each respondent (person key)
    "Steekproef",   # P – Sample indicator (1 = Core survey, 4 = Extra North-Wing, 6 = Extra Rotterdam-The Hague, 8 = Extra Utrecht)
    "Mode",         # P – Response mode (1 = CAWI – web; other modes not used in 2022)
    "Corop",        # P – COROP region of residence (1 = East Groningen, 2 = Rest Groningen, … 40 = Flevoland)
    "BuurtAdam",    # P – Amsterdam neighbourhood combo (0 = Not Amsterdam resident, 036300–036399 = 100+ neighbourhood codes)
    "Jaar",         # P – Reporting year (2022)
    "Maand",        # P – Reporting month (1 = Jan … 12 = Dec)
    "Week",         # P – ISO week number of diary day (1‥53)
    "Dag",          # P – Calendar day of the month (1‥31)
    "Weekdag",      # P – Day of week (1 = Sunday, 2 = Monday, 3 = Tuesday, 4 = Wednesday, 5 = Thursday, 6 = Friday, 7 = Saturday)
    "Feestdag",     # P – Diary day is Dutch public holiday (0 = No, 1 = Yes)
    "PCG",          # V – Dutch border-crossing postcode
    "PCBLG",        # V – Foreign border-crossing postcode
    "GemG",         # V - Dutch border crossing municipality
    "SPlaats1","SPlaats2","SPlaats3","SPlaats4","SPlaats5",  # V – Place name 1–5 (string or <none responded>)
    "AfstSBL",      # V – Series-trip distance abroad (hectometres)
    "RVliegVer",    # R – Flight leg removed (0 No, 1 Before, 2 After, 3 Both)
    "RitNr",        # R – Ride sequence in trip (1‥15) (number of the trip filled in)
    "RitID",        # R – Unique ride ID
    "VertUur",        # V – Departure hour (0‥23)
    "VertMin",        # V – Departure minute (0‥59)
    "AankUur",        # V – Arrival hour  (0‥47)
    "AankMin",        # V – Arrival minute (0‥59)
    "SBegUur",        # V – Serie displacement start hour (0‥23)
    "SBegMin",        # V – Serie displacement start minute (0‥59)
    "SEindUur",       # V – Serie displacement end hour (0‥47, 99 = Unknown)
    "SEindMin",       # V – Serie displacement end minute (0‥59, 99 = Unknown)
    "VerplID",        # V – Unique displacement ID (<missing> if not a displacement, id)
    "RVertUur",       # R – Ride departure hour (0‥47)
    "RVertMin",       # R – Ride departure minute (0‥59)
    "RAankUur",       # R – Ride arrival hour  (0‥47, 99 Unknown)
    "RAankMin",       # R – Ride arrival minute (0‥59, 99 Unknown)
    "RVertStat",      # R – Departure rail-station code (000 N/A)
    "RAankStat",      # R – Arrival rail-station code   (000 N/A)
    "Wogem",          # P - Municipality code (14...1991 = Dutch municipality code)
    "VerplNr",        # V – Displacement sequence number (<missing> if not a displacement, 1‥25)
    "VertPCBL",       # V – Departure postcode abroad ((1000..999) Departure postcode Belgium, (01000..99999) departure postcode Germany, 0000 unknown Belgium, 00000 unknown Germany, 1 Departure in NL, 0 No departure point in Belgium or Germany, <empty> empty if not a displacement)
    "AankPCBL",       # V – Arrival postcode abroad ((1000..999) Departure postcode Belgium, (01000..99999) departure postcode Germany, 0000 unknown Belgium, 00000 unknown Germany, 1 Departure in NL, 0 No departure point in Belgium or Germany, <empty> empty if not a displacement)
    "RCorrSnelh",     # R – Speed-correction flag (0 No, 1 Dist↓, 2 Time↑, 3 Dist↓+Time↑, 4 Dist↑, 5 Time↓, 6 Dist↑+Time↓)
    "GehBLVer",       # V – Entirely-abroad trip removed (0 No, 1 Removed before, 2 Removed after, 3 removed before and removed after)
    "AantRit",        # V – Number of trips in displacement (<missing> if not a displacement or if a serial displacement, 1‥15)
    "ReisduurOP",     # P – Total travel-time regular displacements (min) (0 = no displacements, 1-999999)
    "RReisduur",      # R – Trip travel-time NL (min) (0 = not a trip in NL, 1-9999 = time, <missing> missing if not a displacement)
    "RReisduurBL",    # R – Trip travel-time abroad (min) (0 = not a trip abroad, 1-9999 = time, <missing> missing if not a displacement)
]


if __name__ == "__main__":
    print(len(numerical_cols + categorical_cols + ordinal_cols +drop_cols))