import os
import numpy as np
import pandas as pd

from codebase.data.load_demographics import load_excel


def make_ml_dataset(df: pd.DataFrame, target_col, drop_cols, categorical_cols=None, target_vals=None, test_size=0.2, random_state=42, stratification_col=None, group_col=None) -> tuple:
    """
    Splits the dataset into training and testing sets.
    """

    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GroupShuffleSplit

    if stratification_col is not None and group_col is not None:
        raise ValueError("Cannot use both stratification_col and group_col for splitting.")

    # Drop specified columns
    df_: pd.DataFrame = df.drop(columns=drop_cols)

    # Split the data into features and target
    X = df_.drop(columns=[target_col])
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True) if categorical_cols else X
    y = df_[target_col].isin(target_vals) if target_vals is not None else df_[target_col]
    stratification = df_[stratification_col] if stratification_col else None

    # Split the data into training and testing sets
    if group_col:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(X, y, groups=df_[group_col]))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratification)

    # 1. Find the set of labels that occur in both y_train and y_test
    common_labels = np.intersect1d(np.unique(y_train), np.unique(y_test))

    print(f"Common labels: {common_labels}")

    # 2. Create boolean masks
    train_mask = y_train.isin(common_labels)
    test_mask = y_test.isin(common_labels)

    # 3. Filter both sets
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]

    return X_train, X_test, y_train, y_test

def load_odin_as_ml_dataset(
        year=2022,
        target_col="WrkVervw",
        target_val=2,
        validation_proportion=None,
        random_state=42,
):
    odin_excel_path = os.path.join(os.getcwd(), "data", "OdiN 2019-2023", f"OdiN {year}", f"ODiN{year}_Databestand.xlsx")
    df = load_excel(odin_excel_path)

    # ── NUMERICAL (68) ────────────────────────────────────────────────────────────
    numerical_cols = [
        "HHPers",      # P – Number of persons in household (1‥9, 10 = 10 or more, 11 = Unknown)
        "HHLft1",      # P – Household members < 6 yrs (0‥9, 10 = 10+, 11 = Unknown)
        "HHLft2",      # P – Household members 6-11 yrs (0‥9, 10 = 10+, 11 = Unknown)
        "HHLft3",      # P – Household members 12-17 yrs (0‥9, 10 = 10+, 11 = Unknown)
        "HHLft4",      # P – Household members ≥ 18 yrs (0‥9, 10 = 10+, 11 = Unknown)
        "WoPC",        # P – Home postal-code (1000‥9999 = Dutch PC)
        "Leeftijd",    # P – Age of respondent (6‥98, 99 = 99 or older)
        "HHRijbewijsAu", # P – Car licences in household (0‥8, 9 = ≥9, 10 = Unknown)
        "HHRijbewijsMo", # P – Motorcycle licences in household (0‥8, 9 = ≥9, 10 = Unknown)
        "HHRijbewijsBr", # P – Moped licences in household (0‥8, 9 = ≥9, 10 = Unknown)
        "HHAuto",      # P – Passenger cars in household (0‥8, 9 = ≥9, 10 = Unknown)
        "HHAutoL",     # P – Leased / company cars in household (0‥8, 9 = ≥9, 10 = Unknown)
        "OPAuto",      # P – Cars registered to respondent (0‥8, 9 = ≥9, 10 = Unknown)
        "BouwjaarPa1", # P – Model-year newest car (1885‥2022, 9994 = Unknown, 9995 = N/A)
        "BouwjaarPa2", # P – Model-year 2nd car        (same coding as above)
        "BouwjaarPaL", # P – Model-year lease/company car (same coding)
        "HHMotor",     # P – Motorcycles in household (0‥8, 9 = ≥9, 10 = Unknown)
        "OPMotor",     # P – Motorcycles on respondent’s name (0‥8, 9 = ≥9, 10 = Unknown)
        "HHBrom",      # P – Mopeds in household (0‥8, 9 = ≥9, 10 = Unknown)
        "OPBrom",      # P – Mopeds on respondent’s name (0‥8, 9 = ≥9, 10 = Unknown)
        "HHSnor",      # P – Light-mopeds in household (0‥8, 9 = ≥9, 10 = Unknown)
        "OPSnor",      # P – Light-mopeds on respondent’s name (0‥8, 9 = ≥9, 10 = Unknown)
        "ReisduurOP",   # P – Total travel-time of regular trips (0 = none, 1‥9 999 min)
        "AfstandOP",    # P – Total distance of regular trips (0 = none, 1‥9 999 999 hm)
        "AfstandSOP",   # P – Total distance of series trips (0 = none, 1‥9 999 999 hm)
        "Verpl",        # V – Trip flag (0 = No, 1 = New trip, 6 = Series trip, 7–8 = Work truck)
        "VerplID",      # V – Unique trip ID (integer key)
        "VerplNr",      # V – Trip sequence number (1‥25)
        "AantRit",      # V – Rides (segments) inside trip (1‥15)
        "VertPC",       # V – Departure postcode NL (1000‥9999; 0 = abroad; 0000 = unknown)
        "VertPCBL",     # V – Departure postcode BE/DE (foreign: 1000‥9999 / 01000‥99999)
        "AankPC",       # V – Arrival postcode NL (same coding as VertPC)
        "AankPCBL",     # V – Arrival postcode BE/DE (same coding as VertPCBL)
        "PCG",          # V – Dutch border-crossing postcode (0 = none, 1000‥9999, 0000 unknown)
        "PCBLG",        # V – Foreign border-crossing postcode (0 = none, 1000‥9999 BE, 01000‥99999 DE)
        "AfstV",        # V – Trip distance NL (0 = none, 1‥99 999 hm)
        "VertUur", "VertMin",       # V – Departure clock-time (hh 0‥23, mm 0‥59)
        "AankUur", "AankMin",       # V – Arrival clock-time (hh 0‥47, mm 0‥59)
        "Reisduur",     # V – Trip travel-time NL (0 = none, 1‥9 999 min)
        "ActDuur",      # V – Activity duration at destination (0‥9 999 min)
        "AfstS", "AfstSBL",         # V – Series-trip distance NL / abroad (1‥99 999 hm)
        "SBegUur", "SBegMin",       # V – Series start-time (hh 0‥23, mm 0‥59)
        "SEindUur", "SEindMin",     # V – Series end-time (hh 0‥47/99, mm 0‥59/99)
        "SAantAdr",      # V – Addresses visited in a series (1 = 3, 2 = 4, 3 = 5, 4 = 6–10, 5 = 11–20, 6 = ≥21, 7 = Unknown)
        "Rit", "RitID", "RitNr",    # R – Ride flag / unique ID / sequence (see codes)
        "AfstR", "AfstRBL",         # R – Ride distance NL / abroad (hectometers)
        "RAantIn",       # R – Occupants in car (1‥8, 9 = ≥9, 10 = Unknown, 11 = N/A)
        "RVertUur", "RVertMin",     # R – Ride departure time (hh 0‥47, mm 0‥59)
        "RAankUur", "RAankMin",     # R – Ride arrival time  (hh 0‥47/99, mm 0‥59/99)
        "RReisduur", "RReisduurBL", # R – Ride travel-time NL / abroad (0 = none, 1‥9 999 min)
        "RVertStat", "RAankStat",   # R – Rail station codes (000 = N/A, 001‥997 list, 998 = Other, 999 = Unknown)
        "RTSamen",       # R – Train party size (1‥8, 9 = 9-12, 10 = 12-20, 11 = ≥20, 12 = Unknown, 13 = N/A)
        "RCorrSnelh",    # R – Speed correction flag (0 = No, 1 = Dist↓, 2 = Time↑, 3 = Dist↓+Time↑, 4 = Dist↑, 5 = Time↓, 6 = Dist↑+Time↓)
        "RVliegVer",     # R – Flight leg removed (0 = No, 1 = Before, 2 = After, 3 = Both)
        "FactorH", "FactorP", "FactorV",  # W – Weight factors (floating-point)
    ]
    # 22 numerical variables

    # ── BINARY (12) ────────────────────────────────────────────────────────────────
    binary_cols = [
        "OPRijbewijsAu", # P – Respondent holds car licence (0 = No, 1 = Yes, 2 = Unknown)
        "OPRijbewijsMo", # P – Respondent holds motorcycle licence (0 = No, 1 = Yes, 2 = Unknown)
        "OPRijbewijsBr", # P – Respondent holds moped licence (0 = No, 1 = Yes, 2 = Unknown)
        "HHEFiets",      # P – E-bike present in household (0 = No, 1 = Yes, 2 = Unknown)
        "Kind6",        # V – Child(ren) under 6 travelling along (0 = No, 1 = Yes, 2 = Unknown)
        "CorrVerpl",    # V – Trip split into rides by editor (0 = No, 1 = Yes)
        "SDezPlts",     # V – All series addresses in same place (0 = No, 1 = Yes)
        "Toer",         # V – Round-trip flag (0 = No, 1 = Yes)
        "VertMRA", "VertMRDH", "VertUtr",  # V – Departure in MRA/MRDH/Utrecht? (0 = No, 1 = Zone code > 0)
        "AankMRA", "AankMRDH", "AankUtr",  # V – Arrival in same metro regions (same coding)
    ]

    # ───────── ORDINAL COLUMNS (90) ───────────────────────────
    # ── ORDINAL (128) ──────────────────────────────────────────────────────────────
    ordinal_cols = [
        "HHSam",        # P – Household composition (1 = Single, 2 = Couple, 3 = Couple + children, 4 = Couple + children + others, 5 = Couple + others, 6 = Single-parent + children, 7 = Single-parent + children + others, 8 = Other, 9 = Unknown)
        "HHPlOP",       # P – Respondent’s position in household (1 = Single, 2 = Household core, 3 = Partner, 4 = Child, 5 = Other member, 6 = Unknown)
        "Sted",         # P – Urbanisation class (1 = Very strong, 2 = Strong, 3 = Moderate, 4 = Slight, 5 = Rural)
        "GemGr",        # P – Municipality size class (1 = <5 k, 2 = 5–10 k, 3 = 10–20 k, 4 = 20–50 k, 5 = 50–100 k, 6 = 100–150 k, 7 = 150–250 k, 8 = ≥250 k)
        "Prov",         # P – Province (1 = Groningen, 2 = Friesland, 3 = Drenthe, 4 = Overijssel, 5 = Flevoland, 6 = Gelderland, 7 = Utrecht, 8 = North Holland, 9 = South Holland)
        "MRA",          # P – Amsterdam metro-region zone (1 = Centre, 2 = North, 3 = West, 4 = New-West, 5 = South, 6 = East, 7 = South-East, 8 = Waterland, 9 = Zaanstreek, 10 = IJmond)
        "MRDH",         # P – Rotterdam/The Hague metro zone (1 = The Hague Centre, 2 = South-West, 3 = North-West, 4 = East, 5 = South-West region, 6 = South region)
        "Utr",          # P – Utrecht provincial zone (1 = De Ronde Venen etc, … 21 = Rest Netherlands)
        "Herkomst",     # P – Migration background (1 = Dutch, 2 = Western, 3 = Non-Western, 4 = Unknown)
        "BetWerk",      # P – Paid work (0 = None, 1 < 12h, 2 = 12–30h, 3 ≥ 30h, 4 = Unknown, 5 = Not asked (<15 yr))
        "OnbBez",       # P – Unpaid activity (1 = Homemaker, 2 = Retired, 3 = Student, 4 = Disabled, 5 = Unemployed, 6 = Unpaid worker, 7 = Other, 8 = Unknown)
        "MaatsPart",    # P – Social participation class (1 = 12–30 h work, 2 ≥ 30 h, 3 = Own household, 4 = Student, 5 = Unemployed, 6 = Disabled, 7 = Retired, 8 = Other, 9 = Unknown)
        "Opleiding",    # P – Highest completed education (0 = None, 1 = Primary, 2 = Lower vocational, 3 = Upper sec/VET, 4 = Higher ed, 5 = Other, 6 = Unknown)
        "HHBestInkG",   # P – Household disposable-income decile (1 = Lowest 10 %, … 10 = Highest 10 %, 11 = Unknown)
        "HHGestInkG",   # P – Standardised household income decile (same scale as above)
        "HHLaagInk",    # P – Deviation from low-income threshold (1 ≤ 80 %, … 8 ≥ 110 %, 9 = Unknown)
        "HHSocInk",     # P – Deviation from social-minimum (1 ≤ 101 %, … 9 ≥ 150 %, 10 = Unknown)
        "HHWelvG",      # P – Household wealth decile (1 = Lowest 10 %, … 10 = Highest 10 %, 11 = Unknown)
        "KBouwjaarPa1", "KBouwjaarPa2", "KBouwjaarPaL",  # P – Model-year class car (1 ≤ 2010, … 7 = 2022, 8 = Unknown, 9 = N/A)
        "KGewichtPa1", "KGewichtPa2", "KGewichtPaL",     # P – Weight class car (1 < 951 kg, … 5 > 1550 kg, 6 = Unknown, 7 = N/A)
        "HHBezitVm",    # P – Household vehicle-ownership class (0 = None, 1 = ≥3 cars, 2 = 2 cars, 3 = 1 car, 4 = ≥1 motorcycle, 5 = ≥1 moped, 6 = ≥1 light-moped, 7 = ≥1 e-bike, 8 = Unknown)
        "OPBezitVm",    # P – Respondent’s vehicle-ownership class (same coding as above)
        "FqLopen", "FqNEFiets", "FqEFiets", "FqBTM", "FqTrein",
        "FqAutoB", "FqAutoP", "FqMotor", "FqBrSnor",       # P – Travel frequency (1 = Daily, 2 = Few/week, 3 = Few/month, 4 = Few/year, 5 = Never)
        "OVStKaart",    # P – Holds student OV-chip card (0 = No, 1 = Week, 2 = Weekend, 3 = Unknown, 4 = N/A)
        "WrkVervw",     # P – Main commute mode by km (1 = Walk, 2 = Bike/e-bike, 3 = Moped, 4 = Car, 5 = Van, 6 = Motorcycle, 7 = Train, 8 = Bus/Tram/Metro, 9 = Other, 10 = Unknown)
        "WrkVerg", "VergVast", "VergKm", "VergBrSt", "VergOV",
        "VergAans", "VergVoer", "VergBudg", "VergPark", "VergStal", "VergAnd",   # P – Employer reimbursements (0 = No, 1 = Yes, 2 = N/A)
        "BerWrk", "RdWrkA", "RdWrkB",
        "BerOnd", "RdOndA", "RdOndB",
        "BerSup", "RdSupA", "RdSupB",
        "BerZiek", "RdZiekA", "RdZiekB",
        "BerArts", "RdArtsA", "RdArtsB",
        "BerStat", "RdStatA", "RdStatB",
        "BerHalte", "RdHalteA", "RdHalteB",
        "BerFam", "RdFamA", "RdFamB",
        "BerSport", "RdSportA", "RdSportB",
        "Weggeweest",   # P – Was away yesterday? (0 = No, 1 = Yes, 6 = Series trip, 7 = Work truck, 8 = Work truck series)
        "RedenNW", "RedenNWZ", "RedenNWW", "RedenNWB",   # P – Reasons for not travelling (see full codes)
        "AantVpl", "AantOVVpl", "AantSVpl",              # P – Trip counts (0 = None, 1‥25)
        "EFiets",        # P – Type of e-bike used (0 = N/A, 1 = E-bike, 2 = Speed-pedelec, 3 = Both, 4 = Unknown)
        "AutoEig", "AutoHhl", "AutoLWg", "AutoLPl", "AutoBed",
        "AutoDOrg", "AutoDPart",                         # P – Car source flags (0 = N/A, 1 = Yes, 2 = No, 3 = Unknown, 4 = Not asked)
        "KAfstV",      # V – Trip distance class NL (0 = None, 1 = 0.1-0.5 km … 15 = ≥100 km)
        "KVertTijd",   # V – Departure-time class (1 = 0-4 h, 2 = 4-7 h, … 13 = 20-24 h)
        "KReisduur",   # V – Travel-time class NL (0 = None, 1 = 1-5 min, … 11 = ≥120 min)
        "VolgWerk",    # V – Sequence of work trips (1 = Stand-alone, 2 = 1st-of-2, 3 = 2nd-of-2, … 9 = Not work)
        "KMotiefV",    # V – Motive group (1 = Commute, 2 = Business, 3 = Services, 4 = Shopping, … 9 = Other)
        "MeerWink",    # V – Multiple shops visited (0 = No, 1 = Yes, 2 = Unknown, 3 = N/A)
        "AardWerk",    # V – Kind of work (1 = Construction, 2 = Service, 3 = Delivery, 4 = Goods transport, 5 = Passenger transport, 6 = Care, 7 = Emergency, 8 = Business, 9 = Collection, 10 = Other, 11 = Unknown, 12 = N/A)
        "KAfstR",      # R – Ride distance class NL (0 = None, 1 = 0.1-0.5 km … 15 = ≥100 km)
        "KRvm",        # R – Ride mode class (1 = Car-driver, 2 = Car-passenger, 3 = Train, 4 = BTM, 5 = Bicycle, 6 = Walk, 7 = Other)
        "KHvm",        # V – Trip mode class (same coding as KRvm)
        # — Reachability & reason scales already partly listed; add missing twins —
        "BerHalte", "RdHalteA", "RdHalteB",
        "BerFam",  "RdFamA",   "RdFamB",
        "BerSport","RdSportA", "RdSportB",
        "BerWrk",  "RdWrkA",   "RdWrkB",
        "BerOnd",  "RdOndA",   "RdOndB",
        "BerSup",  "RdSupA",   "RdSupB",
        "BerZiek", "RdZiekA",  "RdZiekB",
        "BerArts", "RdArtsA",  "RdArtsB",
        "BerStat", "RdStatA",  "RdStatB",
        # — Daily-pattern flags —
        "ByzDag","ByzAdr","ByzVvm","ByzTyd","ByzDuur","ByzRoute","ByzReden",
    ]

    categorical_cols = [
        "BrandstofPa1",   # P – Primary fuel newest car (1 = Petrol, 2 = Diesel, 3 = LPG, 4 = Electric, 5 = Other, 6 = Unknown, 7 = N/A)
        "XBrandstofPa1",  # P – Secondary fuel newest car (0 = None, 1 = Petrol, 2 = Diesel, 3 = LPG, 4 = Electric, 5 = Other, 6 = Unknown, 7 = N/A)
        "BrandstofEPa1",  # P – Electric drivetrain type newest car (0 = Not electric, 1 = Full EV, 2 = Plug-in hybrid, 3 = Hybrid, 4 = Other, 5 = Unknown, 6 = N/A)
        "TenaamPa1",      # P – Registered owner newest car (1 = Own name, 2 = Other household member, 3 = N/A)
        "BrandstofPa2", "XBrandstofPa2", "BrandstofEPa2", "TenaamPa2", # -- same coding for 2nd car
        "BrandstofPaL", "XBrandstofPaL", "BrandstofEPaL",              # -- same coding lease/company car
        "Doel", "MotiefV",         # V – Detailed purpose / motive (see 14 & 13-code lists)
        "VertLoc",                 # V – Departure location type (1 = Home, 2 = Other home, 3 = Work, 4 = Other)
        "VertGeb", "AankGeb",      # V – Departure / arrival country (0 = NL, 1‥99 see table)
        "VertGem", "AankGem",      # V – Departure / arrival municipality (14‥1991, 9999 Unknown)
        "VertProv","AankProv",     # V – Departure / arrival province (0 = Abroad or none, 1‥12 NL provinces, 99 Unknown)
        "VertCorop","AankCorop",   # V – Departure / arrival COROP region (0 = Abroad or none, 1‥40 list, 99 Unknown)
        "SPlaats1","SPlaats2","SPlaats3","SPlaats4","SPlaats5",  # V – Place names in series trip (free text / code)
        "SVvm1","SVvm2","SVvm3","SVvm4",  # V – Up-to-four modes used in a series (1 = Car, … 24 = Other w/o motor, 25 = N/A)
        "Rvm","Hvm",               # R/V – Detailed ride / main trip mode (1 = Car, 2 = Train, … 24)
        "HvmRol", "RvmRol",        # Role in mode (1 = Driver, 2 = Passenger, 3 = Unknown, 4 = N/A)
    ]


    drop_cols = [
    "OP",        # P – New-person row flag (0 = No new person, 1 = New person)
    "OPID",      # P – Unique ID for each respondent (person key)
    "Steekproef",# P – Sample indicator (1 = Core survey, 4 = Extra North-Wing, 6 = Extra Rotterdam-The Hague, 8 = Extra Utrecht)
    "Mode",      # P – Response mode (1 = CAWI – web; other modes not used in 2022)
    "Corop",     # P – COROP region of residence (1 = East Groningen, 2 = Rest Groningen, … 40 = Flevoland)
    "BuurtAdam", # P – Amsterdam neighbourhood combo (0 = Not Amsterdam resident, 036300–036399 = 100+ neighbourhood codes)
    "KLeeft",    # P – Age class (2 = 6–11 y, 3 = 12–14 y, 4 = 15–17 y, 5 = 18–19 y, 6 = 20–24 y, 7 = 25–29 y, 8 = 30–34 y, 9 = 35–39 y, 10 = 40–44 y, 11 = 45–49 y)
    "Jaar",      # P – Reporting year (2022)
    "Maand",     # P – Reporting month (1 = Jan … 12 = Dec)
    "Week",      # P – ISO week number of diary day (1‥53)
    "Dag",       # P – Calendar day of the month (1‥31)
    "Weekdag",   # P – Day of week (1 = Sunday, 2 = Monday, 3 = Tuesday, 4 = Wednesday, 5 = Thursday, 6 = Friday, 7 = Saturday)
    "Feestdag",  # P – Diary day is Dutch public holiday (0 = No, 1 = Yes)
]



    for col in drop_cols:
        if col not in df.columns:
            print(f"Drop Column {col} not in dataframe")
            drop_cols.remove(col)

    for col in categorical_cols:
        if col not in df.columns:
            print(f"Categorical Column {col} not in dataframe")
            categorical_cols.remove(col)

    X_train, X_test, y_train, y_test = make_ml_dataset(
        df, 
        target_col=target_col, 
        target_vals=target_val,
        categorical_cols=categorical_cols, 
        drop_cols=drop_cols
        )
    
    if validation_proportion is not None:
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_proportion, random_state=random_state)
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        return X_train, X_test, y_train, y_test
    