import os
import pandas as pd

from codebase.data.load_demographics import load_excel


def make_ml_dataset(df, target_col, drop_cols, categorical_cols=None, target_val=None, test_size=0.2, random_state=42, stratification_col=None) -> tuple:
    """
    Splits the dataset into training and testing sets.
    """
    from sklearn.model_selection import train_test_split

    # Drop specified columns
    df_ = df.drop(columns=drop_cols)

    # Split the data into features and target
    X = df_.drop(columns=[target_col])
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True) if categorical_cols else X
    y = df_[target_col] == target_val if target_val is not None else df_[target_col]
    stratification = df_[stratification_col] if stratification_col else None

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratification)

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

    moving_cols = [
    "Verpl",
    "VerplID",
    "VerplNr",
    "Toer",
    "AantRit"
    "VertLoc", "VertGeb", "VertPC", "VertPCBL", "VertGem", "VertProv", "VertCorop", "VertMRA", "VertMRDH", "VertUtr",
    "AankGeb", "AankPC", "AankPCBL", "AankGem", "AankProv", "AankCorop", "AankMRA", "AankMRDH", "AankUtr",
    "PCG", "GemG", "PCBLG", "AfstV", "KAfstV",
    "Hvm", "HvmRol", "KHvm",
    "VertUur", "VertMin", "KVertTijd",
    "AankUur", "AankMin", "Reisduur", "KReisduur",
    "ActDuur", "Kind6", "VolgWerk",
    "SAantAdr", "SDezPlts",
    "SPlaats1", "SPlaats2", "SPlaats3", "SPlaats4", "SPlaats5",
    "AfstS", "AfstSBL",
    "SVvm1", "SVvm2", "SVvm3", "SVvm4",
    "SBegUur", "SBegMin", "SEindUur", "SEindMin",
    "CorrVerpl", "GehBLVer",
    "Rit", "RitID", "RitNr",
    "AfstR", "AfstRBL", "KAfstR",
    "Rvm", "RvmRol", "RAantIn", "KRvm",
    "RVertUur", "RVertMin", "RAankUur", "RAankMin",
    "RReisduur", "RReisduurBL",
    "RVertStat", "RAankStat",
    "RTSamen", "RCorrSnelh", "RVliegVer",
    "FactorH", "FactorP", "FactorV"
    ]    

    drop_cols = [
        "OP", 
        "OPID",
        "Steekproef", 
        "Mode",
        "Corop",
        "BuurtAdam",
        "KLeeft",
        "Jaar",
        "Maand",
        "Week",
        "Dag",
        "Weekdag",
        "Feestdag",
    ] + moving_cols

    numerical_cols = [
        "HHPers",
        "HHLft1",
        "HHLft2",
        "HHLft3",
        "HHLft4",
        "Leeftijd",
        "HHRijbweijsAu",
        "HHRijbewijsMo",
        "HHRijbewijsBr",
        "HHAuto",
        "HHAutoL",
        "OPAuto",
        "BouwjaarPa1",
        "BouwjaarPa2",
        "BouwjaarPaL",
        "HHMotor",
        "OPBrom",
        "HHSnor",
        "OPSnor",
        "HHFiets",
        "ReisduurOP",
        "AfstandOP",
        "AfstandSOP",
    ]

    binary_cols = [
        "OPRijbewijsAu",
        "OPRijbewijsMo",
        "OPRijbewijsBr"
    ]

    ordinal_cols = [
        "HHBestInkG",
        "HHGestInkG",
        "HHLaagInk",
        "HHSocInk",
        "HHWelvG",
        "KBouwjaarPa1",
        "KBouwjaarPa2",
        "KGewitchPa1",
        "KGewichtPa2",
        "KBouwjaarPaL",
        "KGewichtPaL",
        "FqLopen",
        "FqNEFiets",
        "FqEFiets",
        "FqBTM",
        "FqTrein",
        "FqAutoB",
        "FqAutoP",
        "FqMotor",
        "FqBrSnor",
        "BerWrk",
        "BerOnd",
        "AantVpl",
        "AantOVVPl",
        "AantSVpl",
    ]

    categorical_cols = [
        "HHSam",
        "HHPlOP",
        "WoPC",
        "Wogem",
        "Sted",
        "GemGr",
        "Prov",
        "MRA",
        "MRDH",
        "Utr",
        "Geslacht",
        "Herkomst",
        "BetWerk",
        "OnbBez",
        "MaatsPart",
        "Opleiding",
        "BrandstofPa1",
        "XBrandstofPa1",
        "BrandstofEPa1",
        "BrandstofPaL",
        "XBrandstofPaL",
        "BrandstofEPaL",
        "TenaamPa2",
        "HHBezitVm",
        "OPBezitVm",
        "OVStKaart",
        # Yes/no/not relevant:
        "WrkVerg",
        "VergVast",
        "VergKm",
        "VergBrSt",
        "VergOV",
        "VergAans",
        "VergVoer",
        "VergBudg",
        "VergPark",
        "VergStal",
        "VergAnd",
        # End of yes/no/not relevant
        "RdWrkA",
        "RdWrkB",
        "RdOndA",
        "RdOndB",
        "BerSup",
        "RdSupA",
        "RdSupB",
        "BerZiek",  # last 3 are unknown, rest ranked (Ber seems to be the prefix for this)
        "RdZiekA",
        "RdZiekB",
        "BerArts",  # last 3 are unknown, rest ranked
        "RdArtsA",
        "RdArtsB",
        "BerStat",  # last 3 are unknown, rest ranked
        "RdStatA",
        "RdStatB",
        "BerHalte",  # last 3 are unknown, rest ranked
        "RDHalteA",
        "BerFam",
        "RdFamA",
        "RdFamB",
        "BerSport",
        "RdSportA",
        "RdSportB",
        "Weggeweest",
        "RedenNW",
        "RedenNWZ",
        "RedenNWW",
        "RedenNWB",
        "EFiets",
        # 0, 3, 4 unknownm 1, 2 are yes/no
        "AutoEig",
        "AutoHhl",
        "AutoLWg",
        "AutoLPl",
        "AutoBed",
        "AutoDOrg",
        "AutoDPart",
        "AutoDBek",
        "AutoLeen",
        "AutoHuur",
        "AutoAnd",
        "ByzDag",
        "ByzAdr",
        "ByzVvm",
        "ByzTyd",
        "ByzDuur",
        "ByzRoute",
        "ByzReden",
        "Doel",
        "MotiefV",
        "KMotiefV",
        "MeerWink",     # 0, 1 yes/no, 2, 3 unknown
        "AardWerk",
    ] + ordinal_cols + binary_cols

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
        target_val=target_val,
        categorical_cols=categorical_cols, 
        drop_cols=drop_cols
        )
    
    if validation_proportion is not None:
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_proportion, random_state=random_state)
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        return X_train, X_test, y_train, y_test
    