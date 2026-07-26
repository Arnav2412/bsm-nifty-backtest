from pathlib import Path
from zipfile import ZipFile, BadZipFile
from io import TextIOWrapper

import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

RAW_FOLDER = Path("data/raw_options")
OUTPUT_FILE = Path("data/nifty50_options.csv")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# STANDARD OUTPUT COLUMNS
# ============================================================

OUTPUT_COLUMNS = [
    "Date",
    "Expiry",
    "Strike",
    "Type",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "OpenInterest",
    "Underlying",
]


# ============================================================
# HELPER: CLEAN COLUMN NAMES
# ============================================================

def clean_column_names(df):
    """
    Remove extra spaces from NSE column names.
    """

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    return df


# ============================================================
# PROCESS OLD NSE BHAVCOPY
# ============================================================

def process_old_bhavcopy(df):

    df = clean_column_names(df)

    # --------------------------------------------------------
    # Keep only NIFTY index options
    # --------------------------------------------------------

    df["INSTRUMENT"] = df["INSTRUMENT"].astype(str).str.strip()
    df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip()

    df = df[
        (df["INSTRUMENT"] == "OPTIDX") &
        (df["SYMBOL"] == "NIFTY")
    ].copy()

    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    # --------------------------------------------------------
    # Rename NSE columns into our standard names
    # --------------------------------------------------------

    rename_map = {
        "TIMESTAMP": "Date",
        "EXPIRY_DT": "Expiry",
        "STRIKE_PR": "Strike",
        "OPTION_TYP": "Type",
        "OPEN": "Open",
        "HIGH": "High",
        "LOW": "Low",
        "CLOSE": "Close",
        "CONTRACTS": "Volume",
        "OPEN_INT": "OpenInterest",
    }

    df = df.rename(columns=rename_map)

    if "UNDERLYING" in df.columns:
        df["Underlying"] = df["UNDERLYING"]

    elif "UNDERLYING_VALUE" in df.columns:
        df["Underlying"] = df["UNDERLYING_VALUE"]

    else:
        df["Underlying"] = pd.NA
    # --------------------------------------------------------
    # Keep only required columns
    # --------------------------------------------------------

    df = df[OUTPUT_COLUMNS]

    return df


# ============================================================
# FIND UDIFF COLUMN
# ============================================================

def find_column(df, possible_names):

    normalized = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for name in possible_names:

        if name.lower() in normalized:
            return normalized[name.lower()]

    return None


# ============================================================
# PROCESS NEW UDIFF BHAVCOPY
# ============================================================

def process_udiff_bhavcopy(df):

    df = clean_column_names(df)

    # --------------------------------------------------------
    # Clean relevant text columns
    # --------------------------------------------------------

    df["FinInstrmTp"] = (
        df["FinInstrmTp"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["TckrSymb"] = (
        df["TckrSymb"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["OptnTp"] = (
        df["OptnTp"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # --------------------------------------------------------
    # Keep only NIFTY index options
    #
    # IDO = Index Derivative Option
    # --------------------------------------------------------

    df = df[
        (df["FinInstrmTp"] == "IDO") &
        (df["TckrSymb"] == "NIFTY") &
        (df["OptnTp"].isin(["CE", "PE"]))
    ].copy()

    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    # --------------------------------------------------------
    # Convert UDiFF schema to our standard schema
    # --------------------------------------------------------

    result = pd.DataFrame()

    result["Date"] = df["TradDt"]

    result["Expiry"] = df["XpryDt"]

    result["Strike"] = df["StrkPric"]

    result["Type"] = df["OptnTp"]

    result["Open"] = df["OpnPric"]

    result["High"] = df["HghPric"]

    result["Low"] = df["LwPric"]

    result["Close"] = df["ClsPric"]

    result["Volume"] = df["TtlTradgVol"]

    result["OpenInterest"] = df["OpnIntrst"]

    result["Underlying"] = df["UndrlygPric"]

    return result
def read_zip_file(zip_path):

    try:

        with ZipFile(zip_path, "r") as zip_file:

            csv_files = [
                name
                for name in zip_file.namelist()
                if name.lower().endswith(".csv")
            ]

            if not csv_files:
                raise ValueError(
                    f"No CSV found inside {zip_path}"
                )

            csv_name = csv_files[0]

            with zip_file.open(csv_name) as csv_file:

                df = pd.read_csv(
                    TextIOWrapper(
                        csv_file,
                        encoding="utf-8",
                        errors="replace",
                    ),
                    low_memory=False,
                )

                return df

    except BadZipFile:
        raise ValueError(
            f"Invalid ZIP file: {zip_path}"
        )


# ============================================================
# CLEAN FINAL DATASET
# ============================================================

def clean_final_dataset(df):

    # --------------------------------------------------------
    # Convert dates
    # --------------------------------------------------------

    df["Date"] = pd.to_datetime(
    df["Date"],
    format="mixed",
    errors="coerce",
    dayfirst=True,
)

    df["Expiry"] = pd.to_datetime(
        df["Expiry"],
        format="mixed",
        errors="coerce",
        dayfirst=True,
    )
    # --------------------------------------------------------
    # Convert numerical columns
    # --------------------------------------------------------

    numeric_columns = [
        "Strike",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "OpenInterest",
        "Underlying",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Standardize CE / PE
    # --------------------------------------------------------

    df["Type"] = (
        df["Type"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # --------------------------------------------------------
    # Remove invalid rows
    # --------------------------------------------------------

    print("\nDate parsing check:")
    print(f"Invalid Date values   : {df['Date'].isna().sum():,}")
    print(f"Invalid Expiry values : {df['Expiry'].isna().sum():,}")
    print(f"Earliest parsed date  : {df['Date'].min()}")
    print(f"Latest parsed date    : {df['Date'].max()}")
    df = df.dropna(
        subset=[
            "Date",
            "Expiry",
            "Strike",
            "Type",
            "Close",
        ]
    )

    df = df[
        df["Type"].isin(["CE", "PE"])
    ]

    # Strike must be positive
    df = df[
        df["Strike"] > 0
    ]

    # Option price cannot be negative
    df = df[
        df["Close"] >= 0
    ]

    # Expiry cannot be before trade date
    df = df[
        df["Expiry"] >= df["Date"]
    ]

    # --------------------------------------------------------
    # Remove duplicate observations
    # --------------------------------------------------------

    df = df.drop_duplicates(
        subset=[
            "Date",
            "Expiry",
            "Strike",
            "Type",
        ],
        keep="last",
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df = df.sort_values(
        by=[
            "Date",
            "Expiry",
            "Strike",
            "Type",
        ]
    )

    df = df.reset_index(drop=True)

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n============================================")
    print(" NIFTY OPTIONS PREPROCESSOR")
    print("============================================\n")

    zip_files = sorted(
        RAW_FOLDER.rglob("*.zip")
    )

    print(
        f"Found {len(zip_files)} ZIP files.\n"
    )

    if not zip_files:
        print(
            "No ZIP files found in "
            f"{RAW_FOLDER.resolve()}"
        )
        return

    all_data = []

    successful = 0
    failed = 0

    # --------------------------------------------------------
    # Process every downloaded bhavcopy
    # --------------------------------------------------------

    for number, zip_path in enumerate(
        zip_files,
        start=1,
    ):

        try:

            df = read_zip_file(zip_path)

            filename = zip_path.name

            # Old NSE filenames begin with "fo"
            if filename.lower().startswith("fo"):

                nifty = process_old_bhavcopy(df)

                file_type = "OLD"

            else:

                nifty = process_udiff_bhavcopy(df)

                file_type = "UDIFF"

            if not nifty.empty:
                all_data.append(nifty)

            successful += 1

            print(
                f"[{number}/{len(zip_files)}] "
                f"{file_type:<5} "
                f"{filename} "
                f"-> {len(nifty):,} NIFTY options"
            )

        except Exception as e:

            failed += 1

            print(
                f"[ERROR] {zip_path.name}"
            )

            print(
                f"        {e}"
            )

    # --------------------------------------------------------
    # Make sure something was extracted
    # --------------------------------------------------------

    if not all_data:

        print("\nNo NIFTY option data extracted.")
        return

    # --------------------------------------------------------
    # Combine every trading day
    # --------------------------------------------------------

    print("\nCombining files...")

    final_df = pd.concat(
        all_data,
        ignore_index=True,
    )

    print(
        f"Rows before cleaning: "
        f"{len(final_df):,}"
    )

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    final_df = clean_final_dataset(
        final_df
    )

    print(
        f"Rows after cleaning : "
        f"{len(final_df):,}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    final_df.to_csv(
        OUTPUT_FILE,
        index=False,
        date_format="%Y-%m-%d",
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n============================================")
    print(" PREPROCESSING COMPLETE")
    print("============================================")

    print(
        f"Files processed : {successful:,}"
    )

    print(
        f"Files failed    : {failed:,}"
    )

    print(
        f"Option rows     : {len(final_df):,}"
    )

    print(
        f"First date      : "
        f"{final_df['Date'].min().date()}"
    )

    print(
        f"Last date       : "
        f"{final_df['Date'].max().date()}"
    )

    print(
        f"Output          : {OUTPUT_FILE}"
    )

    print("============================================")

    # Show first few rows
    print("\nFirst 10 rows:\n")

    print(
        final_df.head(10).to_string(
            index=False
        )
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()