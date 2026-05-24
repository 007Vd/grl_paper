import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

feature_files = list(
    PROCESSED_DATA_DIR.glob("*_features.csv")
)

MACRO_FEATURES = {
    "RECPROUSM156N": "recession_probability",
    "CORESTICKM159SFRBATL": "core_sticky_inflation",
    "PCETRIM12M159SFRBDAL": "trimmed_pce",
    "CPALTT01USM657N": "cpi",
    "PSAVERT": "personal_savings_rate",
    "AISRSA": "real_auto_sales",
    "ANFCI": "financial_conditions",
    "UNEMPLOY": "unemployment"
}

def load_macro_feature(file_name, column_name):

    macro_path = RAW_DATA_DIR / f"{file_name}.csv"

    macro_df = pd.read_csv(macro_path)

    macro_df["Date"] = pd.to_datetime(
        macro_df["Date"]
    )

    macro_df.columns = [
        "date",
        column_name
    ]

    macro_df = macro_df.set_index("date")

    macro_daily = macro_df.resample("D").ffill()

    return macro_daily

for file in feature_files:

    print(f"\nprocessing {file.name}")

    try:

        asset_df = pd.read_csv(file)

        asset_df["date"] = pd.to_datetime(
            asset_df["date"]
        )

        asset_df = asset_df.set_index("date")

        for fred_id, readable_name in MACRO_FEATURES.items():

            macro_daily = load_macro_feature(
                fred_id,
                readable_name
            )

            asset_df = asset_df.join(
                macro_daily,
                how="left"
            )

        asset_df = asset_df.ffill()

        save_name = (
            file.stem.replace(
                "_features",
                "_merged"
            ) + ".csv"
        )

        save_path = PROCESSED_DATA_DIR / save_name

        asset_df.to_csv(save_path)

        print(f"saved: {save_path}")

    except Exception as e:

        print(
            f"ERROR merging datasets for "
            f"{file.name}: {e}"
        )