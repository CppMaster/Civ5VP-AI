import pandas as pd
import sys
from pathlib import Path


CSV_PATH = Path("data/processed/game_data.csv")


def _min_max_normalize(series: pd.Series) -> pd.Series:
    """Normalize a numeric series to [0, 1]."""
    min_val = series.min()
    max_val = series.max()
    if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
        # Avoid division by zero; keep zeros if constant or all NaN
        return pd.Series(0.0, index=series.index, dtype=float)
    return (series - min_val) / (max_val - min_val)


def load_and_clean_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load CSV file, drop identifiers, one-hot encode categorical fields,
    convert win flag to int, and normalize remaining numeric columns.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Load CSV into DataFrame
    df = pd.read_csv(csv_path)
    
    # Remove specified columns
    columns_to_remove = ['game_id', 'player_id', 'victory_type']
    
    # Check for unnamed index column (pandas reads empty header as 'Unnamed: 0')
    if df.columns[0].startswith('Unnamed'):
        columns_to_remove.append(df.columns[0])
    
    # Drop the columns (errors='ignore' handles missing columns gracefully)
    df = df.drop(columns=columns_to_remove, errors='ignore')

    # Convert won to integer (True/False -> 1/0)
    if 'won' in df.columns:
        df['won'] = df['won'].astype(int)

    # One-hot encode categorical columns
    categorical_cols = [col for col in ['civilization', 'chosen_ancient_policy'] if col in df.columns]
    df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols, dummy_na=False)

    # Identify one-hot columns to exclude from normalization
    one_hot_prefixes = [f"{col}_" for col in categorical_cols]
    one_hot_cols = [col for col in df.columns for prefix in one_hot_prefixes if col.startswith(prefix)]

    # Normalize numeric columns except the win flag and one-hot columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    skip_normalization = set(['won']) | set(one_hot_cols)
    cols_to_normalize = [col for col in numeric_cols if col not in skip_normalization]

    for col in cols_to_normalize:
        df[col] = _min_max_normalize(df[col])
    
    return df


def main() -> None:
    """Main entry point."""
    try:
        df = load_and_clean_csv(CSV_PATH)
        print("CSV loaded and cleaned successfully:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"\nFirst few rows:")
        print(df.head())
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

