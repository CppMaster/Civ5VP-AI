import pandas as pd
from sklearn.model_selection import train_test_split  # type: ignore
import sys
from pathlib import Path


CSV_PATH = Path("data/processed/v1/game_data.csv")
OUTPUT_DIR = Path("data/processed/v1")


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


def split_features_target(df: pd.DataFrame, target_column: str = 'won') -> tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into features (X) and target (Y) for machine learning.
    
    Args:
        df: Preprocessed DataFrame
        target_column: Name of the target column (default: 'won')
    
    Returns:
        Tuple of (X, Y) where X is features DataFrame and Y is target Series
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    X = df.drop(columns=[target_column])
    Y = df[target_column]
    
    return X, Y


def split_train_valid(
    X: pd.DataFrame,
    Y: pd.Series,
    valid_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split features and target into train/validation sets.

    Args:
        X: Features DataFrame
        Y: Target Series
        valid_size: Fraction of data to use for validation (default 0.2)
        random_state: Seed for reproducibility

    Returns:
        X_train, X_valid, Y_train, Y_valid
    """
    return train_test_split(
        X, Y, test_size=valid_size, random_state=random_state, stratify=Y
    )


def save_processed_data(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    Y_train: pd.Series,
    Y_valid: pd.Series,
    output_dir: Path,
) -> None:
    """
    Save processed train/validation data to CSV files.
    
    Args:
        X_train: Training features DataFrame
        X_valid: Validation features DataFrame
        Y_train: Training target Series
        Y_valid: Validation target Series
        output_dir: Directory to save the files
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save features
    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_valid.to_csv(output_dir / "X_valid.csv", index=False)
    
    # Save targets (convert Series to DataFrame for consistency)
    Y_train.to_csv(output_dir / "Y_train.csv", index=False, header=['won'])
    Y_valid.to_csv(output_dir / "Y_valid.csv", index=False, header=['won'])
    
    print(f"\nData saved to {output_dir}:")
    print(f"  - X_train.csv")
    print(f"  - X_valid.csv")
    print(f"  - Y_train.csv")
    print(f"  - Y_valid.csv")


def main() -> None:
    """Main entry point."""
    try:
        df = load_and_clean_csv(CSV_PATH)
        print("CSV loaded and cleaned successfully:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        
        # Split into X and Y
        X, Y = split_features_target(df)
        print(f"\nSplit into features (X) and target (Y):")
        print(f"  X shape: {X.shape}")
        print(f"  Y shape: {Y.shape}")
        print(f"  Y value counts:\n{Y.value_counts()}")
        
        # Train/validation split
        X_train, X_valid, Y_train, Y_valid = split_train_valid(X, Y)
        print("\nTrain/Validation split (valid_size=0.2):")
        print(f"  X_train shape: {X_train.shape}")
        print(f"  X_valid shape: {X_valid.shape}")
        print(f"  Y_train shape: {Y_train.shape}")
        print(f"  Y_valid shape: {Y_valid.shape}")
        print(f"  Y_train value counts:\n{Y_train.value_counts()}")
        print(f"  Y_valid value counts:\n{Y_valid.value_counts()}")
        
        # Save processed data
        save_processed_data(X_train, X_valid, Y_train, Y_valid, OUTPUT_DIR)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

