import pandas as pd
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import joblib  # type: ignore


DATA_DIR = Path("data/processed/v1")
MODEL_DIR = Path("data/models/dpv1/v1")


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load train and validation datasets.
    
    Args:
        data_dir: Directory containing X_train.csv, X_valid.csv, Y_train.csv, Y_valid.csv
    
    Returns:
        X_train, X_valid, Y_train, Y_valid
    """
    X_train = pd.read_csv(data_dir / "X_train.csv")
    X_valid = pd.read_csv(data_dir / "X_valid.csv")
    Y_train = pd.read_csv(data_dir / "Y_train.csv")["won"]
    Y_valid = pd.read_csv(data_dir / "Y_valid.csv")["won"]
    
    return X_train, X_valid, Y_train, Y_valid


def train_model(
    X_train: pd.DataFrame,
    Y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: int = None,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier to predict win probability.
    
    Args:
        X_train: Training features
        Y_train: Training target
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of trees (None for unlimited)
        random_state: Random seed for reproducibility
    
    Returns:
        Trained RandomForestClassifier model
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,  # Use all available cores
    )
    
    print("Training Random Forest classifier...")
    model.fit(X_train, Y_train)
    print("Training completed!")
    
    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_valid: pd.DataFrame,
    Y_valid: pd.Series,
) -> dict:
    """
    Evaluate model performance on validation set.
    
    Args:
        model: Trained classifier
        X_valid: Validation features
        Y_valid: Validation target
    
    Returns:
        Dictionary with evaluation metrics
    """
    Y_pred = model.predict(X_valid)
    Y_pred_proba = model.predict_proba(X_valid)[:, 1]
    
    accuracy = accuracy_score(Y_valid, Y_pred)
    precision = precision_score(Y_valid, Y_pred, zero_division=0)
    recall = recall_score(Y_valid, Y_pred, zero_division=0)
    f1 = f1_score(Y_valid, Y_pred, zero_division=0)
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }
    
    print("\n" + "=" * 60)
    print("Validation Set Performance:")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(Y_valid, Y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(Y_valid, Y_pred))
    
    return metrics


def save_model(model: RandomForestClassifier, model_dir: Path) -> None:
    """
    Save trained model to disk.
    
    Args:
        model: Trained classifier
        model_dir: Directory to save the model
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "win_prediction_model.joblib"
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")


def print_feature_importance(model: RandomForestClassifier, feature_names: list, top_n: int = 20) -> None:
    """
    Print top N most important features.
    
    Args:
        model: Trained RandomForestClassifier
        feature_names: List of feature names
        top_n: Number of top features to display
    """
    importances = model.feature_importances_
    feature_importance = list(zip(feature_names, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop {top_n} Most Important Features:")
    print("-" * 60)
    for i, (feature, importance) in enumerate(feature_importance[:top_n], 1):
        print(f"{i:2d}. {feature:40s} {importance:.6f}")


def main() -> None:
    """Main entry point."""
    try:
        # Load data
        print("Loading data...")
        X_train, X_valid, Y_train, Y_valid = load_data(DATA_DIR)
        print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"Validation set: {X_valid.shape[0]} samples")
        print(f"Target distribution (train):\n{Y_train.value_counts()}")
        
        # Train model
        model = train_model(X_train, Y_train)
        
        # Evaluate model
        metrics = evaluate_model(model, X_valid, Y_valid)
        
        # Print feature importance
        print_feature_importance(model, list(X_train.columns))
        
        # Save model
        save_model(model, MODEL_DIR)
        
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

