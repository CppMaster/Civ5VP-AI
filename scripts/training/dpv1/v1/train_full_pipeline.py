import pandas as pd
import numpy as np
import sys
import copy
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
import joblib  # type: ignore


CSV_PATH = Path("data/processed/v1/game_data.csv")
MODEL_DIR = Path("data/models/dpv1/v1")


class WinPredictionDataset(Dataset):
    """PyTorch Dataset for win prediction."""
    
    def __init__(self, features: pd.DataFrame, targets: pd.Series):
        """
        Args:
            features: DataFrame with feature columns
            targets: Series with target values (0 or 1)
        """
        # Convert to numpy arrays
        self.features = features.values.astype(np.float32)
        self.targets = targets.values.astype(np.float32)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.targets[idx])


class WinPredictionNet(nn.Module):
    """Neural network for predicting win probability."""
    
    def __init__(self, input_size: int, hidden_sizes: list[int] = [128, 64, 32], dropout: float = 0.3):
        """
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            dropout: Dropout probability
        """
        super(WinPredictionNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.BatchNorm1d(hidden_size))
            prev_size = hidden_size
        
        # Output layer (single neuron for binary classification)
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())  # Sigmoid for probability output
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze()


class PreprocessingPipeline:
    """Pipeline for preprocessing data for inference."""
    
    def __init__(self):
        self.columns_to_remove = ['game_id', 'player_id', 'victory_type']
        self.categorical_cols = ['civilization', 'chosen_ancient_policy']
        self.normalization_params = {}  # {col_name: {'min': val, 'max': val}}
        self.feature_columns = []  # Final feature column order
        self.categorical_mappings = {}  # Store one-hot encoding mappings
    
    def fit(self, df: pd.DataFrame):
        """Fit the preprocessing pipeline on training data."""
        # Store original categorical values for reference
        for col in self.categorical_cols:
            if col in df.columns:
                unique_vals = sorted(df[col].dropna().unique())
                self.categorical_mappings[col] = unique_vals
        
        # Identify numeric columns that need normalization
        numeric_cols = df.select_dtypes(include=['number']).columns
        one_hot_prefixes = [f"{col}_" for col in self.categorical_cols]
        one_hot_cols = [col for col in df.columns for prefix in one_hot_prefixes if col.startswith(prefix)]
        skip_normalization = set(['won']) | set(one_hot_cols)
        cols_to_normalize = [col for col in numeric_cols if col not in skip_normalization]
        
        # Store min/max for normalization
        for col in cols_to_normalize:
            min_val = df[col].min()
            max_val = df[col].max()
            if not (pd.isna(min_val) or pd.isna(max_val) or min_val == max_val):
                self.normalization_params[col] = {'min': float(min_val), 'max': float(max_val)}
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted pipeline."""
        df = df.copy()
        
        # Remove specified columns
        if df.columns[0].startswith('Unnamed'):
            self.columns_to_remove.append(df.columns[0])
        df = df.drop(columns=self.columns_to_remove, errors='ignore')
        
        # Convert won to integer if present
        if 'won' in df.columns:
            df['won'] = df['won'].astype(int)
        
        # One-hot encode categorical columns
        categorical_cols = [col for col in self.categorical_cols if col in df.columns]
        df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols, dummy_na=False)
        
        # Normalize numeric columns
        for col, params in self.normalization_params.items():
            if col in df.columns:
                min_val = params['min']
                max_val = params['max']
                if max_val != min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
                else:
                    df[col] = 0.0
        
        # Store feature columns order (excluding target) - only set during fit/initial transform
        if not self.feature_columns:
            if 'won' in df.columns:
                self.feature_columns = [col for col in df.columns if col != 'won']
            else:
                self.feature_columns = list(df.columns)
        
        # Ensure all expected feature columns exist (for inference with missing one-hot columns)
        if self.feature_columns:
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0
            # Reorder columns to match expected order
            if 'won' in df.columns:
                return df[self.feature_columns + ['won']]
            else:
                return df[self.feature_columns]
        
        return df


def load_and_preprocess(csv_path: Path, pipeline: PreprocessingPipeline = None) -> tuple[pd.DataFrame, PreprocessingPipeline]:
    """Load CSV and preprocess it."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Create or use existing pipeline
    if pipeline is None:
        pipeline = PreprocessingPipeline()
        pipeline.fit(df)
    
    # Transform data
    df = pipeline.transform(df)
    
    return df, pipeline


def split_features_target(df: pd.DataFrame, target_column: str = 'won') -> tuple[pd.DataFrame, pd.Series]:
    """Split DataFrame into features (X) and target (Y)."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    X = df.drop(columns=[target_column])
    Y = df[target_column]
    
    return X, Y


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for features, targets in dataloader:
        features = features.to(device)
        targets = targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for features, targets in dataloader:
            features = features.to(device)
            targets = targets.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Store predictions and targets
            predictions = (outputs > 0.5).float().cpu().numpy()
            all_predictions.extend(predictions)
            all_targets.extend(targets.cpu().numpy())
    
    avg_loss = total_loss / num_batches
    return avg_loss, np.array(all_predictions), np.array(all_targets)


def train_model(
    X_train: pd.DataFrame,
    Y_train: pd.Series,
    X_valid: pd.DataFrame,
    Y_valid: pd.Series,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    hidden_sizes: list[int] = [128, 64, 32],
    dropout: float = 0.3,
    device: torch.device = None,
) -> nn.Module:
    """Train neural network model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Create datasets and dataloaders
    train_dataset = WinPredictionDataset(X_train, Y_train)
    valid_dataset = WinPredictionDataset(X_valid, Y_valid)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    input_size = X_train.shape[1]
    model = WinPredictionNet(input_size, hidden_sizes, dropout).to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\nStarting training for {epochs} epochs...")
    print("=" * 60)
    
    best_valid_loss = float('inf')
    patience_counter = 0
    patience = 10
    best_model_state = None
    
    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        valid_loss, valid_preds, valid_targets = evaluate(model, valid_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(valid_loss)
        
        # Calculate metrics
        accuracy = accuracy_score(valid_targets, valid_preds)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}]")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Valid Loss: {valid_loss:.4f}")
            print(f"  Valid Accuracy: {accuracy:.4f}")
        
        # Early stopping
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            if (epoch + 1) % 5 != 0 and epoch != 0:
                print(f"Epoch [{epoch+1}/{epochs}] - New best validation loss: {valid_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                print(f"Best validation loss: {best_valid_loss:.4f}")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break
    
    print("=" * 60)
    print("Training completed!")
    
    return model


def evaluate_model(
    model: nn.Module,
    X_valid: pd.DataFrame,
    Y_valid: pd.Series,
    device: torch.device = None,
) -> dict:
    """Evaluate model performance on validation set."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    valid_dataset = WinPredictionDataset(X_valid, Y_valid)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
    
    all_predictions = []
    all_probabilities = []
    all_targets = []
    
    with torch.no_grad():
        for features, targets in valid_loader:
            features = features.to(device)
            outputs = model(features)
            
            probabilities = outputs.cpu().numpy()
            predictions = (probabilities > 0.25).astype(int)
            
            all_probabilities.extend(probabilities)
            all_predictions.extend(predictions)
            all_targets.extend(targets.numpy())
    
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_targets = np.array(all_targets)
    
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, zero_division=0)
    recall = recall_score(all_targets, all_predictions, zero_division=0)
    f1 = f1_score(all_targets, all_predictions, zero_division=0)
    auc = roc_auc_score(all_targets, all_probabilities)
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": auc,
    }
    
    print("\n" + "=" * 60)
    print("Validation Set Performance:")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_targets, all_predictions))
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_targets, all_predictions))
    
    return metrics


def save_pipeline(
    model: nn.Module,
    pipeline: PreprocessingPipeline,
    model_dir: Path,
    hidden_sizes: list[int],
    dropout: float,
) -> None:
    """Save model and preprocessing pipeline for inference."""
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = model_dir / "win_prediction_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': len(pipeline.feature_columns),
        'hidden_sizes': hidden_sizes,
        'dropout': dropout,
    }, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save preprocessing pipeline
    pipeline_path = model_dir / "preprocessing_pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)
    print(f"Preprocessing pipeline saved to: {pipeline_path}")
    
    # Save metadata as JSON for easy inspection
    metadata = {
        'feature_columns': pipeline.feature_columns,
        'normalization_params': pipeline.normalization_params,
        'categorical_mappings': pipeline.categorical_mappings,
        'columns_to_remove': pipeline.columns_to_remove,
        'categorical_cols': pipeline.categorical_cols,
        'model_config': {
            'input_size': len(pipeline.feature_columns),
            'hidden_sizes': hidden_sizes,
            'dropout': dropout,
        }
    }
    metadata_path = model_dir / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Model metadata saved to: {metadata_path}")


def main() -> None:
    """Main entry point."""
    try:
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        
        # Load and preprocess data
        print("Loading and preprocessing data...")
        df, pipeline = load_and_preprocess(CSV_PATH)
        print(f"Data shape: {df.shape}")
        print(f"Columns: {len(df.columns)}")
        
        # Split into X and Y
        X, Y = split_features_target(df)
        print(f"\nSplit into features (X) and target (Y):")
        print(f"  X shape: {X.shape}")
        print(f"  Y shape: {Y.shape}")
        print(f"  Y value counts:\n{Y.value_counts()}")
        
        # Train/validation split
        X_train, X_valid, Y_train, Y_valid = train_test_split(
            X, Y, test_size=0.2, random_state=42, stratify=Y
        )
        print("\nTrain/Validation split (valid_size=0.2):")
        print(f"  X_train shape: {X_train.shape}")
        print(f"  X_valid shape: {X_valid.shape}")
        print(f"  Y_train shape: {Y_train.shape}")
        print(f"  Y_valid shape: {Y_valid.shape}")
        print(f"  Y_train value counts:\n{Y_train.value_counts()}")
        print(f"  Y_valid value counts:\n{Y_valid.value_counts()}")
        
        # Train model
        print("\n" + "=" * 60)
        print("Training Model")
        print("=" * 60)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = train_model(
            X_train, Y_train, X_valid, Y_valid,
            epochs=50,
            batch_size=64,
            learning_rate=0.001,
            hidden_sizes=[128, 64, 32],
            dropout=0.3,
            device=device,
        )
        
        # Evaluate model
        metrics = evaluate_model(model, X_valid, Y_valid, device)
        
        # Save everything
        print("\n" + "=" * 60)
        print("Saving Model and Pipeline")
        print("=" * 60)
        save_pipeline(model, pipeline, MODEL_DIR, hidden_sizes=[128, 64, 32], dropout=0.3)
        
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)
        print(f"\nAll files saved to: {MODEL_DIR}")
        print("  - win_prediction_model.pt (model weights and config)")
        print("  - preprocessing_pipeline.joblib (preprocessing pipeline)")
        print("  - model_metadata.json (metadata for inspection)")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

