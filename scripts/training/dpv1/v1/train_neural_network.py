import pandas as pd
import numpy as np
import sys
import copy
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


DATA_DIR = Path("data/processed/v1")
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
    """
    Train neural network model.
    
    Args:
        X_train: Training features
        Y_train: Training targets
        X_valid: Validation features
        Y_valid: Validation targets
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        hidden_sizes: Hidden layer sizes
        dropout: Dropout probability
        device: Device to train on (CPU or CUDA)
    
    Returns:
        Trained model
    """
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
        
        # Early stopping - check if validation loss improved
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0
            # Save the best model state
            best_model_state = copy.deepcopy(model.state_dict())
            if (epoch + 1) % 5 != 0 and epoch != 0:
                print(f"Epoch [{epoch+1}/{epochs}] - New best validation loss: {valid_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                print(f"Best validation loss: {best_valid_loss:.4f}")
                # Restore best model
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
    """
    Evaluate model performance on validation set.
    
    Args:
        model: Trained model
        X_valid: Validation features
        Y_valid: Validation targets
        device: Device to use
    
    Returns:
        Dictionary with evaluation metrics
    """
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


def save_model(model: nn.Module, model_dir: Path) -> None:
    """
    Save trained model to disk.
    
    Args:
        model: Trained model
        model_dir: Directory to save the model
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "win_prediction_nn.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
    }, model_path)
    print(f"\nModel saved to: {model_path}")


def main() -> None:
    """Main entry point."""
    try:
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        
        # Load data
        print("Loading data...")
        X_train, X_valid, Y_train, Y_valid = load_data(DATA_DIR)
        print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"Validation set: {X_valid.shape[0]} samples")
        print(f"Target distribution (train):\n{Y_train.value_counts()}")
        
        # Train model
        model = train_model(
            X_train, Y_train, X_valid, Y_valid,
            epochs=50,
            batch_size=64,
            learning_rate=0.001,
            hidden_sizes=[128, 64, 32],
            dropout=0.3,
        )
        
        # Evaluate model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        metrics = evaluate_model(model, X_valid, Y_valid, device)
        
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

