from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import copy
from sklearn.model_selection import train_test_split


class MechanisticProbe(ABC):
    """
    An abstract base class for mechanistic binary probes.
    Probes are binary classifiers trained on the internal activations of a model
    to check for separable concepts (both linearly and nonlinearly separable).
    """
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.accuracy = 0.0
        self.is_trained = False

    def _validate_binary_labels(self, labels):
        """Validate that labels are binary (0 and 1)."""
        unique_labels = np.unique(labels)
        unique_set = set(unique_labels)
        
        # Check if labels are already valid binary (0, 1)
        if unique_set == {0, 1}:
            return np.array(labels)
        
        # Check if labels are single class (all 0s or all 1s)
        if unique_set == {0} or unique_set == {1}:
            return np.array(labels)
        
        # Auto-convert if exactly 2 unique values
        if len(unique_labels) == 2:
            label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
            converted_labels = np.array([label_map[label] for label in labels])
            print(f"Warning: Labels auto-converted from {unique_labels} to [0, 1]")
            return converted_labels
        else:
            raise ValueError(f"Labels must be binary (0 and 1). Found unique values: {unique_labels}")

    def _validate_input_dimensions(self, data, labels):
        """Validate input data and labels have consistent dimensions."""
        if len(data) != len(labels):
            raise ValueError(f"Data and labels must have the same length. Got {len(data)} samples and {len(labels)} labels")
        
        if len(data) == 0:
            raise ValueError("Cannot train on empty dataset")
        
        # Convert to numpy array if needed
        data = np.array(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        return data, labels

    @abstractmethod
    def train_probe(self, train_data, train_labels, **kwargs):
        """Trains the probe on the given data."""
        pass

    @abstractmethod
    def evaluate(self, test_data, test_labels, verbose=False):
        """Evaluates the probe on the test data, returning key classification metrics."""
        pass

    def plot_confusion_matrix(self, conf_matrix):
        """Helper function to plot a confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()

    @abstractmethod
    def save_probe(self, file_path):
        """Saves the probe's state to a file."""
        pass

    @classmethod
    @abstractmethod
    def load_probe(cls, file_path, map_location=None):
        """Loads a probe from a file."""
        pass


class LinearProbe(MechanisticProbe):
    """A linear binary probe implemented using a single-layer PyTorch model."""
    def __init__(self, input_size, device=None):
        super().__init__()
        
        if not isinstance(input_size, int) or input_size <= 0:
            raise ValueError(f"input_size must be a positive integer, got {input_size}")
            
        self.input_size = input_size
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"LinearProbe using device: {self.device}")
        
        # A linear probe is a single linear layer
        self.model = nn.Linear(self.input_size, 1)
        self.model.to(self.device)

    def train_probe(self, train_data, train_labels, num_epochs=500, lr=0.001, use_early_stopping=True, patience=10, validation_split=0.1, **kwargs):
        train_data, train_labels = self._validate_input_dimensions(train_data, train_labels)
        train_labels = self._validate_binary_labels(train_labels)

        if train_data.shape[1] != self.input_size:
            raise ValueError(f"Input data has {train_data.shape[1]} features, but model expects {self.input_size}")

        # --- Create validation set for early stopping ---
        if use_early_stopping:
            # Stratify to handle imbalanced datasets, handle cases with single class validation set
            stratify_labels = train_labels if np.min(np.bincount(train_labels)) > 1 else None
            X_train, X_val, y_train, y_val = train_test_split(
                train_data, train_labels, test_size=validation_split, random_state=42, stratify=stratify_labels
            )
        else:
            X_train, y_train = train_data, train_labels
            X_val, y_val = None, None

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32, device=self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=self.device)

        if X_val is not None and len(X_val) > 0:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32, device=self.device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32, device=self.device)
        else:
            use_early_stopping = False # Cannot do early stopping without validation data

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr) # Use Adam optimizer

        # --- Early Stopping Initializations ---
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None

        for epoch in range(num_epochs):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_train_tensor).squeeze(-1)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

            # --- Validation Check ---
            if use_early_stopping and (epoch + 1) % 10 == 0:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor).squeeze(-1)
                    val_loss = criterion(val_outputs, y_val_tensor)
                
                print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    best_model_state = copy.deepcopy(self.model.state_dict())
                else:
                    epochs_no_improve += 1
                
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    break
            elif (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
        self.is_trained = True

        if use_early_stopping and best_model_state is not None:
            print("Loading best model from early stopping.")
            self.model.load_state_dict(best_model_state)
        
        # Final training accuracy on the training split
        self.model.eval()
        with torch.no_grad():
            final_outputs = self.model(X_train_tensor).squeeze(-1)
            final_preds = (torch.sigmoid(final_outputs) > 0.5).float()
            final_train_accuracy = accuracy_score(y_train_tensor.cpu().numpy(), final_preds.cpu().numpy())
            print(f"Linear probe final training accuracy: {final_train_accuracy:.4f}")
        
        return final_train_accuracy

    def evaluate(self, test_data, test_labels, verbose=False):
        if not self.is_trained:
            raise ValueError("Probe must be trained before evaluation")
        
        if not hasattr(self.scaler, 'mean_'):
            raise ValueError("Scaler has not been fitted. Train the probe first.")
        
        test_data, test_labels = self._validate_input_dimensions(test_data, test_labels)
        test_labels = self._validate_binary_labels(test_labels)
        
        if test_data.shape[1] != self.input_size:
            raise ValueError(f"Input data has {test_data.shape[1]} features, but model expects {self.input_size}")

        self.model.eval()

        X_test_scaled = self.scaler.transform(test_data)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            outputs = self.model(X_test_tensor).squeeze(-1)
            probs_class1 = torch.sigmoid(outputs)
            predictions = (probs_class1 > 0.5).float()
        
        y_true_np = np.array(test_labels, dtype=int)
        y_pred_np = predictions.cpu().numpy().astype(int)

        probs_class0 = 1 - probs_class1
        probabilities = torch.stack([probs_class0, probs_class1], dim=1).cpu().numpy()

        accuracy = accuracy_score(y_true_np, y_pred_np)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_np, y_pred_np, average='binary', zero_division=0
        )
        conf_matrix = confusion_matrix(y_true_np, y_pred_np)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        self.accuracy = accuracy

        if verbose:
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1-Score: {metrics['f1_score']:.4f}")
            self.plot_confusion_matrix(conf_matrix)
        
        return metrics, conf_matrix, probabilities

    def save_probe(self, file_path):
        state = {
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'input_size': self.input_size,
            'accuracy': self.accuracy,
            'is_trained': self.is_trained
        }
        torch.save(state, file_path)
        print(f"LinearProbe saved to {file_path}")

    @classmethod
    def load_probe(cls, file_path, map_location=None):
        import pickle
        try:
            state = torch.load(file_path, map_location=map_location, weights_only=True)
        except (RuntimeError, TypeError, pickle.UnpicklingError) as e:
            if "weights_only" in str(e) or "Weights only load failed" in str(e):
                print(f"Warning: Using unsafe loading due to weights_only compatibility")
                state = torch.load(file_path, map_location=map_location, weights_only=False)
            else:
                raise e
        
        required_keys = ['model_state_dict', 'input_size']
        for key in required_keys:
            if key not in state:
                raise ValueError(f"Missing required key '{key}' in saved state")
        
        input_size = state['input_size']
        if not isinstance(input_size, int) or input_size <= 0:
            raise ValueError(f"Invalid input_size: {input_size}")
            
        probe = cls(input_size)
        
        try:
            probe.model.load_state_dict(state['model_state_dict'])
        except Exception as e:
            raise ValueError(f"Failed to load model state: {e}")
            
        probe.scaler = state.get('scaler', StandardScaler())
        probe.accuracy = state.get('accuracy', 0.0)
        probe.is_trained = state.get('is_trained', True)
        probe.model.eval()
        
        print(f"LinearProbe loaded from {file_path}")
        return probe


class MLPProbe(MechanisticProbe):
    """An MLP binary probe implemented using PyTorch."""
    def __init__(self, input_size, hidden_size=8, device=None):
        super().__init__()
        
        # Validate input parameters
        if not isinstance(input_size, int) or input_size <= 0:
            raise ValueError(f"input_size must be a positive integer, got {input_size}")
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError(f"hidden_size must be a positive integer, got {hidden_size}")
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Create the actual PyTorch model
        self.model = self._create_model()
        self.model.to(self.device)

    def _create_model(self):
        """Create the PyTorch model for binary classification."""
        return nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)  # Single output for binary classification
        )

    def train_probe(self, train_data, train_labels, num_epochs=500, lr=0.001, use_early_stopping=True, patience=10, validation_split=0.1, **kwargs):
        train_data, train_labels = self._validate_input_dimensions(train_data, train_labels)
        train_labels = self._validate_binary_labels(train_labels)
        
        if train_data.shape[1] != self.input_size:
            raise ValueError(f"Input data has {train_data.shape[1]} features, but model expects {self.input_size}")
        
        # --- Create validation set for early stopping ---
        if use_early_stopping:
            stratify_labels = train_labels if np.min(np.bincount(train_labels)) > 1 else None
            X_train, X_val, y_train, y_val = train_test_split(
                train_data, train_labels, test_size=validation_split, random_state=42, stratify=stratify_labels
            )
        else:
            X_train, y_train = train_data, train_labels
            X_val, y_val = None, None

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32, device=self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=self.device)

        if X_val is not None and len(X_val) > 0:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32, device=self.device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32, device=self.device)
        else:
            use_early_stopping = False

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr) # Use Adam optimizer

        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None

        for epoch in range(num_epochs):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_train_tensor).squeeze(-1)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

            # --- Validation or Logging Check ---
            if (epoch + 1) % 10 == 0:
                self.model.eval()
                with torch.no_grad():
                    # Get training accuracy for logging
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    train_acc = accuracy_score(y_train_tensor.cpu().numpy(), preds.cpu().numpy())

                    if use_early_stopping:
                        val_outputs = self.model(X_val_tensor).squeeze(-1)
                        val_loss = criterion(val_outputs, y_val_tensor)
                        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss.item():.4f}')

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            epochs_no_improve = 0
                            best_model_state = copy.deepcopy(self.model.state_dict())
                        else:
                            epochs_no_improve += 1
                        
                        if epochs_no_improve >= patience:
                            print(f"Early stopping triggered after {epoch + 1} epochs.")
                            break
                    else:
                        # Original logging if not early stopping
                        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Train Accuracy: {train_acc:.4f}')

        self.is_trained = True
        
        if use_early_stopping and best_model_state is not None:
            print("Loading best model from early stopping.")
            self.model.load_state_dict(best_model_state)

        # Calculate final training accuracy on the training split
        self.model.eval()
        with torch.no_grad():
            final_outputs = self.model(X_train_tensor).squeeze(-1)
            final_preds = (torch.sigmoid(final_outputs) > 0.5).float()
            final_train_accuracy = accuracy_score(y_train_tensor.cpu().numpy(), final_preds.cpu().numpy())
        
        print(f"MLP probe final training accuracy: {final_train_accuracy:.4f}")
        return final_train_accuracy

    def evaluate(self, test_data, test_labels, verbose=False):
        if not self.is_trained:
            raise ValueError("Probe must be trained before evaluation")
        
        # Check if scaler has been fitted
        if not hasattr(self.scaler, 'mean_'):
            raise ValueError("Scaler has not been fitted. Train the probe first.")
        
        test_data, test_labels = self._validate_input_dimensions(test_data, test_labels)
        test_labels = self._validate_binary_labels(test_labels)
        
        # Validate input dimensions match model expectations
        if test_data.shape[1] != self.input_size:
            raise ValueError(f"Input data has {test_data.shape[1]} features, but model expects {self.input_size}")
        
        self.model.eval()

        X_test_scaled = self.scaler.transform(test_data)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            outputs = self.model(X_test_tensor).squeeze(-1)
            probs_class1 = torch.sigmoid(outputs)
            predictions = (probs_class1 > 0.5).float()
        
        # Convert to numpy for sklearn metrics
        y_true_np = np.array(test_labels, dtype=int)
        y_pred_np = predictions.cpu().numpy().astype(int)

        # Create a 2-column probability array [prob_class_0, prob_class_1]
        probs_class0 = 1 - probs_class1
        probabilities = torch.stack([probs_class0, probs_class1], dim=1).cpu().numpy()

        # Calculate metrics
        accuracy = accuracy_score(y_true_np, y_pred_np)
        # Use zero_division=0 to handle cases with no predicted positives
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_np, y_pred_np, average='binary', zero_division=0
        )
        conf_matrix = confusion_matrix(y_true_np, y_pred_np)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        self.accuracy = accuracy  # Store for backward compatibility

        if verbose:
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1-Score: {metrics['f1_score']:.4f}")
            self.plot_confusion_matrix(conf_matrix)
        
        return metrics, conf_matrix, probabilities

    def save_probe(self, file_path):
        state = {
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'accuracy': self.accuracy,
            'is_trained': self.is_trained
        }
        torch.save(state, file_path)
        print(f"MLPProbe saved to {file_path}")

    @classmethod
    def load_probe(cls, file_path, map_location=None):
        import pickle
        try:
            # Use weights_only=True for security (safer loading)
            state = torch.load(file_path, map_location=map_location, weights_only=True)
        except (RuntimeError, TypeError, pickle.UnpicklingError) as e:
            # Only catch specific exceptions related to weights_only
            if "weights_only" in str(e) or "Weights only load failed" in str(e):
                print(f"Warning: Using unsafe loading due to weights_only compatibility")
                state = torch.load(file_path, map_location=map_location, weights_only=False)
            else:
                raise e
        
        # Validate that required keys exist
        required_keys = ['model_state_dict', 'input_size', 'hidden_size']
        for key in required_keys:
            if key not in state:
                raise ValueError(f"Missing required key '{key}' in saved state")
        
        # Validate architecture parameters
        input_size = state['input_size']
        hidden_size = state['hidden_size']
        
        if not isinstance(input_size, int) or input_size <= 0:
            raise ValueError(f"Invalid input_size: {input_size}")
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError(f"Invalid hidden_size: {hidden_size}")
        
        probe = cls(input_size, hidden_size)
        
        try:
            probe.model.load_state_dict(state['model_state_dict'])
        except Exception as e:
            raise ValueError(f"Failed to load model state: {e}")
        
        probe.scaler = state.get('scaler', StandardScaler())
        probe.accuracy = state.get('accuracy', 0.0)
        probe.is_trained = state.get('is_trained', True)
        probe.model.eval()
        
        print(f"MLPProbe loaded from {file_path}")
        return probe

    def to(self, device):
        """Move the model to a specific device."""
        device = torch.device(device)
        
        # Only move if device is different
        if self.device != device:
            self.device = device
            self.model = self.model.to(device)
            
            # Clear any cached tensors that might be on the wrong device
            if hasattr(self, '_cached_tensors'):
                delattr(self, '_cached_tensors')
        
        return self
