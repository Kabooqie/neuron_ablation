import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
X_PATH = "X_robust.npy"
Y_PATH = "y_robust.npy"
OUTPUT_INDICES = "malice_indices.npy"


def train_sparse_probe(X_path, y_path, k_folds=5):
    # 1. Load Data
    if not os.path.exists(X_path):
        print(f"Error: {X_path} not found. Run harvester.py first.")
        return

    X = np.load(X_path)
    y = np.load(y_path)

    # 2. Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Cross-Validation (The Paper's Statistical Proof)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    scores = []

    print(f"Running {k_folds}-Fold Cross Validation...")
    for train_idx, test_idx in kf.split(X_scaled):
        # L1 penalty induces sparsity to find the "Malice Neurons"
        clf = LogisticRegression(penalty='l1', solver='liblinear', C=0.5)
        clf.fit(X_scaled[train_idx], y[train_idx])
        scores.append(clf.score(X_scaled[test_idx], y[test_idx]))

    print(f"Mean Accuracy: {np.mean(scores) * 100:.2f}% (+/- {np.std(scores) * 100:.2f}%)")

    # 4. Final Fit to save the Indices
    final_clf = LogisticRegression(penalty='l1', solver='liblinear', C=0.5)
    final_clf.fit(X_scaled, y)

    # Identify the "Malice Neurons" (features with non-zero weights)
    malice_indices = np.where(final_clf.coef_[0] != 0)[0]
    np.save(OUTPUT_INDICES, malice_indices)

    print(f"Malice-relevant neurons identified and saved: {len(malice_indices)}")


if __name__ == "__main__":
    train_sparse_probe(X_PATH, Y_PATH)