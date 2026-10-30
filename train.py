import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, brier_score_loss
import joblib

def generate_realistic_data(samples=3000, random_seed=42):
    """
    Generates realistic campus placement data with multi-tier corporate criteria:
    - cgpa: Realistic bell curve centered around 7.2
    - internships: Right-skewed distribution (most have 0-1)
    - projects: 0 to 5
    - dsa_score: Technical assessment score (0 to 100)
    - comm_score: Communication rating (1 to 5)
    - backlogs: Active academic backlogs (0 to 4)
    """
    np.random.seed(random_seed)
    
    # 1. Feature distributions matching engineering campus cohorts
    cgpa = np.clip(np.round(np.random.normal(loc=7.1, scale=1.1, size=samples), 2), 5.0, 9.9)
    internships = np.random.choice([0, 1, 2, 3, 4], size=samples, p=[0.48, 0.32, 0.13, 0.05, 0.02])
    projects = np.random.choice([0, 1, 2, 3, 4, 5], size=samples, p=[0.10, 0.25, 0.35, 0.20, 0.08, 0.02])
    dsa_score = np.clip(np.round(np.random.normal(loc=54.0, scale=18.0, size=samples), 1), 10.0, 100.0)
    comm_score = np.random.choice([1, 2, 3, 4, 5], size=samples, p=[0.12, 0.28, 0.38, 0.16, 0.06])
    backlogs = np.random.choice([0, 1, 2, 3, 4], size=samples, p=[0.72, 0.16, 0.07, 0.03, 0.02])
    
    # 2. Realistic Log-odds with corporate gating rules
    z = (
        1.25 * (cgpa - 7.0)
        + 0.65 * (internships - 0.5)
        + 0.40 * (projects - 2.0)
        + 0.055 * (dsa_score - 55.0)
        + 0.45 * (comm_score - 3.0)
        - 1.45 * backlogs
        - 0.55  # Neutral offset ensuring median applicant isn't trivially predicted as Placed
    )
    
    # Strict penalties: Most tier-1/tier-2 recruiters enforce 60% (6.0/6.5 CGPA) criteria and technical cutoffs
    z[cgpa < 6.0] -= 1.85
    z[dsa_score < 40.0] -= 1.30
    z[backlogs >= 2] -= 1.50
    
    # Sigmoidal ground-truth probability
    true_prob = 1.0 / (1.0 + np.exp(-z))
    placed = (np.random.rand(samples) < true_prob).astype(int)
    
    df = pd.DataFrame({
        "cgpa": cgpa,
        "internships": internships,
        "projects": projects,
        "dsa_score": dsa_score,
        "comm_score": comm_score,
        "backlogs": backlogs,
        "placed": placed
    })
    return df

def train_and_export():
    df = generate_realistic_data()
    print(f"Dataset generated ({len(df)} samples). Overall placement rate: {df['placed'].mean():.1%}")
    
    X = df.drop(columns=["placed"])
    y = df["placed"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Base Random Forest classifier with conservative leaf depth to avoid overfitting
    base_rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        min_samples_leaf=8,
        random_state=42
    )
    
    # Calibrated Classifier: adjusts output probabilities so they reflect real likelihood
    calibrated_clf = CalibratedClassifierCV(estimator=base_rf, method="sigmoid", cv=5)
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", calibrated_clf)
    ])
    
    print("Training calibrated ML pipeline...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    test_probs = pipeline.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)
    
    print("\nModel Evaluation Metrics:")
    print(classification_report(y_test, test_preds, target_names=["Not Placed", "Placed"]))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, test_probs):.4f}")
    print(f"Brier Score (lower is better, 0 = perfect calibration): {brier_score_loss(y_test, test_probs):.4f}")
    
    # Save the pipeline
    joblib.dump(pipeline, "model.joblib")
    print("Saved calibrated pipeline as 'model.joblib'")

if __name__ == "__main__":
    train_and_export()