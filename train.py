import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

def generate_synthetic_data(samples=1000, random_seed=42):
    """
    Generates realistic synthetic placement data:
    - cgpa: 5.0 to 10.0
    - internships: 0 to 5
    - projects: 1 to 6
    - dsa_score: 30 to 100
    - communication_rating: 1 to 5
    """
    np.random.seed(random_seed)
    cgpa = np.random.uniform(5.5, 9.8, samples)
    internships = np.random.choice([0, 1, 2, 3, 4], size=samples, p=[0.25, 0.35, 0.25, 0.10, 0.05])
    projects = np.random.randint(1, 6, samples)
    dsa_score = np.random.uniform(40, 98, samples)
    comm_score = np.random.randint(1, 6, samples)
    
    # Placement probability heuristic
    logits = (
        0.55 * (cgpa - 7.0) +
        0.65 * internships +
        0.30 * projects +
        0.04 * (dsa_score - 60) +
        0.35 * (comm_score - 3)
    )
    prob = 1 / (1 + np.exp(-logits))
    placed = (prob > 0.50).astype(int)
    
    df = pd.DataFrame({
        "cgpa": np.round(cgpa, 2),
        "internships": internships,
        "projects": projects,
        "dsa_score": np.round(dsa_score, 1),
        "comm_score": comm_score,
        "placed": placed
    })
    return df

def train_and_export():
    df = generate_synthetic_data()
    
    X = df.drop(columns=["placed"])
    y = df["placed"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Production-ready Pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=120,
            max_depth=6,
            random_state=42
        ))
    ])
    
    print("Training model pipeline...")
    pipeline.fit(X_train, y_train)
    
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Validation Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, preds, target_names=["Not Placed", "Placed"]))
    
    # Serialize complete pipeline
    joblib.dump(pipeline, "model.joblib")
    print("Serialized pipeline saved as 'model.joblib'")

if __name__ == "__main__":
    train_and_export()