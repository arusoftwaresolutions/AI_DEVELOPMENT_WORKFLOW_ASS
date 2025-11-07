import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def synth_student_data(n=2000, random_state=0):
    rng = np.random.RandomState(random_state)
    X = pd.DataFrame({
        "gpa": rng.normal(3.0, 0.5, size=n).clip(0,4),
        "attendance": rng.uniform(50,100,size=n),
        "assignments_submitted": rng.poisson(8, size=n),
        "financial_aid_flag": rng.binomial(1, 0.3, size=n),
        "lms_logins_last_30d": rng.poisson(10, size=n),
    })
    logits = (-1.5 + -1.0*X["gpa"] + -0.02*X["attendance"] + -0.1*X["assignments_submitted"]
              + 0.8*X["financial_aid_flag"] -0.01*X["lms_logins_last_30d"])
    prob = 1/(1+np.exp(-logits))
    y = (rng.rand(n) < prob).astype(int)
    return X, y

def synth_readmission_data(n=2000, random_state=0):
    rng = np.random.RandomState(random_state)
    X = pd.DataFrame({
        "age": rng.randint(18, 90, size=n),
        "num_prev_adm_12mo": rng.poisson(0.5, size=n),
        "charlson_index": rng.poisson(1.0, size=n),
        "last_creatinine": np.round(rng.normal(1.0,0.4,size=n),3).clip(0.4,10),
        "length_of_stay": rng.poisson(4, size=n)+1,
    })
    logits = (-3.0 + 0.03*X["age"] + 0.8*X["num_prev_adm_12mo"] + 0.4*X["charlson_index"]
              + 0.5*(X["last_creatinine"]-1.0) + 0.05*X["length_of_stay"])
    prob = 1/(1+np.exp(-logits))
    y = (rng.rand(n) < prob).astype(int)
    return X, y

def train_student_model(X, y, save_name="student_rf.joblib", random_state=0):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    joblib.dump(model, MODELS_DIR / save_name)
    return {"model_path": str(MODELS_DIR / save_name), "report": classification_report(y_test, preds, output_dict=True)}

def train_readmission_model(X, y, save_name="readmit_gb.joblib", random_state=0):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=random_state)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    joblib.dump(model, MODELS_DIR / save_name)
    return {"model_path": str(MODELS_DIR / save_name), "report": classification_report(y_test, preds, output_dict=True)}

def load_model(path):
    return joblib.load(path)

def predict(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:,1]
    return model.predict(X)