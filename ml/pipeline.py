"""ML pipeline: 1 Data Acquisition -> 2 Data Preparation -> 3 Model Selection -> 4 Model Evaluation."""
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import DATA, DATASETS, EYE_COLS, HEART_COLS, KIDNEY_COLS, MODELS, REPORTS

warnings.filterwarnings("ignore")
SEED = 42


# ---------------------------------------------------------------- 1. DATA ACQUISITION
def _load_heart():
    frames = []
    for site in ("cleveland", "hungarian", "switzerland", "va"):
        f = pd.read_csv(DATA / "heart" / f"processed.{site}.data", header=None,
                        names=HEART_COLS, na_values="?")
        f["site"] = site
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def _load_kidney():
    """Parse the ARFF by hand: the file has stray tabs, blanks and 2 rows with a trailing comma."""
    rows = []
    for line in open(DATA / "kidney" / "chronic_kidney_disease.arff", encoding="latin-1"):
        line = line.strip()
        if not line or line[0] in "@%":
            continue
        vals = [v.strip() for v in line.split(",")]
        if len(vals) == 26:                       # spurious empty field
            vals = [v for v in vals if v != ""]
        rows.append(vals)
    df = pd.DataFrame(rows, columns=KIDNEY_COLS + ["class"]).replace("?", np.nan)
    return df


def _load_eye():
    rows = [[v.strip() for v in l.split(",")] for l in open(DATA / "eye" / "messidor_features.arff")
            if l.strip() and l.strip()[0] not in "@%"]
    return pd.DataFrame(rows, columns=EYE_COLS + ["class"]).apply(pd.to_numeric, errors="coerce").dropna()


def acquire(key):
    df = {"heart": _load_heart, "kidney": _load_kidney, "eye": _load_eye}[key]()
    feats = [c for c in df.columns if c not in ("num", "class", "site")]
    miss = (df[feats].isna().mean() * 100).round(1).sort_values(ascending=False)
    label = {"heart": lambda: df["num"].gt(0), "kidney": lambda: df["class"].eq("ckd"),
             "eye": lambda: df["class"].eq(1)}[key]()
    summary = dict(
        source=DATASETS[key]["source"], rows=int(len(df)), features=len(feats),
        positives=int(label.sum()), negatives=int((~label).sum()),
        sites=df["site"].value_counts().to_dict() if key == "heart" else None,
        missing_pct={k: float(v) for k, v in miss.head(6).items() if v > 0},
        missing_cells_pct=round(float(df[feats].isna().mean().mean() * 100), 1),
    )
    return df, summary


# ---------------------------------------------------------------- 2. DATA PREPARATION
def prepare(key, df):
    cfg, notes = DATASETS[key], []
    df = df.copy()
    if key == "heart":
        y = df.pop("num").gt(0).astype(int)
        df = df.drop(columns="site")
        n0 = df[["chol", "trestbps"]].eq(0).sum().sum()
        df[["chol", "trestbps"]] = df[["chol", "trestbps"]].replace(0, np.nan)
        notes += ["Merged the 4 hospital files (Cleveland, Hungarian, Switzerland, VA) into one table of "
                  f"{len(df)} patients.",
                  "Target 'num' (0-4 vessel narrowing) converted to binary: 0 = no disease, 1-4 = disease.",
                  f"{n0} impossible zero readings of cholesterol / blood pressure treated as missing."]
        for c in cfg["cat_cols"]:
            df[c] = df[c].map(lambda v: str(int(v)) if pd.notna(v) else np.nan)
    elif key == "eye":
        y = df.pop("class").astype(int)
        notes += ["Image-derived features of the Messidor retinal dataset (microaneurysm counts at 6 confidence "
                  "levels, 8 exudate measures, optic-disc geometry, image quality and pre-screening flags).",
                  "Target 'Class' kept as binary: 1 = signs of diabetic retinopathy, 0 = none. No missing values."]
    else:
        y = df.pop("class").eq("ckd").astype(int)
        notes += ["Parsed the ARFF manually and repaired malformed rows (stray tabs, blank fields, trailing commas).",
                  "Target 'class' converted to binary: ckd = 1, notckd = 0."]
    for c in cfg["num_cols"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    dup = df.assign(_y=y).duplicated().sum()
    keep = ~df.assign(_y=y).duplicated()
    df, y = df[keep].reset_index(drop=True), y[keep].reset_index(drop=True)
    notes.append(f"Removed {dup} duplicate record(s); {len(df)} rows remain.")
    notes += ["Missing numeric values -> median; missing categorical values -> most frequent value "
              "(no 'missing' flag is kept, so a blank form field cannot act as a signal of its own).",
              "Numeric features standardised (z-score); categorical features one-hot encoded. All fitted "
              "inside a scikit-learn Pipeline on the training data only, so no test-set leakage.",
              "Stratified 80/20 train/test split (seed 42)."]
    parts = [("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), cfg["num_cols"])]
    if cfg["cat_cols"]:
        parts.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                       ("oh", OneHotEncoder(handle_unknown="ignore"))]), cfg["cat_cols"]))
    pre = ColumnTransformer(parts)
    X = df[cfg["num_cols"] + cfg["cat_cols"]]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    return Xtr, Xte, ytr, yte, pre, notes


# ---------------------------------------------------------------- 3. MODEL SELECTION
def candidates():
    return {
        "Logistic Regression": (LogisticRegression(max_iter=2000), {"C": [0.1, 1, 10]}),
        "K-Nearest Neighbours": (KNeighborsClassifier(), {"n_neighbors": [5, 9, 15]}),
        "Support Vector Machine": (SVC(probability=True, random_state=SEED), {"C": [0.5, 1, 5]}),
        "Decision Tree": (DecisionTreeClassifier(random_state=SEED), {"max_depth": [3, 5, None]}),
        "Random Forest": (RandomForestClassifier(random_state=SEED),
                          {"n_estimators": [200, 400], "max_depth": [None, 6]}),
        "Gradient Boosting": (GradientBoostingClassifier(random_state=SEED),
                              {"learning_rate": [0.05, 0.1], "n_estimators": [100, 200], "max_depth": [2, 3]}),
    }


def select_model(pre, Xtr, ytr):
    """5-fold stratified CV + grid search for every candidate; winner = best CV ROC-AUC."""
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    scoring = {"roc_auc": "roc_auc", "accuracy": "accuracy", "f1": "f1", "recall": "recall"}
    table, best = [], (None, -1, None)
    for name, (est, grid) in candidates().items():
        pipe = Pipeline([("pre", pre), ("clf", est)])
        gs = GridSearchCV(pipe, {f"clf__{k}": v for k, v in grid.items()}, scoring=scoring,
                          refit="roc_auc", cv=cv, n_jobs=-1).fit(Xtr, ytr)
        i = gs.best_index_
        row = dict(model=name, params={k[5:]: v for k, v in gs.best_params_.items()},
                   **{m: round(float(gs.cv_results_[f"mean_test_{m}"][i]), 4) for m in scoring},
                   roc_auc_std=round(float(gs.cv_results_["std_test_roc_auc"][i]), 4))
        table.append(row)
        print(f"   {name:24s} CV AUC={row['roc_auc']:.3f} acc={row['accuracy']:.3f} recall={row['recall']:.3f}")
        if row["roc_auc"] > best[1]:
            best = (name, row["roc_auc"], gs.best_estimator_)
    for r in table:
        r["selected"] = r["model"] == best[0]
    return best[0], best[2], sorted(table, key=lambda r: -r["roc_auc"])


# ---------------------------------------------------------------- 4. MODEL EVALUATION
def evaluate(key, model, name, Xte, yte):
    prob = model.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(int)
    cm = confusion_matrix(yte, pred)
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    m = dict(accuracy=accuracy_score(yte, pred), precision=precision_score(yte, pred),
             recall=recall_score(yte, pred), specificity=tn / (tn + fp),
             f1=f1_score(yte, pred), roc_auc=roc_auc_score(yte, prob))
    m = {k: round(float(v), 4) for k, v in m.items()}
    # permutation importance (on raw input features)
    pi = permutation_importance(model, Xte, yte, scoring="roc_auc", n_repeats=20, random_state=SEED)
    imp = pd.Series(pi.importances_mean, index=Xte.columns).sort_values().tail(10)
    return m, dict(tn=tn, fp=fp, fn=fn, tp=tp), {k: round(float(v), 4) for k, v in imp[::-1].items()}


def healthy_ranges(key, Xtr, ytr, imp):
    """Eye features have no clinical 'normal': use the middle 50% of non-retinopathy training patients."""
    if key != "eye":
        return {}
    fields = [f["name"] for f in DATASETS[key]["fields"] if f["kind"] == "num"]
    ranked = [f for f in imp if f in fields]       # most influential first
    top = [f for f in ranked if f.startswith("ma")][:3] + [f for f in ranked if f.startswith("ex")][:3]
    h = Xtr[ytr == 0]
    return {f: [float(h[f].quantile(.25)), float(h[f].quantile(.75)),
                float(Xtr[f].quantile(.01)), float(Xtr[f].quantile(.99))] for f in top}


def examples(key, Xte, yte):
    """One real positive and one real negative test patient, to pre-fill the web form."""
    cfg, out = DATASETS[key], {}
    req = [f["name"] for f in cfg["fields"] if f["required"]]
    for label, name in ((1, "positive"), (0, "negative")):
        sub = Xte[(yte == label) & Xte[req].notna().all(axis=1)]
        if sub.empty:
            continue
        row, vals = sub.iloc[0], {}
        for f in cfg["fields"]:
            v = row[f["name"]]
            if pd.isna(v):
                vals[f["name"]] = ""
            elif f["kind"] == "select" and f.get("dtype") == "num":
                vals[f["name"]] = next((o for o, _ in f["options"] if abs(float(o) - float(v)) < 1e-9), "")
            elif f["kind"] == "num":
                vals[f["name"]] = f"{v:.6g}"
            else:
                vals[f["name"]] = str(v)
        out[name] = vals
    return out


# ---------------------------------------------------------------- ORCHESTRATION
def run(key):
    print(f"\n=== {DATASETS[key]['title']} ===")
    print("1. Data acquisition")
    df, acq = acquire(key)
    print(f"   {acq['rows']} rows, {acq['features']} features, {acq['positives']} positive")
    print("2. Data preparation")
    Xtr, Xte, ytr, yte, pre, notes = prepare(key, df)
    print(f"   train={len(Xtr)} test={len(Xte)}")
    print("3. Model selection (5-fold CV + grid search)")
    name, model, table = select_model(pre, Xtr, ytr)
    print(f"   -> selected: {name}")
    print("4. Model evaluation (held-out test set)")
    metrics, cm, imp = evaluate(key, model, name, Xte, yte)
    print("  ", metrics)
    MODELS.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)
    joblib.dump(model, MODELS / f"{key}.joblib")
    report = dict(key=key, title=DATASETS[key]["title"], acquisition=acq, preparation=notes,
                  split=dict(train=len(Xtr), test=len(Xte)), selection=table, selected=name,
                  metrics=metrics, confusion=cm, importance=imp,
                  examples=examples(key, Xte, yte),
                  healthy_ranges=healthy_ranges(key, Xtr, ytr, imp))
    (REPORTS / f"{key}.json").write_text(json.dumps(report, indent=2))
    return report
