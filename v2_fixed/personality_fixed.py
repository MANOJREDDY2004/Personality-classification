"""
Fixed evaluation for the PAN15 personality project (English).

Run from the root of your repo (the folder that contains `dataset/`):
    pip install scikit-learn pandas scipy
    python personality_fixed.py

What it fixes compared to the original notebook
  1. Drops the bogus row (truth.txt itself was read in as a "user", label 'unknown').
  2. Re-labels from the raw extroversion score. The original rule (score >= 0.5)
     gave only 10 "extroverts" out of 142 because scores range -0.3..0.5
     (median 0.2). Here: median split -> "higher" vs "lower" extroversion.
  3. Uses repeated stratified CV instead of one 29-sample test split.
  4. Reports a majority-class baseline, macro-F1 and ROC-AUC next to accuracy.
  5. Adds a regression check on the raw score (Pearson r, RMSE vs. mean baseline).
"""
import re
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import (RepeatedKFold, RepeatedStratifiedKFold,
                                     cross_val_predict, cross_validate)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

# ---- paths (edit if your folders differ) -----------------------------------
CSV_PATH = "../v1_original/dataset/english_dataset.csv"
TRUTH_PATH = "../v1_original/dataset/english_text/truth.txt"    # PAN15 truth file
# ----------------------------------------------------------------------------

# ---------- load + clean -----------------------------------------------------
df = pd.read_csv(CSV_PATH)
df = df[df["userid"] != "truth"].copy()           # bug fix: bogus 'truth.txt' row

truth = pd.read_csv(TRUTH_PATH, sep=":::", header=None, engine="python")
truth.columns = ["userid", "gender", "age", "extro", "stable", "agree", "consc", "open"]
df = df.merge(truth[["userid", "extro"]], on="userid", how="inner")


def clean(s: str) -> str:
    s = re.sub(r"@\w+|http\S+|www\S+", " ", s.lower())
    return re.sub(r"\s+", " ", s)


df["x"] = df["text"].astype(str).map(clean)
print(f"n users = {len(df)}  (original notebook used 143 rows incl. 1 bogus row)")
print(f"extroversion score: min={df.extro.min()}  median={df.extro.median()}  max={df.extro.max()}")
print(f"original rule (score>=0.5): {(df.extro >= 0.5).sum()} 'extroverts' of {len(df)}\n")

# ---------- classification: median split ------------------------------------
y = (df["extro"] >= df["extro"].median()).astype(int)
print(f"median split -> high extroversion = {y.sum()}, lower = {(1 - y).sum()}\n")

word = lambda: TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, max_features=20000)
char = lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                               sublinear_tf=True, max_features=40000)
models = {
    "Baseline (majority class)": Pipeline([("c", DummyClassifier(strategy="most_frequent"))]),
    "KNN k=5, word TF-IDF":      Pipeline([("t", word()), ("c", KNeighborsClassifier(5))]),
    "LogReg, word TF-IDF":       Pipeline([("t", word()), ("c", LogisticRegression(C=1, max_iter=2000))]),
    "LogReg, char TF-IDF":       Pipeline([("t", char()), ("c", LogisticRegression(C=1, max_iter=2000))]),
    "LinearSVC, word TF-IDF":    Pipeline([("t", word()), ("c", LinearSVC(C=0.5))]),
}
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
rows = []
for name, m in models.items():
    scoring = {"acc": "accuracy", "f1m": "f1_macro"}
    if "Baseline" not in name:
        scoring["auc"] = "roc_auc"
    r = cross_validate(m, df["x"], y, cv=cv, scoring=scoring)
    rows.append({"model": name,
                 "accuracy": r["test_acc"].mean(), "acc_std": r["test_acc"].std(),
                 "macro_f1": r["test_f1m"].mean(),
                 "roc_auc": r["test_auc"].mean() if "test_auc" in r else np.nan})
table = pd.DataFrame(rows).set_index("model").round(3)
print("== Classification (5-fold CV x3 repeats) ==")
print(table.to_string())
table.to_csv("results_classification.csv")

# ---------- regression on the raw score -------------------------------------
print("\n== Regression on raw extroversion score ==")
rcv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
reg = {"Baseline (mean)": Pipeline([("c", DummyRegressor())]),
       "Ridge, word TF-IDF": Pipeline([("t", word()), ("c", Ridge(alpha=3))]),
       "Ridge, char TF-IDF": Pipeline([("t", char()), ("c", Ridge(alpha=3))])}
rrows = []
for name, m in reg.items():
    r = cross_validate(m, df["x"], df["extro"], cv=rcv, scoring={"rmse": "neg_root_mean_squared_error"})
    pred = cross_val_predict(m, df["x"], df["extro"], cv=5)
    rr = pearsonr(pred, df["extro"])[0] if np.std(pred) > 0 else np.nan
    rrows.append({"model": name, "rmse": -r["test_rmse"].mean(), "pearson_r": rr})
rtable = pd.DataFrame(rrows).set_index("model").round(3)
print(rtable.to_string())
rtable.to_csv("results_regression.csv")

# ---------- what words drive the prediction? (interpretability) -------------
vec = TfidfVectorizer(min_df=5, sublinear_tf=True, stop_words="english")
Xv = vec.fit_transform(df["x"])
lr = LogisticRegression(C=1, max_iter=2000).fit(Xv, y)
vocab = np.array(vec.get_feature_names_out())
order = np.argsort(lr.coef_[0])
print("\nWords pushing toward LOWER extroversion:", ", ".join(vocab[order[:15]]))
print("Words pushing toward HIGHER extroversion:", ", ".join(vocab[order[-15:][::-1]]))
print("\n(n=142: treat these words as hints, not findings.)")
