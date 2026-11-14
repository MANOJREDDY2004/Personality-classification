"""
v3: handling real class imbalance on the Kaggle MBTI dataset (Introvert vs Extrovert).

Setup
    pip install scikit-learn imbalanced-learn pandas numpy
    put mbti_1.csv (columns: type, posts) at v3_imbalance/data/mbti_1.csv
Run from the v3_imbalance folder:
    python v3_imbalance.py

What it does
  - Target: first letter of the MBTI type. POSITIVE class = Extrovert (the minority).
  - Strips label leakage (the 16 type codes and plurals, URLs) from the posts.
  - 5-fold stratified CV. TF-IDF is fit once per fold on the training part only.
  - Compares: majority baseline, plain LogReg, class-weighted LogReg,
    random oversampling, (optional) SMOTE, and class-weighted LogReg with a
    decision threshold tuned on a held-out slice of the TRAINING fold.
  - Optional leakage demo: the same model without stripping type codes.
Reported: accuracy, macro-F1, minority precision / recall / F1, PR-AUC, ROC-AUC.
"""
import re
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from imblearn.over_sampling import RandomOverSampler, SMOTE

warnings.filterwarnings("ignore")

# ---- settings ---------------------------------------------------------------
CSV_PATH = "data/mbti_1.csv"
RUN_SMOTE = False        # slow on big sparse matrices; set True if you have time
LEAKAGE_DEMO = True      # also run the model WITHOUT stripping type codes
MAX_ROWS = None          # e.g. 2000 for a quick test run
SEED = 42
# -----------------------------------------------------------------------------

TYPES = ["infj", "infp", "intj", "intp", "isfj", "isfp", "istj", "istp",
         "enfj", "enfp", "entj", "entp", "esfj", "esfp", "estj", "estp"]
TYPE_RE = re.compile(r"\b(?:" + "|".join(TYPES) + r")s?\b", re.I)
URL_RE = re.compile(r"http\S+|www\S+")
# "Strict" mode also removes community jargon that gives the label away indirectly:
# cognitive functions (Ni, Ne, Fi...), enneagram wings (7w8) and instincts (sx), and test names.
JARGON_RE = re.compile(
    r"\b(?:ni|ne|si|se|ti|te|fi|fe|sx|sp|mbti|enneagram|myers|briggs|socionics|"
    r"inferior|dominant|auxiliary|tertiary)\b|\b[1-9]w[1-9]\b", re.I)


def prep(posts: str, strip_types: bool, strict: bool = False) -> str:
    s = posts.replace("|||", " ")
    s = URL_RE.sub(" ", s)
    if strip_types:
        s = TYPE_RE.sub(" ", s)
    if strict:
        s = JARGON_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s.lower()).strip()


df = pd.read_csv(CSV_PATH).dropna(subset=["type", "posts"])
if MAX_ROWS:
    df = df.sample(MAX_ROWS, random_state=SEED)
y = (df["type"].str[0].str.upper() == "E").astype(int).values   # 1 = Extrovert (minority)
print(f"n = {len(df)} | extrovert share = {y.mean():.3f} | introvert share = {1 - y.mean():.3f}")
leak_rate = df["posts"].str.contains(TYPE_RE).mean()
print(f"users whose posts contain a type code like 'INFJ': {leak_rate:.1%}\n")

texts_clean = df["posts"].map(lambda s: prep(s, True)).values
texts_leaky = df["posts"].map(lambda s: prep(s, False)).values
texts_strict = df["posts"].map(lambda s: prep(s, True, True)).values


def metrics(y_true, pred, score):
    return {
        "accuracy": accuracy_score(y_true, pred),
        "macro_f1": f1_score(y_true, pred, average="macro"),
        "min_prec": precision_score(y_true, pred, zero_division=0),
        "min_rec": recall_score(y_true, pred, zero_division=0),
        "min_f1": f1_score(y_true, pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, score) if score is not None else np.nan,
        "roc_auc": roc_auc_score(y_true, score) if score is not None else np.nan,
    }


def lr(class_weight=None):
    return LogisticRegression(C=1.0, max_iter=1000, class_weight=class_weight)


def best_threshold(model, Xva, yva):
    s = model.predict_proba(Xva)[:, 1]
    grid = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(yva, (s >= t).astype(int), zero_division=0) for t in grid]
    return grid[int(np.argmax(f1s))]


def run_cv(texts, include_all=True):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for fold, (tr, te) in enumerate(skf.split(texts, y), 1):
        t0 = time.time()
        vec = TfidfVectorizer(min_df=5, max_features=20000, sublinear_tf=True,
                              stop_words="english")
        Xtr = vec.fit_transform(texts[tr])
        Xte = vec.transform(texts[te])
        ytr, yte = y[tr], y[te]

        def add(name, pred, score):
            rows.append({"model": name, "fold": fold, **metrics(yte, pred, score)})

        if include_all:
            add("Baseline (always introvert)", np.zeros_like(yte), None)

        m = lr().fit(Xtr, ytr)
        add("LogReg (no imbalance handling)", m.predict(Xte), m.predict_proba(Xte)[:, 1])

        m = lr("balanced").fit(Xtr, ytr)
        add("LogReg class_weight=balanced", m.predict(Xte), m.predict_proba(Xte)[:, 1])

        if include_all:
            Xo, yo = RandomOverSampler(random_state=SEED).fit_resample(Xtr, ytr)
            m = lr().fit(Xo, yo)
            add("Random oversampling + LogReg", m.predict(Xte), m.predict_proba(Xte)[:, 1])

            if RUN_SMOTE:
                Xs, ys = SMOTE(random_state=SEED).fit_resample(Xtr, ytr)
                m = lr().fit(Xs, ys)
                add("SMOTE + LogReg", m.predict(Xte), m.predict_proba(Xte)[:, 1])

            # threshold tuned on a validation slice of the TRAINING fold only
            Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.2,
                                              stratify=ytr, random_state=SEED)
            mv = lr("balanced").fit(Xa, ya)
            thr = best_threshold(mv, Xv, yv)
            m = lr("balanced").fit(Xtr, ytr)
            s = m.predict_proba(Xte)[:, 1]
            add(f"Balanced LogReg + tuned threshold", (s >= thr).astype(int), s)
        print(f"fold {fold}/5 done in {time.time() - t0:.0f}s")
    res = pd.DataFrame(rows)
    return res.groupby("model").mean(numeric_only=True).drop(columns="fold")


print("== Main experiment (type codes stripped) ==")
table = run_cv(texts_clean).round(3).sort_values("macro_f1", ascending=False)
print(table.to_string())
table.to_csv("results_v3.csv")

if LEAKAGE_DEMO:
    print("\n== Leakage demo: balanced LogReg at three levels of cleaning ==")
    key = "LogReg class_weight=balanced"
    leaky = run_cv(texts_leaky, include_all=False).loc[[key]].round(3)
    strict = run_cv(texts_strict, include_all=False).loc[[key]].round(3)
    clean = table.loc[[key]]
    demo = pd.concat([leaky.assign(version="1) raw posts (type codes present)"),
                      clean.assign(version="2) type codes stripped"),
                      strict.assign(version="3) + jargon stripped (strict)")]).set_index("version")
    print(demo[["accuracy", "macro_f1", "min_rec", "pr_auc", "roc_auc"]].to_string())
    demo.to_csv("results_v3_leakage_demo.csv")

# Words driving the Extrovert prediction AFTER strict cleaning (fit once on all data, for illustration)
vec = TfidfVectorizer(min_df=5, max_features=20000, sublinear_tf=True, stop_words="english")
X = vec.fit_transform(texts_strict)
m = lr("balanced").fit(X, y)
vocab = np.array(vec.get_feature_names_out())
order = np.argsort(m.coef_[0])
print("\nWords pushing toward INTROVERT:", ", ".join(vocab[order[:15]]))
print("Words pushing toward EXTROVERT:", ", ".join(vocab[order[-15:][::-1]]))
print("\nPick the best model by macro-F1 / minority recall / PR-AUC, not accuracy.")
