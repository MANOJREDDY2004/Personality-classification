"""Train the final model on all 142 users and save it.
Run from the repo root:  python train_model.py
"""
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from textutil import clean

CSV_PATH = "../v1_original/dataset/english_dataset.csv"
TRUTH_PATH = "../v1_original/dataset/english_text/truth.txt"

df = pd.read_csv(CSV_PATH)
df = df[df["userid"] != "truth"]                      # drop bogus row
truth = pd.read_csv(TRUTH_PATH, sep=":::", header=None, engine="python")
truth.columns = ["userid", "gender", "age", "extro", "stable", "agree", "consc", "open"]
df = df.merge(truth[["userid", "extro"]], on="userid")

y = (df["extro"] >= df["extro"].median()).astype(int)   # 1 = higher extroversion

model = Pipeline([
    ("tfidf", TfidfVectorizer(preprocessor=clean, ngram_range=(1, 2), min_df=3,
                              sublinear_tf=True, max_features=20000)),
    ("knn", KNeighborsClassifier(n_neighbors=5)),
])
model.fit(df["text"], y)
joblib.dump(model, "personality_model.joblib")
print(f"Trained on {len(df)} users. Saved personality_model.joblib")
