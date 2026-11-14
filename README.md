# Personality classification from text: an audit and a rebuild

This repo started as a 4-member university research project (TF-IDF + KNN on PAN Author Profiling data)
that reported **93.1% accuracy**. On re-checking, that number equalled the majority-class baseline.
This repo keeps the original work, explains what went wrong, and rebuilds it with sound evaluation.

| Folder | What it is |
|---|---|
| [`v1_original/`](v1_original) | The team's original code and data, exactly as submitted (unchanged on purpose). |
| [`POSTMORTEM.md`](POSTMORTEM.md) | Why the 93.1% was meaningless: a label threshold that created the imbalance, a bogus data row, and no baseline. |
| [`v2_fixed/`](v2_fixed) | Corrected labels and evaluation on the same data (n=142), plus a Streamlit demo. |
| [`v3_imbalance/`](v3_imbalance) | A larger, naturally imbalanced dataset (8,675 users): imbalance handling and label-leakage analysis. |

## Headline results
**v2 (PAN15 English, n=142, median-split labels, repeated 5-fold CV):** KNN reached 72.3% accuracy and
ROC-AUC 0.81, against a 51.4% majority baseline.

**v3 (MBTI forum posts, n=8,675, 77/23 imbalance, 5-fold CV):**
- 95.9% of users write their own type code in their posts. Leaving it in inflates macro-F1 from about 0.68 to 0.78.
- With leakage removed, balanced logistic regression gets macro-F1 0.68 vs. 0.44 for "always introvert",
  with extrovert recall of 57%. Plain logistic regression had higher accuracy but found only 5% of extroverts.

## What I learned
Check the label distribution before choosing a threshold, always report a majority baseline next to
accuracy, and look for leakage when a result seems too good.

## Run
```
pip install -r v2_fixed/requirements.txt
cd v2_fixed && python personality_fixed.py      # reproduces the v2 table
cd v2_fixed && python -m streamlit run app.py   # demo

pip install -r v3_imbalance/requirements.txt
cd v3_imbalance && python v3_imbalance.py       # needs data/mbti_1.csv (download from Kaggle, not included)
```

## Credits
v1 was a team project (Manoj R A, G Koushik Kumar Reddy, Anil P, Jayantth Pawar E). v2 and v3 are my
follow-up work. Data: PAN 2015 Author Profiling (CLEF); Kaggle MBTI dataset (`datasnaek/mbti-type`).
