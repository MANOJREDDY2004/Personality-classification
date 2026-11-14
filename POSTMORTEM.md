# Post-mortem: why our original 93.1% was meaningless

This folder (`v1_original/`) is our team's coursework exactly as submitted. It is left unchanged on purpose.
This note explains what was wrong with it, how I found out, and what changed in `v2_fixed/`.

## Summary
Our English model reported **93.1% test accuracy**. That number equals what you get by predicting
"introvert" for everyone. The model learned nothing about extroversion, and the class imbalance we
blamed in the report was created by our own labeling rule, not by the data.

## What was wrong

### 1. The label threshold manufactured the imbalance
Labels were built from the PAN15 extroversion score with the rule `score >= 0.5 -> extrovert`.
The scores in this data range from **-0.3 to 0.5, with a median of 0.2**. Only **10 of 142 users**
(7.0%) reach 0.5, so 132 users (92.3%) became "introvert". A user scoring 0.4 was labeled introvert.
The 92/7 split came from this rule, not from the population.

### 2. A bogus row in the dataset
`truth.txt` (the label file itself) was read in as a "user" and tagged `unknown`.
The notebook therefore used 143 rows; the real number of users is **142**.
The report says the row was removed, but the 143 count and the 114/29 split show it was still in.

### 3. One tiny test split, and accuracy as the only headline
The 29-user test set contained 2 extroverts. The classifier predicted introvert for all 29, so:
- accuracy = 27/29 = 93.1% (the majority-class baseline)
- extrovert precision, recall, and F1 = 0.00
- macro-F1 = 0.48

### 4. A misread diagnostic
The accuracy was identical for every K from 1 to 15. The report read this as "clean introvert
clusters". The simpler explanation is that the model predicted the majority class regardless of K.

### 5. Other consistency issues in the written report
- The report names PAN 2017; the files used are PAN 2015 (the folders are named `pan15-...`),
  which is the edition that contains Big Five trait scores.
- The Spanish confusion matrix (17/18 correct) does not match the stated 83.3% accuracy.
- The Spanish data was not re-checked in this audit. It may have the same threshold problem.

## How I found it
Reading `create_dataset.py` showed the 0.5 rule; comparing it with the raw `truth.txt` showed the score
range; recomputing the baseline showed the accuracy equalled it.

## What changed in v2 (`v2_fixed/`)
- Removed the bogus row (n = 142).
- Re-labeled by median split of the raw score (73 higher / 69 lower extroversion).
- 5-fold stratified CV repeated 3 times, with a majority baseline, macro-F1, and ROC-AUC.

| Model | Accuracy | Macro-F1 | ROC-AUC |
|---|---|---|---|
| Majority baseline | 51.4% | 0.34 | n/a |
| KNN k=5, word TF-IDF | 72.3% | 0.72 | 0.81 |
| LogReg, char TF-IDF | 70.9% | 0.71 | 0.80 |
| LinearSVC, word TF-IDF | 68.8% | 0.69 | 0.79 |
| LogReg, word TF-IDF | 68.5% | 0.68 | 0.78 |

The median split was chosen after seeing the score distribution. Accuracy varies by about 5 points
across folds, so these numbers are a modest, real signal on a small dataset, not a solved problem.
Some top-weighted words are language or app artifacts (for example "que", "ik", "nowplaying"), so
part of the signal may not be personality.

## What I would do differently
1. Look at the label distribution before choosing a threshold.
2. Always report a majority baseline next to accuracy.
3. Use cross-validation when the dataset has fewer than a few hundred rows.
4. Check that diagnostics (like a flat K curve) are not explained by trivial behavior.

## Next: v3 (`v3_imbalance/`)
Natural class imbalance on a larger real dataset, with leakage removal, class weights vs. oversampling,
threshold tuning, and error analysis.
