# v3: Handling class imbalance and label leakage (MBTI Introvert vs. Extrovert)

v1 showed what goes wrong when accuracy hides a majority-class model (see `../POSTMORTEM.md`).
v3 applies the fixes to a larger, naturally imbalanced dataset.

## Data
Kaggle "(MBTI) Myers-Briggs Personality Type Dataset" (`datasnaek/mbti-type`): the last 50 forum posts
of 8,675 PersonalityCafe users, each labeled with a four-letter type. Target here = first letter (I vs. E).
The raw file is not committed; download it from Kaggle and place it in this folder.

- n = 8,675 | Extrovert (minority, positive class) = 23.0% | Introvert = 77.0%
- An "always introvert" model gets **77.0% accuracy**, so accuracy alone is not informative.

## Method
- 5-fold stratified CV; TF-IDF (20k unigrams, min_df=5, sublinear tf) fit on each training fold only.
- Metrics: macro-F1, minority precision/recall/F1, PR-AUC (chance level = 0.23), ROC-AUC, plus accuracy.
- Compared: majority baseline, plain logistic regression, class-weighted logistic regression,
  random oversampling, and class-weighted regression with a decision threshold tuned on a validation
  slice of the training fold (never on the test fold).

## Result 1: imbalance handling (type codes like "INFJ" stripped)
| Model | Accuracy | Macro-F1 | Extrovert recall | Extrovert precision | PR-AUC |
|---|---|---|---|---|---|
| Always introvert | 0.770 | 0.435 | 0.000 | 0.000 | n/a |
| LogReg, no handling | 0.778 | 0.487 | 0.054 | 0.739 | 0.523 |
| LogReg, balanced class weights | 0.765 | 0.686 | 0.571 | 0.492 | 0.524 |
| Random oversampling + LogReg | 0.772 | 0.682 | 0.522 | 0.504 | 0.522 |
| Balanced + tuned threshold | 0.735 | 0.674 | 0.659 | 0.453 | 0.524 |

- Plain logistic regression has the best accuracy and finds only 5% of extroverts.
- The three imbalance methods land within about 0.01 macro-F1 of each other. With one 5-fold run and no
  confidence intervals, I do not claim one is better; the tuned threshold is a precision/recall trade-off.

## Result 2: label leakage
95.9% of users write a type code (e.g. "INFJ") in their own posts. Same model (balanced logistic regression):

| Cleaning level | Accuracy | Macro-F1 | Extrovert recall | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| 1. Raw posts | 0.839 | 0.780 | 0.702 | 0.707 | 0.879 |
| 2. Type codes stripped | 0.765 | 0.686 | 0.571 | 0.524 | 0.785 |
| 3. + MBTI/enneagram jargon stripped | 0.756 | 0.677 | 0.566 | 0.493 | 0.770 |

Leakage inflated macro-F1 by about 0.10 and PR-AUC by about 0.21. After level 2, the top words still
included cognitive-function and enneagram labels (ni, ne, 7w8, sx), so level 3 removes them.
**Level 3 is the honest result:** macro-F1 0.68 vs. 0.44 for the majority baseline, PR-AUC 0.49 vs. 0.23 chance.

Words pushing toward Extrovert at level 3: fun, bored, guys, friends, lol, awesome, crazy, xd, super, hey.
Toward Introvert: family, quiet, games, nature, dream, mind, anime, music.
Everyday language, not type labels, though a few leftovers remain (for example "nts").

## Limitations
- Labels are self-reported by forum members; MBTI has weak psychometric validity.
- One forum's users are not a general population.
- Single 5-fold CV (no repeats or intervals); hyperparameters were not tuned.
- The jargon list is hand-built, so some leakage may remain.
- SMOTE is implemented in the script (`RUN_SMOTE`) but was not run for these results.
- Not suitable for judging real people.

## Reproduce
```
pip install scikit-learn imbalanced-learn pandas numpy
python v3_imbalance.py
```
