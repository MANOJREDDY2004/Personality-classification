from pathlib import Path
import joblib
import streamlit as st

MODEL_PATH = Path(__file__).parent / "personality_model.joblib"

st.set_page_config(page_title="Extroversion from Text (PAN15)", page_icon="🧠")
st.title("Higher vs. lower extroversion from text")
st.caption("TF-IDF + KNN trained on 142 English Twitter users (PAN 2015). A learning project, not a psychological test.")


@st.cache_resource
def load():
    return joblib.load(MODEL_PATH)


model = load()

text = st.text_area(
    "Paste text written by one person (the model was trained on ~100 tweets per user)",
    height=220,
    placeholder="Paste a batch of tweets or posts here...",
)

if st.button("Predict") and text.strip():
    words = len(text.split())
    if words < 150:
        st.warning(f"Only {words} words. Training samples were far longer, so treat this result as unreliable.")
    p_high = float(model.predict_proba([text])[0][1])
    label = "Higher extroversion" if p_high >= 0.5 else "Lower extroversion"
    st.subheader(label)
    st.write(f"Share of 5 nearest training users with higher extroversion: **{p_high:.0%}**")
    st.progress(p_high)

with st.expander("How good is this model, honestly?"):
    st.markdown(
        """
**Cross-validated results (5-fold x3, n=142, median-split labels)**

| Model | Accuracy | Macro-F1 | ROC-AUC |
|---|---|---|---|
| Majority baseline | 51.4% | 0.34 | n/a |
| **KNN k=5 (used here)** | **72.3%** | **0.72** | **0.81** |

- About 72% is well above chance but far from reliable for any single person (fold-to-fold spread about 5 points).
- Labels come from a median split of the PAN extroversion score, so "higher/lower" is relative to this dataset.
- Some top-weighted words are language or app artifacts (e.g. "que", "ik", "nowplaying"), so part of the signal may not be personality.
- Short inputs, other platforms, and other languages are out of distribution.
- Do not use this to judge real people.
"""
    )
