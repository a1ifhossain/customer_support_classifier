"""A small local demo for the support classifier."""
import pandas as pd
import streamlit as st

from config import EXAMPLES, INTENTS, ROOT
from predict import classify, load_model

st.set_page_config(page_title="customer_support_classifier", page_icon="💬", layout="centered")
st.title("customer_support_classifier")
st.write("Categorize an English customer message into one of ten banking support intents.")


@st.cache_resource
def get_bundle():
    return load_model()


if not (ROOT / "artifacts" / "model.joblib").exists():
    st.info("Run python train.py in this project folder, then refresh this page.")
    st.stop()

bundle = get_bundle()
with st.expander("Supported categories and examples"):
    for category in INTENTS.values():
        st.write(f"• {category}")
    st.caption("Example: " + EXAMPLES[1])

with st.form("prediction"):
    message = st.text_area("Customer message", placeholder=EXAMPLES[0], max_chars=2000)
    submitted = st.form_submit_button("Classify message", type="primary")

if submitted:
    try:
        results = classify(message, bundle)
    except ValueError as error:
        st.warning(str(error))
    else:
        st.success(results[0]["label"])
        st.caption("Top three model scores — uncalibrated, not a guarantee of correctness.")
        st.dataframe(pd.DataFrame(results)[["label", "score"]], hide_index=True, width="stretch")

st.caption("This model always selects a supported category, including for unrelated questions. English demo only.")
metrics = bundle["metadata"]
st.divider()
st.subheader("Held-out evaluation")
left, right = st.columns(2)
left.metric("Macro-F1", f"{metrics['test_selected_model']['macro_f1']:.3f}")
right.metric("Test messages", str(metrics["split_sizes"]["test"]))
st.caption("Selected 10-intent BANKING77 subset; exact text overlaps with training were removed from test.")
with st.expander("Model and dataset details"):
    st.write("Model:", metrics["selected_model"].replace("_", " "))
    st.write("Text features: TF-IDF with word unigrams and bigrams.")
    st.write("Selection: validation macro-F1; final model retrained on the cleaned official training subset.")
    st.write("Dataset: BANKING77 by PolyAI, CC BY 4.0.")
    st.link_button("Dataset source", "https://github.com/PolyAI-LDN/task-specific-datasets")
