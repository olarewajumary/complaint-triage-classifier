import streamlit as st

st.set_page_config(page_title="Complaint Triage Classifier")


@st.cache_resource
def load_predict_function():
    from src.predict import CONFIDENCE_THRESHOLD, load_model, predict

    load_model()
    return predict, CONFIDENCE_THRESHOLD


st.title("Consumer Complaint Triage Classifier")
st.write(
    "Paste a consumer complaint. The model picks one of 10 categories, "
    "or flags it for a human when it is not confident enough."
)

with st.spinner("Loading model..."):
    predict, threshold = load_predict_function()

text = st.text_area("Complaint text", height=200)

if st.button("Classify"):
    with st.spinner("Classifying..."):
        result = predict(text)

    if result["prediction"] == "invalid_input":
        st.error(result["error"])
    elif result["prediction"] == "needs_human_review":
        st.warning("Needs human review")
        st.write(
            f"Closest guess: {result['top_guess']} "
            f"(confidence {result['confidence']:.1%})"
        )
        st.caption(f"Confidence is below the {threshold} threshold.")
    else:
        st.success(result["prediction"])
        st.write(f"Confidence: {result['confidence']:.1%}")

st.caption(
    "Trained on CFPB complaints from 2015 to 2019. The threshold of 0.87 was "
    "chosen on validation data, where it sent 22.5% of complaints to a human "
    "and was 95.0% accurate on the rest."
)