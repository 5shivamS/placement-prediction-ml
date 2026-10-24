import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Placement Predictor", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

try:
    pipeline = load_model()
except FileNotFoundError:
    st.error("Model artifact 'model.joblib' missing. Run `python train.py` first.")
    st.stop()

st.title("🎓 Campus Placement Predictor")
st.write("Predict student placement likelihood using academic & interview-readiness metrics.")

col1, col2 = st.columns(2)

with col1:
    cgpa = st.slider("CGPA", min_value=5.0, max_value=10.0, value=7.5, step=0.1)
    internships = st.number_input("Internships Completed", min_value=0, max_value=8, value=1, step=1)
    projects = st.number_input("Major Projects", min_value=0, max_value=10, value=2, step=1)

with col2:
    dsa_score = st.slider("DSA / Technical Assessment Score (0-100)", min_value=0.0, max_value=100.0, value=65.0, step=1.0)
    comm_score = st.select_slider("Communication Rating (1: Basic, 5: Fluent)", options=[1, 2, 3, 4, 5], value=3)

if st.button("Evaluate Placement Odds", use_container_width=True):
    input_df = pd.DataFrame([{
        "cgpa": cgpa,
        "internships": internships,
        "projects": projects,
        "dsa_score": dsa_score,
        "comm_score": comm_score
    }])
    
    prob = pipeline.predict_proba(input_df)[0][1]
    prediction = pipeline.predict(input_df)[0]
    
    st.markdown("---")
    st.subheader("Result Analysis")
    st.progress(float(prob))
    
    if prediction == 1:
        st.success(f"**High Placement Probability**: {prob * 100:.1f}%")
        st.write("Profile meets competitive thresholds for core technical campus drives.")
    else:
        st.warning(f"**Placement at Risk**: Estimated chance {prob * 100:.1f}%")
        st.write("Recommended focus areas: Improve technical assessment score or add verifiable internship/capstone project experience.")
        