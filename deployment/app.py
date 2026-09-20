"""Streamlit UI for Tourism Package purchase prediction."""
import os
import joblib
import pandas as pd
import streamlit as st

MODEL_FILENAME = "best_tourism_package_model_v1.joblib"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILENAME)
THRESHOLD = 0.45

@st.cache_resource(show_spinner=False)
def load_model(path):
    if not os.path.exists(path):
        st.error(f"Model file not found at {path}. Run the pipeline first.")
        st.stop()
    return joblib.load(path)

model = load_model(MODEL_PATH)

st.set_page_config(page_title="Tourism Package Prediction", page_icon="✈️")
st.title("✈️ Tourism Package Prediction")
st.caption("Predict whether a customer will purchase the Wellness Tourism Package.")

with st.form("customer_form"):
    c1, c2 = st.columns(2)
    with c1:
        Age                    = st.slider("Age", 18, 70, 30)
        TypeofContact          = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
        CityTier               = st.selectbox("City Tier", [1, 2, 3])
        DurationOfPitch        = st.slider("Duration of Pitch (min)", 0, 100, 15)
        Occupation             = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        Gender                 = st.selectbox("Gender", ["Male", "Female", "Others"])
        NumberOfPersonVisiting = st.slider("Number of Persons Visiting", 1, 5, 2)
        NumberOfFollowups      = st.slider("Number of Follow-ups", 1, 10, 3)
        ProductPitched         = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    with c2:
        PreferredPropertyStar    = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
        MaritalStatus            = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Unmarried"])
        NumberOfTrips            = st.slider("Number of Trips", 1, 20, 3)
        Passport                 = st.selectbox("Has Passport?", ["Yes", "No"])
        PitchSatisfactionScore   = st.slider("Pitch Satisfaction Score", 1, 5, 3)
        OwnCar                   = st.selectbox("Owns a Car?", ["Yes", "No"])
        NumberOfChildrenVisiting = st.slider("Number of Children Visiting", 0, 5, 1)
        Designation              = st.selectbox("Designation", ["Executive", "Manager", "AVP", "VP", "Sr. Manager"])
        MonthlyIncome            = st.number_input("Monthly Income", min_value=1000.0, value=30000.0, step=500.0)
    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([{
        "Age": Age, "TypeofContact": TypeofContact, "CityTier": CityTier,
        "DurationOfPitch": DurationOfPitch, "Occupation": Occupation,
        "Gender": Gender, "NumberOfPersonVisiting": NumberOfPersonVisiting,
        "NumberOfFollowups": NumberOfFollowups, "ProductPitched": ProductPitched,
        "PreferredPropertyStar": PreferredPropertyStar, "MaritalStatus": MaritalStatus,
        "NumberOfTrips": NumberOfTrips,
        "Passport": 1 if Passport == "Yes" else 0,
        "PitchSatisfactionScore": PitchSatisfactionScore,
        "OwnCar": 1 if OwnCar == "Yes" else 0,
        "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
        "Designation": Designation, "MonthlyIncome": MonthlyIncome,
    }])
    prob = float(model.predict_proba(input_df)[0, 1])
    pred = int(prob >= THRESHOLD)
    st.divider()
    if pred == 1:
        st.success(f"✅ Likely to purchase — probability {prob:.1%}")
    else:
        st.warning(f"⚠️ Unlikely to purchase — probability {prob:.1%}")
    st.progress(min(max(prob, 0.0), 1.0))
    st.caption(f"Decision threshold: {THRESHOLD}")
