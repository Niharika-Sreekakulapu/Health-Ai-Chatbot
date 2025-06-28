import streamlit as st
import pandas as pd
import plotly.express as px
from utils import init_granite_model, get_sample_patient_data # Ensure get_patient_profile is NOT imported here
import datetime

# --- Streamlit Application Configuration ---
st.set_page_config(
    page_title="HealthAI: Intelligent Healthcare Assistant",
    page_icon="⚕️",
    layout="wide"
)

# Initialize the IBM Granite model once
try:
    granite_model = init_granite_model()
except ValueError as e:
    st.error(f"Configuration Error: {e}. Please ensure your .env file is correctly set up.")
    granite_model = None

# --- Main Application Title ---
st.title("⚕️ HealthAI: Intelligent Healthcare Assistant")

# --- Initialize Patient Profile in Session State ---
# This ensures the profile persists across page reloads within the session.
# Default values are set if the profile isn't already in session state.
if 'patient_profile' not in st.session_state:
    st.session_state.patient_profile = {
        "age": 35,  # Default values
        "gender": "Female", # Default values
        "medical_history": "No significant medical history." # Default values
    }

# --- Sidebar for Navigation ---
st.sidebar.title("Navigation")
feature_selection = st.sidebar.radio(
    "Choose a feature:",
    ("Patient Chat", "Disease Prediction", "Treatment Plans", "Health Analytics") # "Manage Patient Profile" removed
)

# --- IMPORTANT: Removed the static patient profile display from the sidebar ---
# The patient profile inputs will now only appear within the specific forms
# where they are relevant (Disease Prediction, Treatment Plans).
# Users can update their profile directly in these forms, and the values
# will persist across features within the session.


# --- Feature Implementations ---

if feature_selection == "Patient Chat":
    st.header("Patient Chat")
    st.write("Ask any health-related question and get an empathetic, evidence-based response.")

    if granite_model:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        query = st.chat_input("Type your health question here...")
        if query:
            st.session_state.chat_history.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Prompt for Patient Chat
                    query_prompt = f"""As a healthcare AI assistant, provide a helpful, accurate, and evidence-based response to the following patient question:

PATIENT QUESTION: {query}

Provide a clear, empathetic response that:
- Directly addresses the question
- Includes relevant medical facts
- Acknowledges limitations (when appropriate)
- Suggests when to seek professional medical advice
- Avoids making definitive diagnoses
- Uses accessible, non-technical language

RESPONSE:
"""
                    response = granite_model.generate_text(prompt=query_prompt)
                    st.markdown(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
    else:
        st.warning("Granite model not initialized. Please check your configuration in the .env file and utils.py.")


elif feature_selection == "Disease Prediction":
    st.header("Disease Prediction")
    st.write("Enter your symptoms to get potential condition predictions.")

    if granite_model:
        with st.form("disease_prediction_form"):
            symptoms = st.text_area("Describe your symptoms (e.g., persistent headache, fatigue, mild fever):", height=150)

            st.subheader("Your Profile Information (editable for this prediction)")
            col_age, col_gender = st.columns(2)
            with col_age:
                age = st.number_input("Your Age:", min_value=0, max_value=120, value=st.session_state.patient_profile['age'], key="pred_age_input")
                st.session_state.patient_profile['age'] = age
            with col_gender:
                gender_options = ["Male", "Female", "Other"]
                gender_index = gender_options.index(st.session_state.patient_profile['gender']) if st.session_state.patient_profile['gender'] in gender_options else 0
                gender = st.selectbox("Your Gender:", gender_options, index=gender_index, key="pred_gender_input")
                st.session_state.patient_profile['gender'] = gender
            medical_history = st.text_area("Brief Medical History (optional):", value=st.session_state.patient_profile['medical_history'], key="pred_med_history_input")
            st.session_state.patient_profile['medical_history'] = medical_history


            submitted = st.form_submit_button("Get Prediction")

            if submitted and symptoms:
                st.write("Analyzing your symptoms...")
                recent_health_data = get_sample_patient_data().tail(1)
                avg_heart_rate = recent_health_data['Heart Rate'].values[0] if not recent_health_data.empty else 'N/A'
                avg_systolic_bp = recent_health_data['Systolic BP'].values[0] if not recent_health_data.empty else 'N/A'
                avg_diastolic_bp = recent_health_data['Diastolic BP'].values[0] if not recent_health_data.empty else 'N/A'
                avg_blood_glucose = recent_health_data['Blood Glucose'].values[0] if not recent_health_data.empty else 'N/A'

                # --- MODIFIED PROMPT BELOW ---
                prediction_prompt = f"""As a highly accurate and analytical medical AI assistant, analyze the provided patient data and **strictly focus on the specific current symptoms and patient information to infer potential health conditions.**

Patient Information:
- Current Symptoms: {symptoms}
- Age: {age}
- Gender: {gender}
- Medical History: {medical_history}

Recent Health Metrics (if available):
- Average Heart Rate: {avg_heart_rate} bpm
- Average Blood Pressure: {avg_systolic_bp}/{avg_diastolic_bp} mmHg
- Average Blood Glucose: {avg_blood_glucose} mg/dL

**Instructions for Condition Prediction:**
- **Derive predictions SOLELY from the provided 'Current Symptoms' and 'Patient Information'.**
- Provide a list of up to 3 most potential and **distinct** conditions.
- **DO NOT default to common conditions like 'Common Cold' unless the symptoms strongly and uniquely indicate it.**
- For each condition, strictly follow this format:
    1. **[Potential Condition Name]** (Likelihood: [High/Medium/Low] based on provided symptoms)
       * Explanation: [Brief, factual explanation of why this condition is considered, directly linking to the *specific symptoms provided*.]
       * Recommended Next Steps: [Specific, actionable advice, e.g., "Consult a doctor for further diagnosis," "Monitor symptoms," "Rest and hydration."]
- **Crucially, do NOT generate any text before or after this numbered list.**
- **Do NOT provide definitive diagnoses.**
- **Prioritize safety:** If symptoms are severe or unclear, strongly recommend professional medical consultation.

**Example of desired output format (this is just for format, not content):**
1. **Influenza** (Likelihood: High)
   * Explanation: Symptoms such as high fever, body aches, and fatigue are highly consistent with influenza.
   * Recommended Next Steps: Seek medical attention for diagnosis and potential antiviral treatment. Get plenty of rest and fluids.

**Begin the list of potential conditions now:**
"""
                # --- END OF MODIFIED PROMPT ---

                with st.spinner("Generating prediction..."):
                    response = granite_model.generate_text(prompt=prediction_prompt)
                st.subheader("Potential Conditions:")
                st.markdown(response)
            elif submitted and not symptoms:
                st.warning("Please enter your symptoms to get a prediction.")
    else:
        st.warning("Granite model not initialized. Please check your configuration in the .env file and utils.py.")

elif feature_selection == "Treatment Plans":
    st.header("Treatment Plans")
    st.write("Receive personalized treatment recommendations for a diagnosed condition.")

    if granite_model:
        with st.form("treatment_plan_form"):
            condition = st.text_input("Diagnosed Condition:")

            # Display and allow override of profile data within the form
            st.subheader("Your Profile Information (editable for this plan)")
            col_age_tp, col_gender_tp = st.columns(2)
            with col_age_tp:
                # Use st.session_state value as default, update session state if changed
                age = st.number_input("Patient Age:", min_value=0, max_value=120, value=st.session_state.patient_profile['age'], key="tp_age_input")
                st.session_state.patient_profile['age'] = age # Update session state immediately
            with col_gender_tp:
                gender_options = ["Male", "Female", "Other"]
                gender_index = gender_options.index(st.session_state.patient_profile['gender']) if st.session_state.patient_profile['gender'] in gender_options else 0
                gender = st.selectbox("Patient Gender:", gender_options, index=gender_index, key="tp_gender_input")
                st.session_state.patient_profile['gender'] = gender # Update session state immediately
            medical_history = st.text_area("Patient Medical History:", value=st.session_state.patient_profile['medical_history'], key="tp_med_history_input")
            st.session_state.patient_profile['medical_history'] = medical_history # Update session state immediately

            submitted = st.form_submit_button("Generate Treatment Plan")

            if submitted and condition:
                st.write("Generating personalized treatment plan...")
                treatment_prompt = f"""
You are a reliable medical AI assistant. Based on the following patient information, generate a complete treatment plan.

Patient Condition: {condition}
Age: {age}
Gender: {gender}
Medical History: {medical_history}

Please follow this **structured format strictly**:

1.  **Recommended Medications**: Include medicine names (not placeholders), standard dosage, frequency, and purpose. Example: *Paracetamol 500mg twice daily for fever*.
2.  **Lifestyle Modifications**: At least 2 suggestions.
3.  **Follow-up Testing and Monitoring**: Include test names and recommended intervals.
4.  **Dietary Recommendations**: Specific foods to prefer or avoid.
5.  **Physical Activity Guidelines**: Example: *30 minutes brisk walking daily unless contraindicated*.
6.  **Mental Health Considerations**: Simple stress-reduction or psychological support ideas.

⚠️ Do NOT use fake medicine names like abc, xyz, or medicine-1. Only include real, general medical advice or clearly say: “Consult a physician for medicine specifics”.

Begin your structured plan below:
"""

                with st.spinner("Generating treatment plan..."):
                    response = granite_model.generate_text(prompt=treatment_prompt)
                st.subheader(f"Treatment Plan for {condition}:")
                st.markdown(response)
            elif submitted and not condition:
                st.warning("Please enter a diagnosed condition to generate a treatment plan.")
    else:
        st.warning("Granite model not initialized. Please check your configuration in the .env file and utils.py.")


elif feature_selection == "Health Analytics":
    st.header("Health Analytics Dashboard")
    st.write("Visualize your vital signs and receive AI-generated insights.")

    patient_data = get_sample_patient_data()

    st.subheader("Health Metric Trends Over Time")
    fig_hr = px.line(patient_data, x='Date', y='Heart Rate', title='Heart Rate Trend')
    st.plotly_chart(fig_hr, use_container_width=True)

    fig_bp = px.line(patient_data, x='Date', y=['Systolic BP', 'Diastolic BP'], title='Blood Pressure Trend')
    st.plotly_chart(fig_bp, use_container_width=True)

    fig_bg = px.line(patient_data, x='Date', y='Blood Glucose', title='Blood Glucose Trend')
    fig_bg.add_hline(y=100, line_dash="dot", line_color="red", annotation_text="Normal Max (Fasting)")
    st.plotly_chart(fig_bg, use_container_width=True)

    st.subheader("Key Health Indicators Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        latest_hr = patient_data['Heart Rate'].iloc[-1]
        previous_hr = patient_data['Heart Rate'].iloc[-2] if len(patient_data) > 1 else latest_hr
        delta_hr = latest_hr - previous_hr
        st.metric(label="Latest Heart Rate", value=f"{latest_hr} bpm", delta=f"{delta_hr:.1f} bpm")
        st.markdown(f"**Average:** {patient_data['Heart Rate'].mean():.1f} bpm")

    with col2:
        latest_sys = patient_data['Systolic BP'].iloc[-1]
        previous_sys = patient_data['Systolic BP'].iloc[-2] if len(patient_data) > 1 else latest_sys
        delta_sys = latest_sys - previous_sys
        st.metric(label="Latest Systolic BP", value=f"{latest_sys} mmHg", delta=f"{delta_sys:.1f} mmHg")
        st.markdown(f"**Average:** {patient_data['Systolic BP'].mean():.1f} mmHg")

    with col3:
        latest_dia = patient_data['Diastolic BP'].iloc[-1]
        previous_dia = patient_data['Diastolic BP'].iloc[-2] if len(patient_data) > 1 else latest_dia
        delta_dia = latest_dia - previous_dia
        st.metric(label="Latest Diastolic BP", value=f"{latest_dia} mmHg", delta=f"{delta_dia:.1f} mmHg")
        st.markdown(f"**Average:** {patient_data['Diastolic BP'].mean():.1f} mmHg")

    with col4:
        latest_bg = patient_data['Blood Glucose'].iloc[-1]
        previous_bg = patient_data['Blood Glucose'].iloc[-2] if len(patient_data) > 1 else latest_bg
        delta_bg = latest_bg - previous_bg
        st.metric(label="Latest Blood Glucose", value=f"{latest_bg} mg/dL", delta=f"{delta_bg:.1f} mg/dL")
        st.markdown(f"**Average:** {patient_data['Blood Glucose'].mean():.1f} mg/dL")

    st.subheader("AI-Generated Insights")
    if granite_model:
        with st.spinner("Generating insights..."):
            analytics_prompt = f"""Based on the following recent health metrics for a patient (last 7 days):
Heart Rates: {list(patient_data['Heart Rate'].tail(7))}
Systolic BPs: {list(patient_data['Systolic BP'].tail(7))}
Diastolic BPs: {list(patient_data['Diastolic BP'].tail(7))}
Blood Glucoses: {list(patient_data['Blood Glucose'].tail(7))}

Provide a brief summary of potential health observations or general recommendations for improvement based on these trends. Avoid making definitive diagnoses.
"""
            response = granite_model.generate_text(prompt=analytics_prompt)
            st.markdown(response)
    else:
        st.warning("Granite model not initialized. Please check your configuration in the .env file and utils.py.")