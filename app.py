import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('employee_model.pkl')
scaler = joblib.load('scaler.pkl')

# Set page config
st.set_page_config(page_title="Employee Performance Predictor", page_icon="📊")

def set_bg_with_overlay(image_file):
    import base64
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)), url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
            color: #FFFFFF !important;

        }}

        h1, h2, h3, h4, h5, h6, p {{
            color: #FFFFFF !important;

        }}

        .css-10trblm, .css-1v0mbdj, .css-1cpxqw2 {{
            color: #FFFFFF !important;

        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call this function with your local image file
set_bg_with_overlay("background.jpg")



# ---------------- Sidebar Instructions ----------------
# ---------------- Sidebar Instructions ----------------
with st.sidebar:
    st.markdown("""
    <style>
    .sidebar-instructions {
        color: #ffffff;
        font-weight: bold;
        background-color: rgba(0, 0, 0, 0.5);
        padding: 15px;
        border-radius: 10px;
    }
    </style>
    <div class='sidebar-instructions'>
    <h4>📖 Input Field Instructions</h4>

    🎓 <b>Education Level (1-5)</b><br>
    - 1: High School <br>
    - 2: Diploma <br>
    - 3: Bachelor's Degree <br>
    - 4: Master's Degree <br>
    - 5: Doctorate <br><br>

    🧑‍💼 <b>Years of Experience</b><br>
    - Total years of work experience<br><br>

    ⏰ <b>Work Hours Per Week</b><br>
    - Average number of working hours in a week<br><br>

    🙂 <b>Job Satisfaction (1-5)</b><br>
    - 1: Very Dissatisfied<br>
    - 5: Very Satisfied<br><br>

    📁 <b>Number of Projects</b><br>
    - Total active projects<br><br>

    📚 <b>Trainings Completed</b><br>
    - Total professional trainings completed<br>

    </div>
    """, unsafe_allow_html=True)

# ---------------- Page Heading ----------------
st.markdown("## 💼 Employee Performance Predictor")
st.markdown("#### 🔍 Fill in the employee details below to predict their performance:")

# ---------------- Input Sliders ----------------
education = st.slider("🎓 Education Level (1-5)", 1, 5, 3,
                      help="1=High School, 5=Doctorate")
experience = st.slider("🧑‍💼 Years of Experience", 0, 40, 5,
                       help="Total years of professional experience")
work_hours = st.slider("⏰ Work Hours Per Week", 20, 80, 40,
                       help="Average hours the employee works per week")
job_satisfaction = st.slider("🙂 Job Satisfaction (1-5)", 1, 5, 3,
                             help="1=Very Dissatisfied, 5=Very Satisfied")
project_count = st.slider("📁 Number of Projects", 1, 10, 3,
                          help="Current number of active projects")
trainings_completed = st.slider("📚 Trainings Completed", 0, 10, 2,
                                help="Total number of trainings completed")

# ---------------- Style the Predict Button ----------------
st.markdown("""
    <style>
    div.stButton > button {
        color: white;
        background-color: #FF4B4B;
        border: 2px solid #FF4B4B;
        padding: 10px 24px;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
        margin-top: 20px;
    }
    div.stButton > button:hover {
        background-color: #ff7b7b;
        border: 2px solid #ff7b7b;
    }
    </style>
""", unsafe_allow_html=True)


# ---------------- Predict Button ----------------
if st.button("🚀 Predict Performance"):
    input_data = np.array([[education, experience, work_hours,
                            job_satisfaction, project_count, trainings_completed]])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]

    label_map = {0: "🟡 Average", 1: "🟢 High", 2: "🔴 Low"}
    result = label_map[prediction]

    # Typing effect simulation with placeholder
    import time
    result_placeholder = st.empty()
    with result_placeholder.container():
        st.markdown("### 🎯 **Prediction Result**")
        time.sleep(0.5)
        st.success("Predicted")
        time.sleep(1)

        # Show animated result based on prediction
        if prediction == 1:
            st.markdown("### 🟢 High Performance! 🎉")
            st.image("https://media.giphy.com/media/111ebonMs90YLu/giphy.gif", width=200)
            st.balloons()
        elif prediction == 0:
            st.markdown("### 🟡 Average Performance 😐")
            st.image("https://media.giphy.com/media/l0MYC0LajbaPoEADu/giphy.gif", width=200)
        else:
            st.markdown("### 🔴 Low Performance 😟")
            st.image("https://media.giphy.com/media/3o6ZtaO9BZHcOjmErm/giphy.gif", width=200)


