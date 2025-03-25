import streamlit as st
from auth import signup, login, logout, is_admin, set_admin
from PIL import Image
import base64
import pandas as pd
import sqlite3
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# ✅ Set page config
st.set_page_config(page_title="Disease Prediction System", layout="wide")

# ✅ Function to set background image
def set_bg_from_local(image_file):
    with open(image_file, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded_string}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_from_local(r"C:\Users\DELL\Desktop\project\Disease-prediction-modified(2)\Disease-prediction-modified(1)\disease-prediction.jpg")


# ✅ Connect to Database
conn = sqlite3.connect('data/predictions_history.db', check_same_thread=False)
cursor = conn.cursor()

# ✅ Create Tables
cursor.execute('''
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        symptoms TEXT,
        predicted_disease TEXT,
        confidence_score REAL,
        diet TEXT,
        workout TEXT,
        medication TEXT,
        precaution TEXT,
        timestamp TEXT
    )
''')
conn.commit()

# ✅ Load Data
training_data = pd.read_csv('data/Training.csv')
diet_data = pd.read_csv('data/diets.csv')
workout_data = pd.read_csv('data/workout_df.csv')
medications_data = pd.read_csv('data/medications.csv')
precautions_data = pd.read_csv('data/precautions_df.csv')
symptom_description = pd.read_csv('data/symptoms_description.csv')

# ✅ Prepare Training Data
symptoms = sorted(training_data.columns[:-1].tolist())
X = training_data[symptoms]
y = training_data['prognosis']

# ✅ Train Models
clf_dt = DecisionTreeClassifier().fit(X, y)
clf_rf = RandomForestClassifier().fit(X, y)
clf_nb = GaussianNB().fit(X, y)
clf_knn = KNeighborsClassifier().fit(X, y)

# ✅ Authentication State
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

# ✅ Authentication UI
def authentication_ui():
    if not st.session_state.logged_in:
        auth_option = st.sidebar.radio("Login / Signup", ["Login", "Signup"])
        if auth_option == "Login":
            username = st.sidebar.text_input("Username")
            password = st.sidebar.text_input("Password", type="password")
            if st.sidebar.button("Login"):
                success, is_admin_flag = login(username, password)
                if success:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.is_admin = is_admin_flag
                    st.rerun()  # ✅ Force refresh after login
        elif auth_option == "Signup":
            username = st.sidebar.text_input("Choose a Username")
            password = st.sidebar.text_input("Choose a Password", type="password")
            if st.sidebar.button("Signup"):
                signup(username, password)
    else:
        st.sidebar.write(f"👤 Logged in as **{st.session_state.username}**")
        
        # ✅ Double-check admin status in case session resets
        if is_admin(st.session_state.username):
            st.session_state.is_admin = True

        if st.session_state.is_admin:
            st.sidebar.write("🔑 **Admin** Access")
            new_admin = st.sidebar.text_input("Enter Username to Promote")
            if st.sidebar.button("Promote"):
                if set_admin(new_admin):
                    st.success(f"✅ {new_admin} is now an Admin!")
                    st.rerun()  # ✅ Refresh page after promotion

        if st.sidebar.button("Logout"):
            logout()
            st.rerun()


authentication_ui()

# ✅ Prediction Logic
st.title("🦠 Disease Prediction System")

if st.session_state.logged_in:
    st.markdown("### 🩺 Select Symptoms:")
    symptom_inputs = []
    for i in range(5):
        selected_symptom = st.selectbox(f"Symptom {i + 1}", [''] + symptoms, key=f'symptom_{i}')
        if selected_symptom:
            description = symptom_description.loc[
                symptom_description['Symptom'].str.lower() == selected_symptom.lower(), 'Description'
            ]
            if not description.empty:
                st.write(f"**📝 Description:** {description.iloc[0]}")
        symptom_inputs.append(selected_symptom)

    if st.button('🚀 Predict'):
        selected_symptoms = [s for s in symptom_inputs if s]
        if selected_symptoms:
            input_vector = [1 if s in selected_symptoms else 0 for s in symptoms]
            input_df = pd.DataFrame([input_vector], columns=symptoms)

            # ✅ Get Predictions
            predictions = {
                'Decision Tree': (clf_dt.predict(input_df)[0], clf_dt.predict_proba(input_df)[0].max() * 100),
                'Random Forest': (clf_rf.predict(input_df)[0], clf_rf.predict_proba(input_df)[0].max() * 100),
                'Naive Bayes': (clf_nb.predict(input_df)[0], max(clf_nb.predict_proba(input_df)[0]) * 100),
                'KNN': (clf_knn.predict(input_df)[0], max(clf_knn.predict_proba(input_df)[0]) * 100)
            }

            # ✅ Display Predictions Table
            result_df = pd.DataFrame(predictions).T.reset_index()
            result_df.columns = ["Model", "Disease", "Confidence (%)"]
            st.markdown("### 📊 Prediction Results:")
            st.dataframe(result_df)

            # ✅ Best Prediction
            best_prediction = result_df.loc[result_df['Confidence (%)'].idxmax()]
            best_disease = best_prediction["Disease"]

            # ✅ Get Recommendations
            diet = diet_data.loc[diet_data.iloc[:, 0] == best_disease, diet_data.columns[1]].values[0]
            workout = workout_data.loc[workout_data.iloc[:, 0] == best_disease, workout_data.columns[1]].values[0]
            medication = medications_data.loc[medications_data.iloc[:, 0] == best_disease, medications_data.columns[1]].values[0]
            precaution = precautions_data.loc[precautions_data.iloc[:, 0] == best_disease, precautions_data.columns[1]].values[0]

            # ✅ Display Recommendations
            st.markdown("### 🍎 Recommendations:")
            rec_table = pd.DataFrame({
                'Category': ['Diet', 'Workout', 'Medication', 'Precaution'],
                'Recommendation': [diet, workout, medication, precaution]
            })
            st.table(rec_table)
            # ✅ Confidence Score Bar Chart (NEW)
            import plotly.express as px
            fig_confidence = px.bar(result_df, x="Model", y="Confidence (%)", color="Confidence (%)",
                                 title="Confidence Score by Model", labels={"Confidence (%)": "Confidence"},
                                 text="Confidence (%)", color_continuous_scale="Blues")
            st.plotly_chart(fig_confidence)
            # ✅ Spider Chart (NEW)
            import plotly.graph_objects as go
            fig_spider = go.Figure()
            for _, row in result_df.iterrows():
                fig_spider.add_trace(go.Scatterpolar(
                    r=[row["Confidence (%)"]],
                    theta=[row["Model"]],
                    fill='toself',
                    name=row["Model"]
                ))
            fig_spider.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                title="Model Confidence Comparison (Spider Chart)"
            )
            st.plotly_chart(fig_spider)

            # ✅ Save to History
            cursor.execute('''
                INSERT INTO history (username, symptoms, predicted_disease, confidence_score, diet, workout, medication, precaution, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (st.session_state.username, ', '.join(selected_symptoms), best_disease, best_prediction["Confidence (%)"],
                  diet, workout, medication, precaution, datetime.now()))
            conn.commit()

# ✅ History Section (on button click)
if st.session_state.logged_in and st.button("📜 View Prediction History"):
    if is_admin(st.session_state.username):  # ✅ Double-check admin status
        history_df = pd.read_sql_query("SELECT * FROM history", conn)
        st.markdown("### 🔍 All User History (Admin View)")
    else:
        history_df = pd.read_sql_query(
            "SELECT * FROM history WHERE username = ?", conn, params=(st.session_state.username,)
        )
        st.markdown("### 📜 Your Prediction History")

    if not history_df.empty:
        st.dataframe(history_df)
        
         # ✅ Disease Trend Analysis (NEW)
        import plotly.express as px
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        trend_data = history_df.groupby('predicted_disease').size().reset_index(name='count')
        fig_trend = px.bar(trend_data, x="predicted_disease", y="count", color="count",
                            title="Most Common Predicted Diseases", labels={"count": "Occurrences"},
                            color_continuous_scale="Viridis")
        st.plotly_chart(fig_trend)
    else:
        st.info("No history available.")


# ✅ Close Connection
conn.close()
