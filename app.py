from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
from datetime import datetime
# ============================================================
# 1. Create Flask application
# ============================================================

app = Flask(__name__)


# ============================================================
# 2. Load trained model and LabelEncoder
# ============================================================

model = joblib.load("texas_salary_prediction.pkl")
le = joblib.load("class_title_encoder.pkl")

# Add agency and state number deop down manue
agency_options = joblib.load("agency_options.pkl")
state_number_options = joblib.load("state_number_options.pkl")


# ============================================================
# 3. Home page
# ============================================================

@app.route("/")
def home():

    class_titles = le.classes_.tolist()

    return render_template(
        "index.html",
        class_titles=class_titles,
        agency_options=agency_options,
        state_number_options=state_number_options
    )

# ============================================================
# 4. Prediction
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ----------------------------------------------------
        # Get values from HTML form
        # ----------------------------------------------------

        agency = float(request.form["agency"])
        class_title = request.form["class_title"]

        hrly_rate = float(request.form["hrly_rate"])
        hrs_per_wk = float(request.form["hrs_per_wk"])
        state_number = float(request.form["state_number"])
        status = request.form["status"]
        ethnicity = request.form["ethnicity"]
        gender = request.form["gender"]

        employ_date = pd.to_datetime(request.form["employ_date"])


        # ----------------------------------------------------
        # Calculate EXPERIENCE
        # Same formula used during training
        # ----------------------------------------------------

        experience = (pd.Timestamp.today() - employ_date).days / 365.25


        # ----------------------------------------------------
        # STATUS → Classified / Regular
        # Same logic used during training
        # ----------------------------------------------------

        classified = int("CLASSIFIED" in status.upper())

        regular = int("REGULAR" in status.upper())


        # ----------------------------------------------------
        # ETHNICITY → One-Hot Encoding
        # ----------------------------------------------------

        ethnicity_asian = int(ethnicity == "ASIAN")
        ethnicity_black = int(ethnicity == "BLACK")
        ethnicity_hispanic = int(ethnicity == "HISPANIC")
        ethnicity_other = int(ethnicity == "OTHER")
        ethnicity_white = int(ethnicity == "WHITE")

        # ----------------------------------------------------
        # GENDER → Binary Encoding
        # ----------------------------------------------------

        gender_male = int(gender == "MALE")


        # ----------------------------------------------------
        # CLASS TITLE → Label Encoding
        # ----------------------------------------------------

        class_title_encoded = le.transform([class_title])[0]

        # ----------------------------------------------------
        # Create DataFrame
        # EXACT same feature order as Random Forest
        # ----------------------------------------------------

        # Create input DataFrame
        input_data = pd.DataFrame([{
            "AGENCY": agency,
            "CLASS TITLE": class_title_encoded,
            "HRLY RATE": hrly_rate,
            "HRS PER WK": hrs_per_wk,
            "STATE NUMBER": state_number,
            "Classified": classified,
            "Regular": regular,
            "ETHNICITY_ASIAN": ethnicity_asian,
            "ETHNICITY_BLACK": ethnicity_black,
            "ETHNICITY_HISPANIC": ethnicity_hispanic,
            "ETHNICITY_OTHER": ethnicity_other,
            "ETHNICITY_WHITE": ethnicity_white,
            "GENDER_MALE": gender_male,
            "EXPERIENCE": experience
        }])
        
        # -----------------------------------------------------------
        # Match the EXACT feature names used during model training
        # -----------------------------------------------------------
        input_data.columns = model.feature_names_in_
        
        # Prediction
        prediction = model.predict(input_data)[0]


        # ----------------------------------------------------
        # Display result
        # ----------------------------------------------------

        return render_template(
            "index.html",
            prediction=f"${prediction:,.2f}",
            class_titles=le.classes_.tolist(),
            agency_options=agency_options,
            state_number_options=state_number_options
        )


    except Exception as e:

        return render_template(
            "index.html",
            error=str(e)
        )


# ============================================================
# 5. Run Flask application
# ============================================================

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))