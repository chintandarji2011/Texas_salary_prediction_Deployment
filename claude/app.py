from flask import Flask, render_template, request
import pandas as pd
import joblib

# ============================================================
# 1. Create Flask application
# ============================================================

app = Flask(__name__)


# ============================================================
# 2. Load trained model, LabelEncoder, and dropdown options
# ============================================================

model        = joblib.load("texas_salary_prediction.pkl")
le           = joblib.load("class_title_encoder.pkl")
agency_options = joblib.load("agency_options.pkl")   # [{"value":101,"label":"101 - SENATE"}, ...]


# ============================================================
# 3. Home page
# ============================================================

@app.route("/")
def home():
    return render_template(
        "index.html",
        class_titles=le.classes_.tolist(),
        agency_options=agency_options
    )


# ============================================================
# 4. Prediction
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ----------------------------------------------------
        # 4.1  Raw inputs from form
        # ----------------------------------------------------

        agency      = float(request.form["agency"])
        class_title = request.form["class_title"]
        hrs_per_wk  = float(request.form["hrs_per_wk"])
        monthly_salary = float(request.form["monthly_salary"])   # NEW — user enters this
        status      = request.form["status"]
        ethnicity   = request.form["ethnicity"]
        gender      = request.form["gender"]
        employ_date = pd.to_datetime(request.form["employ_date"])

        # ----------------------------------------------------
        # 4.2  Auto-compute derived fields
        #      User never sees or enters these manually
        # ----------------------------------------------------

        # ANNUAL  = MONTHLY × 12  (confirmed: zero difference across all 149,481 rows)
        annual_salary = monthly_salary * 12

        # HRLY RATE — derived from monthly salary and hours
        # Formula used in dataset: MONTHLY / (HRS_PER_WK × 4.33)
        # 4.33 = avg weeks per month (52 ÷ 12)
        hrly_rate = monthly_salary / (hrs_per_wk * 4.33) if hrs_per_wk > 0 else 0

        # summed_annual_salary = total annual across all jobs
        # For 94.7% of employees this equals their single job ANNUAL
        # User can optionally add a second job salary — handled below
        other_jobs_annual  = float(request.form.get("other_jobs_annual", 0) or 0)
        summed_annual_salary = annual_salary + other_jobs_annual

        # EXPERIENCE = (today - employ_date) / 365.25
        experience = (pd.Timestamp.today() - employ_date).days / 365.25

        # ----------------------------------------------------
        # 4.3  STATUS → Classified / Regular flags
        # ----------------------------------------------------

        classified = int("CLASSIFIED" in status.upper())
        regular    = int("REGULAR"    in status.upper())

        # ----------------------------------------------------
        # 4.4  ETHNICITY → One-Hot Encoding
        # ----------------------------------------------------

        ethnicity_asian    = int(ethnicity == "ASIAN")
        ethnicity_black    = int(ethnicity == "BLACK")
        ethnicity_hispanic = int(ethnicity == "HISPANIC")
        ethnicity_other    = int(ethnicity == "OTHER")
        ethnicity_white    = int(ethnicity == "WHITE")

        # ----------------------------------------------------
        # 4.5  GENDER → Binary
        # ----------------------------------------------------

        gender_male = int(gender == "MALE")

        # ----------------------------------------------------
        # 4.6  CLASS TITLE → Label Encoding
        # ----------------------------------------------------

        class_title_clean = class_title.strip().upper()

        if class_title_clean in le.classes_:
            class_title_encoded = le.transform([class_title_clean])[0]
        else:
            class_title_encoded = 0   # fallback for unseen titles

        # ----------------------------------------------------
        # 4.7  Build input DataFrame — exact feature order
        # ----------------------------------------------------

        input_data = pd.DataFrame([{
            "AGENCY"              : agency,
            "CLASS TITLE"         : class_title_encoded,
            "HRLY RATE"           : hrly_rate,
            "HRS PER WK"          : hrs_per_wk,
            "summed_annual_salary": summed_annual_salary,
            "Classified"          : classified,
            "Regular"             : regular,
            "ETHNICITY_ASIAN"     : ethnicity_asian,
            "ETHNICITY_BLACK"     : ethnicity_black,
            "ETHNICITY_HISPANIC"  : ethnicity_hispanic,
            "ETHNICITY_OTHER"     : ethnicity_other,
            "ETHNICITY_WHITE"     : ethnicity_white,
            "GENDER_MALE"         : gender_male,
            "EXPERIENCE"          : experience
        }])

        # ----------------------------------------------------
        # 4.8  Predict
        # ----------------------------------------------------

        prediction = model.predict(input_data)[0]

        return render_template(
            "index.html",
            prediction           = f"${prediction:,.2f}",
            monthly_entered      = f"${monthly_salary:,.2f}",
            annual_computed      = f"${annual_salary:,.2f}",
            summed_computed      = f"${summed_annual_salary:,.2f}",
            experience_computed  = f"{experience:.1f} years",
            hrly_rate_computed   = f"${hrly_rate:.2f}",
            class_titles         = le.classes_.tolist(),
            agency_options       = agency_options
        )

    except Exception as e:

        return render_template(
            "index.html",
            error          = str(e),
            class_titles   = le.classes_.tolist(),
            agency_options = agency_options
        )


# ============================================================
# 5. Run
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)
