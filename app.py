from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import os
import re
import uuid
import hashlib
from flask import flash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime, timedelta
from bson import ObjectId
from sklearn.ensemble import IsolationForest
from werkzeug.security import generate_password_hash, check_password_hash
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

# Load .env before reading any environment variables.

# Load .env before reading any environment variables.
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")

if not app.secret_key:
    raise RuntimeError("FLASK_SECRET_KEY environment variable is missing.")
# =========================================
# EMAIL CONFIGURATION FOR PASSWORD RESET
# =========================================

app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 587
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USE_SSL"] = False
app.config["MAIL_USERNAME"] = os.getenv("MAIL_USERNAME")
app.config["MAIL_PASSWORD"] = os.getenv("MAIL_PASSWORD")
app.config["MAIL_DEFAULT_SENDER"] = os.getenv("MAIL_USERNAME")

mail = Mail(app)

password_reset_serializer = URLSafeTimedSerializer(
    app.secret_key,
    salt="autism-monitor-password-reset"
)


# =========================================
# MONGODB CONNECTION
# =========================================

MONGO_URI = os.getenv("MONGODB_URI")

# Always define these names so a failed connection cannot cause
# NameError. Routes must check MONGO_CONNECTED before using them.
client = None
db = None
users_collection = None
students_collection = None
observations_collection = None
MONGO_CONNECTED = False

if not MONGO_URI:
    print("MongoDB connection failed ❌")
    print("Error: MONGODB_URI was not found. Check the .env file in the project folder.")
else:
    try:
        client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=10000
        )

        # Create collection handles before checking the connection.
        db = client["autism_monitor"]
        users_collection = db["users"]
        students_collection = db["students"]
        observations_collection = db["observations"]
        followups_collection = db["followups"]

        # Verify that MongoDB Atlas is actually reachable.
        client.admin.command("ping")
        MONGO_CONNECTED = True
        print("MongoDB connected successfully! ✅")

    except Exception as e:
        MONGO_CONNECTED = False
        print("MongoDB connection failed ❌")
        print("Error type:", type(e).__name__)
        print("Error:", e)

DATA_FILE = "data/observations.csv"

# =========================================================
# AI ANOMALY DETECTION
# =========================================================


def detect_anomalies(student_df):

    feature_columns = [
        "communication",
        "social_interaction",
        "learning",
        "hyperactivity",
        "emotional_state",
        "sound_sensitivity"
    ]

    # Work on a copy so the original DataFrame is not
    # unintentionally modified by this function.
    student_df = student_df.copy()

    # Ensure the output columns exist.
    student_df["anomaly"] = 0
    student_df["anomaly_status"] = "Not enough data"

    # Convert observation ratings to numeric values.
    for column in feature_columns:
        if column not in student_df.columns:
            return student_df

        student_df[column] = pd.to_numeric(
            student_df[column],
            errors="coerce"
        )

    # Exclude absent records and rows with incomplete ratings.
    if "attendance" in student_df.columns:
        present_mask = (
            student_df["attendance"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["present"])
        )
    else:
        # Support older datasets that don't contain attendance.
        present_mask = pd.Series(
            True,
            index=student_df.index
        )

    valid_mask = (
        present_mask
        & student_df[feature_columns].notna().all(axis=1)
    )

    valid_data = student_df.loc[valid_mask].copy()

    # AI analysis requires at least five complete,
    # present-student observations.
    if len(valid_data) < 5:
        student_df.loc[
            present_mask,
            "anomaly_status"
        ] = "Not enough data"

        student_df.loc[
            ~present_mask,
            "anomaly_status"
        ] = "Absent - Not Analyzed"

        return student_df

    # Train the Isolation Forest model using valid ratings only.
    model = IsolationForest(
        contamination=0.20,
        random_state=42
    )

    predictions = model.fit_predict(
        valid_data[feature_columns]
    )

    # Store results for analyzed observations.
    valid_data["anomaly"] = predictions

    valid_data["anomaly_status"] = valid_data[
        "anomaly"
    ].map({
        -1: "Unusual Pattern",
         1: "Normal Pattern"
    })

    # Copy predictions back to the original row positions.
    student_df.loc[
        valid_data.index,
        "anomaly"
    ] = valid_data["anomaly"]

    student_df.loc[
        valid_data.index,
        "anomaly_status"
    ] = valid_data["anomaly_status"]

    # Keep absent records separate from AI analysis.
    student_df.loc[
        ~present_mask,
        "anomaly"
    ] = 0

    student_df.loc[
        ~present_mask,
        "anomaly_status"
    ] = "Absent - Not Analyzed"

    return student_df


# =========================================================
# ANALYZE CHANGES IN UNUSUAL OBSERVATION
# =========================================================



def analyze_anomaly_changes(student_df):

    feature_columns = [
        "communication",
        "social_interaction",
        "learning",
        "hyperactivity",
        "emotional_state",
        "sound_sensitivity"
    ]

    feature_names = {
        "communication": "Communication",
        "social_interaction": "Social Interaction",
        "learning": "Learning",
        "hyperactivity": "Hyperactivity",
        "emotional_state": "Emotional State",
        "sound_sensitivity": "Sound Sensitivity"
    }

    if student_df.empty or "anomaly" not in student_df.columns:
        return []

    df = student_df.copy()

    # Ensure all ratings are numeric.
    for column in feature_columns:
        if column not in df.columns:
            return []

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Exclude absent observations.
    if "attendance" in df.columns:
        present_mask = (
            df["attendance"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("present")
        )
        df = df.loc[present_mask].copy()

    # Exclude incomplete observations from comparisons.
    df = df.dropna(subset=feature_columns)

    # Keep only actual anomaly flags.
    df["anomaly"] = pd.to_numeric(
        df["anomaly"],
        errors="coerce"
    )

    if "date" in df.columns:
        df["_sort_date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

        df = df.sort_values(
            "_sort_date",
            kind="stable",
            na_position="first"
        )

    df = df.reset_index(drop=True)

    unusual_positions = df.index[
        df["anomaly"] == -1
    ].tolist()

    if not unusual_positions:
        return []

    # Analyze the most recent unusual observation.
    unusual_position = unusual_positions[-1]
    unusual_row = df.iloc[unusual_position]

    # Compare only against eligible observations that
    # appear earlier in chronological order.
    previous_data = df.iloc[:unusual_position]

    if previous_data.empty:
        return []

    changes = []

    for column in feature_columns:

        current_value = unusual_row[column]

        previous_average = previous_data[column].mean()

        if (
            pd.isna(current_value)
            or pd.isna(previous_average)
        ):
            continue

        difference = float(
            current_value - previous_average
        )

        # Highlight differences of at least one rating point.
        if abs(difference) >= 1:

            changes.append({
                "feature": feature_names[column],
                "previous": round(
                    float(previous_average), 2
                ),
                "current": round(
                    float(current_value), 2
                ),
                "difference": round(
                    difference, 2
                ),
                "direction": (
                    "increased"
                    if difference > 0
                    else "decreased"
                )
            })

    return changes



# =========================================================
# GENERATE AI ANOMALY EXPLANATION
# =========================================================



def generate_anomaly_explanation(anomaly_changes):

    if not anomaly_changes:
        return (
            "The overall rating pattern was unusual, but no "
            "individual rating differed from earlier averages "
            "by at least one point. Review the recorded "
            "observation and consider follow-up if appropriate."
        )

    feature_names = [
        change["feature"]
        for change in anomaly_changes
    ]

    if len(feature_names) == 1:
        dimensions = feature_names[0]
    else:
        dimensions = (
            ", ".join(feature_names[:-1])
            + " and "
            + feature_names[-1]
        )

    return (
        "The unusual pattern includes noticeable differences "
        "in "
        + dimensions
        + ". The comparisons below show each current rating "
        "alongside its earlier average. These differences "
        "support teacher review; they do not establish a "
        "cause, improvement, deterioration, or medical diagnosis."
    )



# =========================================================
# GENERATE AI PROGRESS SUMMARY
# =========================================================

def generate_progress_summary(
    student_df,
    averages,
    anomaly_count
):
    # Count only complete observations marked Present.
    feature_columns = [
        "communication",
        "social_interaction",
        "learning",
        "hyperactivity",
        "emotional_state",
        "sound_sensitivity"
    ]

    if student_df.empty:
        return (
            "No observations are available for "
            "a progress summary."
        )

    # Work on a copy to avoid modifying the original data.
    summary_df = student_df.copy()

    # Check attendance.
    if "attendance" in summary_df.columns:
        present_mask = (
            summary_df["attendance"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("present")
        )
    else:
        present_mask = pd.Series(
            True,
            index=summary_df.index
        )

    # Ensure all required rating columns exist.
    for column in feature_columns:
        if column not in summary_df.columns:
            summary_df[column] = None

        summary_df[column] = pd.to_numeric(
            summary_df[column],
            errors="coerce"
        )

    # Keep only complete, present observations.
    complete_mask = (
        summary_df[feature_columns]
        .notna()
        .all(axis=1)
    )

    eligible_df = summary_df.loc[
        present_mask & complete_mask
    ]

    eligible_count = len(eligible_df)

    if eligible_count < 5:
        return (
            "Not enough complete, present-student observations "
            "are available for a reliable progress summary. "
            f"Currently, {eligible_count} eligible observation(s) "
            "are available; at least 5 are required."
        )

    # Calculate averages using eligible observations only.
    averages = {
        column: float(eligible_df[column].mean())
        for column in feature_columns
    }

    learning = averages["learning"]
    communication = averages["communication"]
    hyperactivity = averages["hyperactivity"]
    sound = averages["sound_sensitivity"]

    summary = (
        f"Based on {eligible_count} complete, present-student "
        "observations, the system has been monitoring the "
        "student's recorded learning and behavioral patterns. "
    )

    if learning >= 4:
        summary += (
            "The recorded learning ratings are relatively high. "
        )
    elif learning <= 2:
        summary += (
            "Lower learning ratings appear in the available "
            "observations. "
        )

    if communication >= 4:
        summary += (
            "Communication ratings are generally high. "
        )
    elif communication <= 2:
        summary += (
            "Lower communication ratings appear in the "
            "available observations. "
        )

    if hyperactivity >= 4:
        summary += (
            "Higher hyperactivity ratings have been recorded. "
        )

    if sound >= 4:
        summary += (
            "Higher sound-sensitivity ratings have been recorded. "
        )

    if anomaly_count > 0:
        summary += (
            f"The system flagged {anomaly_count} unusual "
            "observation pattern(s). Continued monitoring may "
            "help establish whether these patterns persist."
        )
    else:
        summary += (
            "No unusual patterns were flagged by the model "
            "in the analyzed observations."
        )

    return summary

# =========================================================
# AI TREND ANALYSIS
# =========================================================



def analyze_trends(student_df):

    feature_columns = [
        "learning",
        "communication",
        "social_interaction",
        "hyperactivity",
        "emotional_state",
        "sound_sensitivity"
    ]

    feature_names = {
        "learning": "Learning",
        "communication": "Communication",
        "social_interaction": "Social Interaction",
        "hyperactivity": "Hyperactivity",
        "emotional_state": "Emotional State",
        "sound_sensitivity": "Sound Sensitivity"
    }

    trends = []

    # Work on a copy.
    student_df = student_df.copy()

    # Ensure all required features exist.
    for column in feature_columns:
        if column not in student_df.columns:
            return trends

        student_df[column] = pd.to_numeric(
            student_df[column],
            errors="coerce"
        )

    # Include only present records when attendance is available.
    if "attendance" in student_df.columns:

        present_mask = (
            student_df["attendance"]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("present")
        )

        student_df = student_df.loc[
            present_mask
        ].copy()

    # Require complete ratings across all six features.
    student_df = student_df.dropna(
        subset=feature_columns
    )

    # Sort chronologically.
    if "date" in student_df.columns:

        student_df["date"] = pd.to_datetime(
            student_df["date"],
            errors="coerce"
        )

        student_df = student_df.sort_values(
            "date",
            ascending=True
        )

    student_df = student_df.reset_index(drop=True)

    # At least four complete observations are needed
    # for this earlier-versus-later comparison.
    if len(student_df) < 4:
        return trends

    # Split into earlier and later groups.
    midpoint = len(student_df) // 2

    earlier_data = student_df.iloc[:midpoint]
    later_data = student_df.iloc[midpoint:]

    for column in feature_columns:

        earlier_average = earlier_data[column].mean()
        later_average = later_data[column].mean()

        if (
            pd.isna(earlier_average)
            or pd.isna(later_average)
        ):
            continue

        difference = (
            later_average - earlier_average
        )

        if difference >= 0.5:
            direction = "increasing"
            label = "Increasing"
            symbol = "📈"

        elif difference <= -0.5:
            direction = "decreasing"
            label = "Decreasing"
            symbol = "📉"

        else:
            direction = "stable"
            label = "Stable"
            symbol = "➡️"

        trends.append({
            "feature": feature_names[column],
            "direction": direction,
            "label": label,
            "symbol": symbol,
            "earlier": round(
                float(earlier_average), 2
            ),
            "later": round(
                float(later_average), 2
            ),
            "difference": round(
                float(difference), 2
            )
        })

    return trends

# =========================================================
# AI OBSERVATION INSIGHTS
# =========================================================

def generate_observation_insights(
    student_df,
    trend_analysis,
    anomaly_count
):

    insights = []

    # -----------------------------------------------------
    # Check trend information
    # -----------------------------------------------------

    for trend in trend_analysis:

        feature = trend["feature"]
        label = trend["label"]

        if trend["direction"] == "increasing":

            if label == "Improving":

                insights.append({

                    "type":
                    "positive",

                    "icon":
                    "📈",

                    "title":
                    feature,

                    "message":
                    f"{feature} shows a higher "
                    "recorded value in the later "
                    "observations."
                })

            else:

                insights.append({

                    "type":
                    "attention",

                    "icon":
                    "⚠️",

                    "title":
                    feature,

                    "message":
                    f"Higher {feature.lower()} "
                    "values were recorded in "
                    "the later observations."
                })

        elif trend["direction"] == "decreasing":

            insights.append({

                "type":
                "attention",

                "icon":
                "⚠️",

                "title":
                feature,

                "message":
                f"{feature} shows a lower "
                "recorded value in the later "
                "observations."
            })

        else:

            insights.append({

                "type":
                "stable",

                "icon":
                "➡️",

                "title":
                feature,

                "message":
                f"{feature} remains relatively "
                "stable across the compared "
                "observations."
            })

    # -----------------------------------------------------
    # Monitoring suggestion
    # -----------------------------------------------------

    if anomaly_count > 0:

        monitoring_message = (
            "Continue recording observations to "
            "determine whether the detected "
            "patterns persist over time."
        )

    else:

        monitoring_message = (
            "Continue regular observations to "
            "build a more complete picture of "
            "the student's progress."
        )

    # -----------------------------------------------------
    # Return complete insight information
    # -----------------------------------------------------

    return {

        "items":
        insights,

        "monitoring_message":
        monitoring_message
    }


# =========================================================
# HOME
# =========================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )



@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "GET":
        return render_template(
            "login.html",
            selected_role="teacher",
            login_error=None,
            signup_success=request.args.get("signup_success")
        )

    role = request.form.get("role", "").strip().lower()

    if role not in ("teacher", "parent"):
        return render_template(
            "login.html",
            login_error="Please select a valid account role.",
            selected_role="teacher"
        )

    if not MONGO_CONNECTED or users_collection is None:
        return render_template(
            "login.html",
            login_error="The database is temporarily unavailable.",
            selected_role=role
        )

    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")

    if not email or not password:
        return render_template(
            "login.html",
            login_error="Please enter your email and password.",
            selected_role=role
        )

    try:
        user = users_collection.find_one({"email": email})

    except Exception:
        app.logger.exception("Login database lookup failed")
        return render_template(
            "login.html",
            login_error="Unable to contact the database. Please try again.",
            selected_role=role
        )

    if not user:
        return render_template(
            "login.html",
            login_error="Invalid email or password.",
            selected_role=role
        )

    # Verify the password saved during signup.
    try:
        password_correct = check_password_hash(
            user.get("password", ""),
            password
        )
    except (TypeError, ValueError):
        password_correct = False

    if not password_correct:
        return render_template(
            "login.html",
            login_error="Invalid email or password.",
            selected_role=role
        )

    # Ensure the selected role matches the registered account.
    if user.get("role") != role:
        return render_template(
            "login.html",
            login_error="This account is not registered for the selected role.",
            selected_role=role
        )

    # Create a fresh authenticated session.
        # Create a fresh authenticated session.
    session.clear()
    session["user_email"] = email
    session["user_role"] = role

    # Remember the login for 30 days if selected.
    session.permanent = request.form.get("remember") == "yes"

    if role == "teacher":
        print("TEACHER LOGIN SUCCESSFUL")
        return redirect(url_for("dashboard"))

    # Parent accounts must have a linked student.
    linked_student_id = str(
        user.get("linked_student_id", "")
    ).strip()

    if not linked_student_id:
        session.clear()
        return render_template(
            "login.html",
            login_error="No student is linked to this parent account. Please contact the teacher.",
            selected_role="parent"
        )

    session["linked_student_id"] = linked_student_id

    print("PARENT LOGIN SUCCESSFUL")

    return redirect(url_for("parent_dashboard"))

# =========================================
# FORGOT PASSWORD
# =========================================


# =========================================
# FORGOT PASSWORD — TEACHER AND PARENT
# =========================================

@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():

    if request.method == "GET":
        return render_template("forgot_password.html")

    email = request.form.get("email", "").strip().lower()

    if not email:
        flash("Please enter your registered email address.", "error")
        return redirect(url_for("forgot_password"))

    if not MONGO_CONNECTED or users_collection is None:
        flash(
            "The service is temporarily unavailable. Please try again later.",
            "error"
        )
        return redirect(url_for("forgot_password"))

    try:
        user = users_collection.find_one({
    "email": {
        "$regex": "^" + re.escape(email) + "$",
        "$options": "i"
    }
})
        app.logger.info(
    "FORGOT PASSWORD: account match = %s",
    bool(user)
)

        if user:
            account_email = user.get("email", "").strip()
            # Generate a signed token with a 15-minute expiry.
            token = password_reset_serializer.dumps(
    account_email,
    salt="password-reset"
)

            # Store only the SHA-256 hash of the token.
            token_hash = hashlib.sha256(
                token.encode("utf-8")
            ).hexdigest()

            # A new request invalidates the previous reset link.
            users_collection.update_one(
                {"_id": user["_id"]},
                {
                    "$set": {
                        "password_reset_token_hash": token_hash
                    }
                }
            )

            reset_url = url_for(
                "reset_password",
                token=token,
                _external=True
            )

            message = Message(
                subject="Autism Monitor - Password Reset",
                recipients=[account_email],
                body=(
                    "Hello,\n\n"
                    "We received a request to reset your "
                    "Autism Monitor password.\n\n"
                    "Use the link below to create a new password:\n\n"
                    f"{reset_url}\n\n"
                    "This link expires in 15 minutes and can only "
                    "be used once.\n\n"
                    "If you did not request this change, you can "
                    "ignore this email.\n\n"
                    "Autism Monitor Team"
                )
            )

            mail.send(message)
            app.logger.info("FORGOT PASSWORD: email send completed")

    except Exception:
        # Do not reveal whether an account exists.
        app.logger.exception("Password reset request failed")

    # Same confirmation whether or not the email is registered.
    flash(
        "If an account exists for that email, password-reset "
        "instructions will be sent.",
        "success"
    )

    return redirect(url_for("forgot_password"))


# =========================================
# RESET PASSWORD — SINGLE-USE TOKEN
# =========================================

@app.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token):

    # Verify signature and 15-minute expiry.
    try:
        email = password_reset_serializer.loads(
            token,
            salt="password-reset",
            max_age=900
        )

    except SignatureExpired:
        flash(
            "This reset link has expired. Please request a new one.",
            "error"
        )
        return redirect(url_for("forgot_password"))

    except BadSignature:
        flash(
            "This reset link is invalid. Please request a new one.",
            "error"
        )
        return redirect(url_for("forgot_password"))

    if not MONGO_CONNECTED or users_collection is None:
        flash(
            "The service is temporarily unavailable. Please try again later.",
            "error"
        )
        return redirect(url_for("forgot_password"))

    # Hash the presented token to compare it with MongoDB.
    token_hash = hashlib.sha256(
        token.encode("utf-8")
    ).hexdigest()

    # Only an unused token belonging to this email is accepted.
    try:
        user = users_collection.find_one({
            "email": email,
            "password_reset_token_hash": token_hash
        })

    except Exception:
        app.logger.exception("Password reset token lookup failed")
        flash(
            "Unable to verify this link right now. Please try again later.",
            "error"
        )
        return redirect(url_for("forgot_password"))

    if not user:
        flash(
            "This reset link is invalid or has already been used. "
            "Please request a new one.",
            "error"
        )
        return redirect(url_for("forgot_password"))

    if request.method == "GET":
        return render_template("reset_password.html", token=token)

    password = request.form.get("password", "")
    confirm_password = request.form.get("confirm_password", "")

    if len(password) < 8:
        flash(
            "Your new password must be at least 8 characters long.",
            "error"
        )
        return render_template("reset_password.html", token=token)

    if password != confirm_password:
        flash("The passwords do not match.", "error")
        return render_template("reset_password.html", token=token)

    try:
        # Atomically change the password and consume the token.
        # Only one request can successfully match this unused token.
        result = users_collection.update_one(
            {
                "_id": user["_id"],
                "email": email,
                "password_reset_token_hash": token_hash
            },
            {
                "$set": {
                    "password": generate_password_hash(password)
                },
                "$unset": {
                    "password_reset_token_hash": ""
                }
            }
        )

        if result.modified_count != 1:
            flash(
                "This reset link has already been used. "
                "Please request a new one.",
                "error"
            )
            return redirect(url_for("forgot_password"))

        flash(
            "Your password has been reset successfully. Please log in.",
            "success"
        )
        return redirect(url_for("login"))

    except Exception:
        app.logger.exception("Password reset database update failed")
        flash(
            "Unable to reset your password right now. Please try again later.",
            "error"
        )
        return render_template("reset_password.html", token=token)





@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET":
        return render_template("signup.html", error=None)

    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    role = request.form.get("role", "").strip().lower()

    if role not in ("teacher", "parent"):
        return render_template(
            "signup.html",
            error="Please select a valid account type."
        )

    if not name or not email or not password:
        return render_template(
            "signup.html",
            error="Please complete all required fields."
        )

    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        return render_template(
            "signup.html",
            error="Please enter a valid email address."
        )

    if len(password) < 8:
        return render_template(
            "signup.html",
            error="Password must contain at least 8 characters."
        )

    if not MONGO_CONNECTED or users_collection is None:
        return render_template(
            "signup.html",
            error="Database unavailable. Please try again later."
        )

    # Teacher registration requires the configured invitation code.
    if role == "teacher":
        entered_code = request.form.get("teacher_code", "").strip()
        expected_code = os.getenv("TEACHER_SIGNUP_CODE", "")

        if not expected_code or entered_code != expected_code:
            return render_template(
                "signup.html",
                error="Invalid teacher registration code."
            )

    # Parent registration must identify an existing student.
    student_id = None
    if role == "parent":
        student_id = request.form.get("student_id", "").strip()
        child_name = request.form.get("child_name", "").strip()

        if not student_id or not child_name:
            return render_template(
                "signup.html",
                error="Enter your child's name and roll number."
            )

        if students_collection is None:
            return render_template(
                "signup.html",
                error="Student records are unavailable."
            )

        try:
            student = students_collection.find_one({
                "student_id": student_id
            })
        except Exception:
            app.logger.exception("Student lookup failed during signup")
            student = None

        if not student:
            return render_template(
                "signup.html",
                error="Student roll number was not found."
            )

        actual_name = str(
            student.get("student_name", student.get("name", ""))
        ).strip()

        if " ".join(actual_name.casefold().split()) != " ".join(
            child_name.casefold().split()
        ):
            return render_template(
                "signup.html",
                error="The child's name and roll number do not match."
            )

    try:
        if users_collection.find_one({"email": email}):
            return render_template(
                "signup.html",
                error="An account with this email already exists."
            )

        user = {
            "name": name,
            "email": email,
            "password": generate_password_hash(password),
            "role": role
        }

        if role == "parent":
            user["linked_student_id"] = student_id

        users_collection.insert_one(user)

        return redirect(
         url_for(
          "login",
           signup_success="Account created successfully. Please log in."
    )
)

    except Exception:
        app.logger.exception("Account registration failed")
        return render_template(
            "signup.html",
            error="Registration failed. Please try again."
        )



# =========================================================
# DASHBOARD
# =========================================================

@app.route("/dashboard")
def dashboard():

    # =========================================
    # CHECK LOGIN SESSION
    # =========================================

    if "user_email" not in session:
        return redirect(url_for("login"))
        # Parents must use their own dashboard.
    if session.get("user_role") == "parent":
        return redirect(url_for("parent_dashboard"))

    # =========================================
    # LOAD OBSERVATIONS FROM MONGODB
    # =========================================

    try:

        mongo_records = list(
            observations_collection.find(
                {},
                {
                    "_id": 0
                }
            )
        )

        print(
            "Dashboard data loaded from MongoDB ✅"
        )

    except Exception as e:

        print(
            "MongoDB Dashboard read failed ❌"
        )

        print("Error:", e)

        return redirect(
            url_for("login")
        )

    # =========================================
    # CONVERT MONGODB DATA TO DATAFRAME
    # =========================================

    if not mongo_records:

        total_students = 0
        total_observations = 0
        anomaly_alerts = 0
        ai_alerts = []

    else:

        df = pd.DataFrame(
            mongo_records
        )

        # =====================================
        # BASIC DASHBOARD STATISTICS
        # =====================================

        total_students = (
            df["student_id"]
            .astype(str)
            .nunique()
        )

        total_observations = len(df)

        anomaly_alerts = 0

        ai_alerts = []

        # =====================================
        # ANALYZE EACH STUDENT
        # =====================================

        for student_id in df[
            "student_id"
        ].astype(str).unique():

            student_df = df[
                df["student_id"].astype(str)
                == str(student_id)
            ].copy()

            # Need at least 5 observations
            if len(student_df) < 5:

                continue

            # =================================
            # RUN AI ANOMALY DETECTION
            # =================================

            student_df = detect_anomalies(
                student_df
            )

            # =================================
            # COUNT UNUSUAL OBSERVATIONS
            # =================================

            student_anomaly_count = int(
                (
                    student_df["anomaly"] == -1
                ).sum()
            )

            anomaly_alerts += (
                student_anomaly_count
            )

            # =================================
            # CREATE AI ALERT
            # =================================

            if student_anomaly_count > 0:

                changes = (
                    analyze_anomaly_changes(
                        student_df
                    )
                )

                student_name = str(
                    student_df.iloc[0][
                        "student_name"
                    ]
                )

                
                # Find the latest observation flagged as unusual
                flagged_rows = student_df.loc[
                    student_df["anomaly"] == -1
                ].copy()

                latest_flagged_id = None
                latest_followup = None

                if not flagged_rows.empty:
                    flagged_rows["_date_sort"] = pd.to_datetime(
                        flagged_rows["date"],
                        errors="coerce"
                    )

                    flagged_rows = flagged_rows.sort_values(
                        "_date_sort",
                        kind="stable"
                    )

                    latest_flagged = flagged_rows.iloc[-1]

                    # The dashboard query excludes MongoDB _id,
                    # so locate the observation using its fields.
                    latest_flagged_date = str(
                        latest_flagged["date"]
                    )

                    latest_flagged_id = None

                    if (
                        MONGO_CONNECTED
                        and followups_collection is not None
                    ):
                        try:
                            matching_observation = (
                                observations_collection.find_one(
                                    {
                                        "student_id": str(student_id),
                                        "date": latest_flagged_date,
                                        "student_name": student_name
                                    },
                                    {"_id": 1}
                                )
                            )

                            if matching_observation:
                                latest_flagged_id = str(
                                    matching_observation["_id"]
                                )

                                latest_followup = (
                                    followups_collection.find_one({
                                        "student_id": str(student_id),
                                        "observation_id": latest_flagged_id
                                    })
                                )

                        except Exception as e:
                            print(
                                "AI alert follow-up lookup failed:",
                                e
                            )

                ai_alerts.append({
                    "student_id": student_id,
                    "student_name": student_name,
                    "changes": changes,
                    "anomaly_count": student_anomaly_count,
                    "flagged_observation_id": latest_flagged_id,
                    "followup": latest_followup
                })


    # =========================================
    # SEND DATA TO DASHBOARD
    # =========================================
        # =========================================
    # TEACHER FOLLOW-UP SUMMARY
    # =========================================

    
    # =========================================
    # TEACHER FOLLOW-UP SUMMARY
    # =========================================

    followup_summary = {
        "pending": 0,
        "reviewed": 0,
        "required": 0
    }

    if MONGO_CONNECTED and followups_collection is not None:
        try:
            # Load all saved follow-up records
            followup_records = list(
                followups_collection.find({})
            )

            # Count saved follow-up statuses
            for record in followup_records:
                status = record.get(
                    "review_status",
                    "Pending"
                )

                if status == "Reviewed":
                    followup_summary["reviewed"] += 1

                elif status == "Follow-up Required":
                    followup_summary["required"] += 1

            # Count observations without a saved follow-up
            reviewed_observation_ids = {
                str(record.get("observation_id", ""))
                for record in followup_records
            }

            all_observations = list(
                observations_collection.find(
                    {},
                    {
                        "_id": 1
                    }
                )
            )

            followup_summary["pending"] = sum(
                1
                for observation in all_observations
                if str(observation["_id"])
                not in reviewed_observation_ids
            )

        except Exception as e:
            print(
                "Follow-up summary could not be loaded:",
                e
            )

    return render_template(
    "dashboard.html",
    total_students=total_students,
    total_observations=total_observations,
    anomaly_alerts=anomaly_alerts,
    ai_alerts=ai_alerts,
    followup_summary=followup_summary
)
# =========================================================
# LOGOUT
# =========================================================

@app.route("/logout")
def logout():

    # Clear the logged-in user's session
    session.clear()

    # Return to login page
    return redirect(url_for("login"))

# =========================================================
# STUDENTS
# =========================================================


@app.route("/students")
def students():

    # =========================================
    # CHECK LOGIN SESSION
    # =========================================

    if "user_email" not in session:
        return redirect(url_for("login"))

    if session.get("user_role") == "parent":
        return redirect(url_for("parent_dashboard"))

    # =========================================
    # LOAD STUDENTS AND OBSERVATIONS
    # =========================================

    try:
        registered_students = list(
            students_collection.find(
                {},
                {
                    "_id": 0,
                    "student_id": 1,
                    "student_name": 1
                }
            )
        )

        mongo_records = list(
            observations_collection.find(
                {},
                {"_id": 0}
            )
        )

        print("Students and observations loaded from MongoDB")

    except Exception as e:
        print("MongoDB Students read failed:", e)
        return "Could not load student records. Please check the database connection.", 503

    # =========================================
    # BUILD STUDENT RECORDS
    # =========================================

    student_data = {}

    # First, include EVERY registered student,
    # even those without observations.
    for student in registered_students:
        student_id = str(
            student.get("student_id", "")
        ).strip()

        if not student_id:
            continue

        student_data[student_id] = {
            "student_id": student_id,
            "student_name": str(
                student.get("student_name", "Unknown")
            ),
            "observations": []
        }

    # Add observations to registered students.
    # Also retain older students who have observations
    # but were never manually registered.
    for record in mongo_records:
        student_id = str(
            record.get("student_id", "")
        ).strip()

        if not student_id:
            continue

        if student_id not in student_data:
            student_data[student_id] = {
                "student_id": student_id,
                "student_name": str(
                    record.get("student_name", "Unknown")
                ),
                "observations": []
            }

        student_data[student_id]["observations"].append(
            record
        )

    # =========================================
    # CALCULATE STUDENT STATISTICS
    # =========================================

    student_list = []

    for student_id, student in student_data.items():

        student_name = student["student_name"]
        student_records = student["observations"]

        observation_count = len(student_records)
        attendance_percentage = 0
        anomaly_count = 0

        if student_records:
            student_df = pd.DataFrame(student_records)

            if "attendance" in student_df.columns:
                present_days = (
                    student_df["attendance"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .eq("present")
                    .sum()
                )

                total_days = len(student_df)

                if total_days > 0:
                    attendance_percentage = round(
                        (present_days / total_days) * 100,
                        1
                    )

            # Preserve existing anomaly detection behavior.
            if observation_count >= 5:
                try:
                    analyzed_df = detect_anomalies(
                        student_df.copy()
                    )

                    if "anomaly" in analyzed_df.columns:
                        anomaly_count = int(
                            (analyzed_df["anomaly"] == -1).sum()
                        )

                except Exception as e:
                    print(
                        f"Anomaly analysis failed for {student_id}:",
                        e
                    )

        student_list.append({
            "student_id": student_id,
            "student_name": student_name,
            "observation_count": observation_count,
            "attendance_percentage": attendance_percentage,
            "anomaly_count": anomaly_count
        })

    # Show students in ID order.
    student_list.sort(
        key=lambda student: student["student_id"]
    )

    # =========================================
    # DISPLAY STUDENTS PAGE
    # =========================================

    return render_template(
        "students.html",
        students=student_list
    )



@app.route("/add_student", methods=["GET", "POST"])
def add_student():

    # Teachers only
    if "user_email" not in session:
        return redirect(url_for("login"))

    if session.get("user_role") != "teacher":
        return "Access denied.", 403

    # Check MongoDB connection
    if not MONGO_CONNECTED or students_collection is None:
        return "Database unavailable.", 503

    if request.method == "POST":

        student_id = request.form.get(
            "student_id", ""
        ).strip()

        student_name = request.form.get(
            "student_name", ""
        ).strip()

        # Validate fields
        if not student_id or not student_name:
            flash("Please enter both Student ID and Student Name.", "error")
            return render_template("add_student.html")

        try:
            # Prevent duplicate student IDs
            existing_student = students_collection.find_one({
                "student_id": student_id
            })

            if existing_student:
                flash("This Student ID already exists.", "error")
                return render_template("add_student.html")

            # Save student without requiring an observation
            students_collection.insert_one({
                "student_id": student_id,
                "student_name": student_name,
                "created_at": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            })

            flash("Student added successfully.", "success")
            return redirect(url_for("students"))

        except Exception as e:
            print("Add student failed:", e)
            flash("Could not add the student. Please try again.", "error")

    return render_template("add_student.html")


@app.route("/edit-student/<student_id>", methods=["GET", "POST"])
def edit_student(student_id):

    # Check login
    if "user_email" not in session:
        return redirect(url_for("login"))

    # Teachers only
    if session.get("user_role") != "teacher":
        return "Access denied.", 403

    # Check database
    if not MONGO_CONNECTED or students_collection is None:
        return "Database unavailable.", 503

    try:
        student = students_collection.find_one(
            {"student_id": student_id},
            {"_id": 0}
        )

        if not student:
            flash("Student not found.", "error")
            return redirect(url_for("students"))

        if request.method == "POST":
            student_name = request.form.get(
                "student_name", ""
            ).strip()

            if not student_name:
                flash("Student name cannot be empty.", "error")
                return render_template(
                    "edit_student.html",
                    student=student
                )

            students_collection.update_one(
                {"student_id": student_id},
                {"$set": {"student_name": student_name}}
            )

            # Keep the name consistent in existing observations.
            observations_collection.update_many(
                {"student_id": student_id},
                {"$set": {"student_name": student_name}}
            )

            flash("Student details updated successfully.", "success")
            return redirect(url_for("students"))

        return render_template(
            "edit_student.html",
            student=student
        )

    except Exception as e:
        print("Edit student failed:", e)
        return "Could not update student details. Please check the terminal.", 500


@app.route("/delete-student/<student_id>", methods=["POST"])
def delete_student(student_id):

    # Check login
    if "user_email" not in session:
        return redirect(url_for("login"))

    # Teachers only
    if session.get("user_role") != "teacher":
        return "Access denied.", 403

    # Check database
    if not MONGO_CONNECTED or students_collection is None:
        return "Database unavailable.", 503

    try:
        student = students_collection.find_one(
            {"student_id": student_id}
        )

        if not student:
            flash("Student not found.", "error")
            return redirect(url_for("students"))

        students_collection.delete_one(
            {"student_id": student_id}
        )

        flash(
            "Student record deleted. Existing observations were preserved.",
            "success"
        )

    except Exception as e:
        print("Delete student failed:", e)
        flash("Could not delete the student.", "error")

    return redirect(url_for("students"))



# =========================================================
# STUDENT PROGRESS
# =========================================================

@app.route("/student/<student_id>/upload-photo", methods=["POST"])
def upload_student_photo(student_id):

    # Require a logged-in user
    if "user_email" not in session:
        return redirect(url_for("login"))
        # Only teachers can add observations
    if session.get("user_role") != "teacher":
        return "Access denied.", 403

    if not MONGO_CONNECTED:
        return redirect(
            url_for("student_progress", student_id=student_id)
        )

    photo = request.files.get("student_photo")

    if not photo or not photo.filename:
        return redirect(
            url_for("student_progress", student_id=student_id)
        )

    # Validate file extension and image MIME type
    filename = secure_filename(photo.filename)
    extension = os.path.splitext(filename)[1].lower()

    allowed_types = {
        ".jpg": {"image/jpeg"},
        ".jpeg": {"image/jpeg"},
        ".png": {"image/png"},
        ".webp": {"image/webp"}
    }

    if (
        extension not in allowed_types
        or photo.mimetype not in allowed_types[extension]
    ):
        return redirect(
            url_for("student_progress", student_id=student_id)
        )

    # Read at most 5 MB plus one byte
    image_data = photo.stream.read(5 * 1024 * 1024 + 1)

    if not image_data or len(image_data) > 5 * 1024 * 1024:
        return redirect(
            url_for("student_progress", student_id=student_id)
        )

    # Generate a unique filename to avoid overwriting other photos
    new_filename = f"{uuid.uuid4().hex}{extension}"

    upload_folder = os.path.join(
        app.root_path,
        "static",
        "uploads",
        "student_photos"
    )

    os.makedirs(upload_folder, exist_ok=True)

    absolute_path = os.path.join(upload_folder, new_filename)
    relative_path = f"uploads/student_photos/{new_filename}"

    try:
        # Save the image
        with open(absolute_path, "wb") as image_file:
            image_file.write(image_data)

        # Store its relative path in the student document
        students_collection.update_one(
            {"student_id": str(student_id)},
            {
                "$set": {
                    "student_id": str(student_id),
                    "photo_path": relative_path
                }
            },
            upsert=True
        )

        print("Student photo saved successfully ✅")

    except Exception as e:
        print("Student photo upload failed ❌")
        print("Error:", e)

        # Remove the newly saved file if database update failed
        if os.path.exists(absolute_path):
            os.remove(absolute_path)

    return redirect(
        url_for("student_progress", student_id=student_id)
    )


@app.route("/student/<student_id>/remove-photo", methods=["POST"])
def remove_student_photo(student_id):

    if "user_email" not in session:
        return redirect(url_for("login"))
        # Only teachers can add observations
    if session.get("user_role") != "teacher":
        return "Access denied.", 403

    try:
        student = students_collection.find_one(
            {"student_id": str(student_id)},
            {"_id": 0, "photo_path": 1}
        )

        if student and student.get("photo_path"):
            photo_path = student["photo_path"]

            # Remove the saved image file
            absolute_path = os.path.join(
                app.root_path,
                "static",
                photo_path
            )

            if os.path.isfile(absolute_path):
                os.remove(absolute_path)

            # Remove only the photo reference from MongoDB
            students_collection.update_one(
                {"student_id": str(student_id)},
                {"$unset": {"photo_path": ""}}
            )

            print("Student photo removed successfully ✅")

    except Exception as e:
        print("Student photo removal failed ❌")
        print("Error:", e)

    return redirect(
        url_for("student_progress", student_id=student_id)
    )


@app.route("/student/<student_id>")
def student_progress(student_id):

    
    # -----------------------------------------------------
    # Check login
    # -----------------------------------------------------

    if "user_email" not in session:
        return redirect(url_for("login"))

    # -----------------------------------------------------
    # Restrict parents to their linked child only
    # Teachers can view any student's progress.
    # -----------------------------------------------------

    if session.get("user_role") == "parent":

        if not MONGO_CONNECTED or users_collection is None:
            return "Database unavailable. Please try again later.", 503

        try:
            parent_user = users_collection.find_one(
                {
                    "email": session.get("user_email"),
                    "role": "parent"
                },
                {
                    "_id": 0,
                    "linked_student_id": 1
                }
            )

        except Exception as e:
            print("Parent access verification failed:", e)
            return "Unable to verify parent access.", 503

        linked_student_id = (
            str(parent_user.get("linked_student_id", ""))
            if parent_user
            else ""
        )

        if not linked_student_id or linked_student_id != str(student_id):
            return "Access denied: this student is not linked to your account.", 403



    # -----------------------------------------------------
    # Load observations from MongoDB
    # -----------------------------------------------------

    try:

        mongo_records = list(
            observations_collection.find(
                {"student_id": str(student_id)},
                {"_id": 0}
            )
        )

        print("Student Progress loaded from MongoDB ✅")

    except Exception as e:

        print("MongoDB Student Progress read failed ❌")
        print("Error:", e)

        return redirect(url_for("students"))

    # -----------------------------------------------------
    # Check whether observations exist
    # -----------------------------------------------------

    if not mongo_records:
        return redirect(url_for("students"))

    # -----------------------------------------------------
    # Convert records to DataFrame
    # -----------------------------------------------------

    student_df = pd.DataFrame(mongo_records)

    # Ensure expected columns exist
    numeric_columns = [
        "communication",
        "social_interaction",
        "learning",
        "hyperactivity",
        "emotional_state",
        "sound_sensitivity"
    ]

    for column in numeric_columns:
        if column not in student_df.columns:
            student_df[column] = None

    if "attendance" not in student_df.columns:
        student_df["attendance"] = None

    if "remarks" not in student_df.columns:
        student_df["remarks"] = ""

    if "date" not in student_df.columns:
        student_df["date"] = ""

    if "student_name" not in student_df.columns:
        student_df["student_name"] = str(student_id)

    # -----------------------------------------------------
    # Sort records by date
    # -----------------------------------------------------

    student_df["date"] = student_df["date"].astype(str)

    student_df = (
        student_df
        .sort_values("date")
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # Student name
    # -----------------------------------------------------

    student_name = str(
        student_df.iloc[0]["student_name"]
    )

    # -----------------------------------------------------
    # Convert ratings to numeric values
    # -----------------------------------------------------

    for column in numeric_columns:

        student_df[column] = pd.to_numeric(
            student_df[column],
            errors="coerce"
        )

    # -----------------------------------------------------
    # Identify present records with complete ratings
    # -----------------------------------------------------

    present_mask = (
        student_df["attendance"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("present")
    )

    complete_ratings_mask = (
        student_df[numeric_columns]
        .notna()
        .all(axis=1)
    )

    analysis_mask = (
        present_mask
        & complete_ratings_mask
    )

    # Preserve all observations for history and attendance.
    # Use only complete present records for behavior analysis.
    analysis_df = (
        student_df.loc[analysis_mask]
        .copy()
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # Run anomaly detection on all records
    # The detector itself excludes absent/incomplete records.
    # -----------------------------------------------------

    student_df = detect_anomalies(student_df)

    # Rebuild the analysis subset after anomaly detection.
    # This subset is for progress summaries and trend analysis.
    analysis_df = (
        student_df.loc[analysis_mask]
        .copy()
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # Analyze unusual observation changes
    # -----------------------------------------------------

    if len(analysis_df) >= 2:

        anomaly_changes = analyze_anomaly_changes(
            analysis_df
        )

        anomaly_explanation = generate_anomaly_explanation(
            anomaly_changes
        )

    else:

        anomaly_changes = []
        anomaly_explanation = (
            "More complete present-student observations "
            "are needed to analyze changes over time."
        )

            # Get the date of the most recent unusual observation
    flagged_observation_date = None

    unusual_rows = analysis_df.loc[
        analysis_df["anomaly"] == -1
    ].copy()

    if not unusual_rows.empty:
        unusual_rows["_date_sort"] = pd.to_datetime(
            unusual_rows["date"],
            errors="coerce"
        )

        unusual_rows = unusual_rows.sort_values(
            "_date_sort",
            kind="stable"
        )

        flagged_observation_date = str(
            unusual_rows.iloc[-1]["date"]
        )
    
    # -----------------------------------------------------
    # Count unusual observations
    # -----------------------------------------------------

    anomaly_count = int(
        (student_df["anomaly"] == -1).sum()
    )

    # Count complete, present-student observations
    analysis_observation_count = len(analysis_df)

    # Minimum observations required for AI analysis
    minimum_ai_observations = 5

    ai_analysis_ready = (
        analysis_observation_count >= minimum_ai_observations
    )

    # -----------------------------------------------------
    # Calculate behavior averages
    # Exclude absent and incomplete records.
    # -----------------------------------------------------

    averages = {}

    for column in numeric_columns:

        if not analysis_df.empty:

            average_value = analysis_df[column].mean()

            averages[column] = (
                round(float(average_value), 2)
                if pd.notna(average_value)
                else None
            )

        else:

            averages[column] = None

    # -----------------------------------------------------
    # Generate progress summary
    # -----------------------------------------------------

    if len(analysis_df) >= 5:

        progress_summary = generate_progress_summary(
            analysis_df,
            averages,
            anomaly_count
        )

    else:

        progress_summary = (
            "Not enough complete present-student observations "
            "are available for a reliable progress summary. "
            "At least five complete observations are required."
        )

    # -----------------------------------------------------
    # Generate trend analysis
    # -----------------------------------------------------

    if len(analysis_df) >= 2:

        trend_analysis = analyze_trends(
            analysis_df
        )

    else:

        trend_analysis = (
            "More complete present-student observations "
            "are needed to identify progress trends."
        )

    # -----------------------------------------------------
    # Generate observation insights
    # -----------------------------------------------------

    if len(analysis_df) >= 2:

        observation_insights = generate_observation_insights(
            analysis_df,
            trend_analysis,
            anomaly_count
        )

    else:

        observation_insights = (
            "Additional complete observations are needed "
            "to generate meaningful observation insights."
        )

    # -----------------------------------------------------
    # Calculate attendance
    # Include all attendance records.
    # -----------------------------------------------------

    total_days = len(student_df)

    present_days = int(
        student_df["attendance"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("present")
        .sum()
    )

    attendance_percentage = (
        round((present_days / total_days) * 100, 1)
        if total_days > 0
        else 0
    )

    # -----------------------------------------------------
    # Recent observations
    # Include present and absent records.
    # -----------------------------------------------------

    recent_observations = (
        student_df
        .sort_values("date", ascending=False)
        .head(10)
        .to_dict("records")
    )

    # -----------------------------------------------------
    # Progress chart data
    # Use None for missing ratings instead of zero.
    # -----------------------------------------------------

    chart_data = student_df.sort_values("date")

    dates = chart_data["date"].astype(str).tolist()

    learning_scores = [
        (
            float(value)
            if pd.notna(value) and str(attendance).strip().lower() == "present"
            else None
        )
        for value, attendance in zip(
            chart_data["learning"],
            chart_data["attendance"]
        )
    ]

    communication_scores = [
        (
            float(value)
            if pd.notna(value) and str(attendance).strip().lower() == "present"
            else None
        )
        for value, attendance in zip(
            chart_data["communication"],
            chart_data["attendance"]
        )
    ]

    social_scores = [
        (
            float(value)
            if pd.notna(value) and str(attendance).strip().lower() == "present"
            else None
        )
        for value, attendance in zip(
            chart_data["social_interaction"],
            chart_data["attendance"]
        )
    ]


    # Load the student's saved profile photo
    student_record = students_collection.find_one(
        {"student_id": str(student_id)},
        {"_id": 0, "photo_path": 1}
    )

    student_photo = (
        student_record.get("photo_path", "")
        if student_record
        else ""
    )
    # -----------------------------------------------------
    # Render Student Progress page
    # -----------------------------------------------------

    return render_template(
        "student_progress.html",
        student_id=student_id,
        student_name=student_name,
        student_photo=student_photo,
        averages=averages,
        attendance_percentage=attendance_percentage,
        total_observations=len(student_df),
        analysis_observation_count=analysis_observation_count,
        minimum_ai_observations=minimum_ai_observations,
        ai_analysis_ready=ai_analysis_ready,
        anomaly_count=anomaly_count,
        flagged_observation_date=flagged_observation_date,
        anomaly_changes=anomaly_changes,
        anomaly_explanation=anomaly_explanation,
        progress_summary=progress_summary,
        trend_analysis=trend_analysis,
        observation_insights=observation_insights,
        recent_observations=recent_observations,
        dates=dates,
        learning_scores=learning_scores,
        communication_scores=communication_scores,
        social_scores=social_scores
    )


# =========================================================
# OBSERVATION HISTORY
# =========================================================

@app.route("/student/<student_id>/history")
def observation_history(student_id):

    # =========================================
    # CHECK LOGIN SESSION
    # =========================================

    if "user_email" not in session:
        return redirect(url_for("login"))

    # =========================================
    # RESTRICT PARENTS TO THEIR LINKED CHILD
    # TEACHERS CAN VIEW ANY STUDENT'S HISTORY
    # =========================================

    if session.get("user_role") == "parent":

        if not MONGO_CONNECTED or users_collection is None:
            return "Database unavailable. Please try again later.", 503

        try:
            parent_user = users_collection.find_one(
                {
                    "email": session.get("user_email"),
                    "role": "parent"
                },
                {
                    "_id": 0,
                    "linked_student_id": 1
                }
            )

        except Exception as e:
            print("Parent history access verification failed:", e)
            return "Unable to verify parent access.", 503

        linked_student_id = (
            str(parent_user.get("linked_student_id", ""))
            if parent_user
            else ""
        )

        if not linked_student_id or linked_student_id != str(student_id):
            return "Access denied: this student's history is not linked to your account.", 403

    # =========================================
    # LOAD OBSERVATIONS FROM MONGODB
    # =========================================

    try:

        mongo_records = list(
    observations_collection.find(
        {
            "student_id": str(student_id)
        }
    )
)
        print(
            "Observation history loaded from MongoDB ✅"
        )

    except Exception as e:

        print(
            "MongoDB history read failed ❌"
        )

        print("Error:", e)

        return redirect(
            url_for("students")
        )

    # =========================================
    # CHECK WHETHER OBSERVATIONS EXIST
    # =========================================

    if not mongo_records:

        return redirect(
            url_for("students")
        )

    # =========================================
    # CONVERT TO DATAFRAME
    # =========================================

    observations_df = pd.DataFrame(
        mongo_records
    )

    # =========================================
    # STUDENT NAME
    # =========================================

    student_name = str(
        observations_df.iloc[0][
            "student_name"
        ]
    )

    # =========================================
    # RUN AI ANOMALY DETECTION
    # =========================================

    if len(observations_df) >= 5:

        try:

            observations_df = (
                detect_anomalies(
                    observations_df
                )
            )

        except Exception as e:

            print(
                "Anomaly detection failed:"
            )

            print(e)

    # =========================================
    # SORT OBSERVATIONS
    # =========================================

    observations_df = (
        observations_df
        .sort_values(
            "date",
            ascending=False
        )
    )

    
    
    observations = observations_df.to_dict("records")

    # Load saved teacher follow-up records
    followup_map = {}

    if MONGO_CONNECTED and followups_collection is not None:
        try:
            followup_records = list(
                followups_collection.find({
                    "student_id": str(student_id)
                })
            )

            for record in followup_records:
                followup_map[str(record.get("observation_id", ""))] = record

        except Exception as e:
            print("Follow-up records could not be loaded:", e)

    # Attach the matching follow-up to each observation
    for observation in observations:
        observation_id = str(observation["_id"])
        observation["_id"] = observation_id

        observation["followup"] = followup_map.get(
            observation_id
        )


    # =========================================
    # DISPLAY HISTORY
    # =========================================

    return render_template(
        "observation_history.html",
        student_id=student_id,
        student_name=student_name,
        observations=observations
    )


# =========================================
# EDIT AN EXISTING OBSERVATION
# =========================================

@app.route(
    "/student/<student_id>/observation/<observation_id>/edit",
    methods=["GET", "POST"]
)
def edit_observation(student_id, observation_id):

    if "user_email" not in session:
       return redirect(url_for("login"))

# Only teachers can edit observations
    if session.get("user_role") != "teacher":
       return "Access denied.", 403
    if not MONGO_CONNECTED:
        return redirect(
            url_for("observation_history", student_id=student_id)
        )

    # Validate the MongoDB observation ID
    try:
        object_id = ObjectId(observation_id)
    except Exception:
        return redirect(
            url_for("observation_history", student_id=student_id)
        )

    # Fetch only the observation belonging to this student
    observation = observations_collection.find_one({
        "_id": object_id,
        "student_id": str(student_id)
    })

    if not observation:
        return redirect(
            url_for("observation_history", student_id=student_id)
        )

    rating_fields = [
        "communication",
        "social_interaction",
        "learning",
        "hyperactivity",
        "emotional_state",
        "sound_sensitivity"
    ]

    # Display the edit form
    if request.method == "GET":
        return render_template(
            "edit_observation.html",
            observation=observation,
            student_id=student_id,
            student_name=observation.get("student_name", "")
        )

    # Read submitted values
    date = request.form.get("date", "").strip()
    attendance = request.form.get("attendance", "").strip()
    remarks = request.form.get("remarks", "").strip()

    if not date:
        return render_template(
            "edit_observation.html",
            observation=observation,
            student_id=student_id,
            student_name=observation.get("student_name", ""),
            error_message="Please select an observation date."
        )

    if attendance not in ("Present", "Absent"):
        return render_template(
            "edit_observation.html",
            observation=observation,
            student_id=student_id,
            student_name=observation.get("student_name", ""),
            error_message="Please select valid attendance."
        )

    # Default all ratings to None.
    # Absent students must not receive behavior ratings.
    ratings = {field: None for field in rating_fields}

    if attendance == "Present":

        for field in rating_fields:
            value = request.form.get(field, "").strip()

            if value not in ("1", "2", "3", "4", "5"):
                return render_template(
                    "edit_observation.html",
                    observation=observation,
                    student_id=student_id,
                    student_name=observation.get("student_name", ""),
                    error_message=(
                        "For a present student, provide a rating "
                        "from 1 to 5 for every behavior field."
                    )
                )

            ratings[field] = int(value)

    # Update only the selected observation
        # Update only the selected observation
    try:
        observations_collection.update_one(
            {
                "_id": object_id,
                "student_id": str(student_id)
            },
            {
                "$set": {
                    "date": date,
                    "attendance": attendance,
                    "remarks": remarks,
                    "updated_at": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    **ratings
                },
                "$unset": {
                    "anomaly": "",
                    "anomaly_status": ""
                }
            }
        )
        print("Observation updated successfully ✅")

    except Exception as e:
        print("Observation update failed ❌")
        print("Error:", e)

        return render_template(
            "edit_observation.html",
            observation=observation,
            student_id=student_id,
            student_name=observation.get("student_name", ""),
            error_message="Unable to update the observation. Please try again."
        )

    flash("✓ Observation updated successfully!", "success")

    return redirect( 
    url_for("observation_history", student_id=student_id)
)


# =========================================
# DELETE AN EXISTING OBSERVATION
# =========================================


@app.route(
    "/student/<student_id>/observation/<observation_id>/delete",
    methods=["POST"]
)
def delete_observation(student_id, observation_id):

    # Check login session
    if "user_email" not in session:
        return redirect(url_for("login"))

    # Only teachers can delete observations
    if session.get("user_role") != "teacher":
        return "Access denied.", 403

    # Check MongoDB connection
    if not MONGO_CONNECTED:
        return redirect(
            url_for("observation_history", student_id=student_id)
        )

    try:
        # Validate the observation ID
        object_id = ObjectId(observation_id)

        # Delete only the selected observation belonging to this student
        result = observations_collection.delete_one({
            "_id": object_id,
            "student_id": str(student_id)
        })

        if result.deleted_count == 1:
            print("Observation deleted successfully.")
            flash("Observation deleted successfully!", "success")
        else:
            print("No matching observation was deleted.")
            flash("Observation not found.", "error")

    except Exception as e:
        print("Observation deletion failed.")
        print("Error:", e)
        flash("Unable to delete the observation. Please try again.", "error")

    return redirect(
        url_for("observation_history", student_id=student_id)
    )


@app.route(
    "/student/<student_id>/observation/<observation_id>/followup",
    methods=["POST"]
)
def save_observation_followup(student_id, observation_id):

    # Check login session
    if "user_email" not in session:
        return redirect(url_for("login"))

    # Only teachers can save follow-up reviews
    if session.get("user_role") != "teacher":
        return "Access denied.", 403

    # Check MongoDB connection
    if not MONGO_CONNECTED:
        flash("MongoDB is unavailable. Follow-up was not saved.", "error")
        return redirect(
            url_for("observation_history", student_id=student_id)
        )

    status = request.form.get("followup_status", "Pending").strip()
    notes = request.form.get("followup_notes", "").strip()
    reobserve = request.form.get("reobserve") == "yes"

    allowed_statuses = [
        "Pending",
        "Reviewed",
        "Follow-up Required"
    ]

    if status not in allowed_statuses:
        flash("Invalid follow-up status.", "error")
        return redirect(
            url_for("observation_history", student_id=student_id)
        )

    try:
        observation_object_id = ObjectId(observation_id)

        # Confirm the observation belongs to this student
        observation = observations_collection.find_one({
            "_id": observation_object_id,
            "student_id": str(student_id)
        })

        if not observation:
            flash("Observation not found.", "error")
            return redirect(
                url_for("observation_history", student_id=student_id)
            )

        followups_collection.update_one(
            {
                "student_id": str(student_id),
                "observation_id": str(observation_object_id)
            },
            {
                "$set": {
                    "student_id": str(student_id),
                    "observation_id": str(observation_object_id),
                    "review_status": status,
                    "teacher_notes": notes,
                    "reobserve": reobserve,
                    "reviewed_by": session.get("user_email")
                }
            },
            upsert=True
        )

        flash("Teacher follow-up saved successfully!", "success")

    except Exception as e:
        print("Saving teacher follow-up failed:", e)
        flash("Unable to save the follow-up. Please try again.", "error")

    return redirect(
        url_for("observation_history", student_id=student_id)
    )

@app.route(
    "/student/<student_id>/observation/<observation_id>/followup",
    methods=["GET"]
)
def teacher_followup(student_id, observation_id):

    if "user_email" not in session:
        return redirect(url_for("login"))

    if session.get("user_role") != "teacher":
        return "Access denied.", 403

    if not MONGO_CONNECTED:
        flash("MongoDB is unavailable.", "error")
        return redirect(
            url_for("observation_history", student_id=student_id)
        )

    try:
        observation_object_id = ObjectId(observation_id)

        observation = observations_collection.find_one({
            "_id": observation_object_id,
            "student_id": str(student_id)
        })

        if not observation:
            flash("Observation not found.", "error")
            return redirect(
                url_for("observation_history", student_id=student_id)
            )

        followup = followups_collection.find_one({
            "student_id": str(student_id),
            "observation_id": str(observation_object_id)
        })

        return render_template(
            "teacher_followup.html",
            student_id=student_id,
            student_name=observation.get("student_name", "Student"),
            observation_date=observation.get("date", "Not available"),
            observation_id=str(observation_object_id),
            followup=followup
        )

    except Exception as e:
        print("Unable to open teacher follow-up:", e)
        flash("Unable to open the follow-up page.", "error")
        return redirect(
            url_for("observation_history", student_id=student_id)
        )


# =========================================================
# ADD OBSERVATION
# =========================================================


@app.route("/add-observation", methods=["GET", "POST"])

def add_observation():

    # Check login session
    if "user_email" not in session:
        return redirect(url_for("login"))
        # Only teachers can add observations
    if session.get("user_role") != "teacher":
        return "Access denied.", 403

    # Ensure MongoDB is connected before using collections
    if not MONGO_CONNECTED:
        return render_template(
            "add_observation.html",
            error_message=(
                "Database connection is unavailable. "
                "Please try again when the database is connected."
            )
        )

    # Display the form on GET
    if request.method == "GET":
        return render_template("add_observation.html")

    # Read form values
    student_id = request.form.get("student_id", "").strip()
    student_name = request.form.get("student_name", "").strip()
    attendance = request.form.get("attendance", "").strip()
    remarks = request.form.get("remarks", "").strip()

    # Validate student information
    if not student_id or not student_name:
        return render_template(
            "add_observation.html",
            error_message="Please enter the Student ID and Student Name."
        )

    if attendance not in ("Present", "Absent"):
        return render_template(
            "add_observation.html",
            error_message="Please select valid attendance."
        )

    rating_fields = [
        "communication",
        "social_interaction",
        "learning",
        "hyperactivity",
        "emotional_state",
        "sound_sensitivity"
    ]

    # Absent students receive missing ratings, not fabricated values
    ratings = {field: None for field in rating_fields}

    # Validate ratings only for present students
    if attendance == "Present":
        for field in rating_fields:
            value = request.form.get(field, "").strip()

            if value not in ("1", "2", "3", "4", "5"):
                return render_template(
                    "add_observation.html",
                    error_message=(
                        "Please provide a valid rating from 1 to 5 "
                        "for every observation field."
                    )
                )

            ratings[field] = int(value)

    # Prepare the observation record
    new_record = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "student_id": student_id,
        "student_name": student_name,
        "attendance": attendance,
        "remarks": remarks,
        **ratings
    }

    # Save the observation
    try:
        observations_collection.insert_one(new_record)
        print("Observation saved to MongoDB ✅")
        print("Attendance:", attendance)

    except Exception as e:
        print("MongoDB observation save failed ❌")
        print("Error:", e)

        return render_template(
            "add_observation.html",
            error_message=(
                "Unable to save the observation. Please try again."
            )
        )

    # Synchronize the student record
    try:
        students_collection.update_one(
            {"student_id": student_id},
            {
                "$set": {
                    "student_id": student_id,
                    "student_name": student_name
                }
            },
            upsert=True
        )

        print("Student record synchronized with MongoDB ✅")

    except Exception as e:
        print("Student MongoDB sync failed ❌")
        print("Error:", e)

    return redirect(url_for("dashboard"))



    # =========================================
    # DISPLAY FORM
    # =========================================

    return render_template(
        "add_observation.html"
    )



@app.route("/parent-dashboard")
def parent_dashboard():

    # ==========================================
    # CHECK LOGIN
    # ==========================================

    if "user_email" not in session:
        return redirect(url_for("login"))

    # Only parents can access this page
    if session.get("user_role") != "parent":
        return "Access denied.", 403

    # ==========================================
    # GET THE STUDENT VERIFIED DURING LOGIN
    # ==========================================

    student_id = str(
        session.get("linked_student_id", "")
    ).strip()

    if not student_id:
        return redirect(url_for("login"))

    # ==========================================
    # CHECK DATABASE CONNECTION
    # ==========================================

    if (
        not MONGO_CONNECTED
        or students_collection is None
    ):
        return "Database unavailable. Please try again later.", 503

    try:

        # Fetch only the verified student
        student = students_collection.find_one(
            {"student_id": student_id},
            {"_id": 0}
        )

        if not student:
            return (
                "The verified student could not be found.",
                404
            )

        # Count observations belonging to this student
        observation_count = (
            observations_collection.count_documents(
                {"student_id": student_id}
            )
            if observations_collection is not None
            else 0
        )

        return render_template(
            "parent_dashboard.html",
            student=student,
            student_id=student_id,
            observation_count=observation_count
        )

    except Exception as e:
        print("Parent dashboard error:", e)
        return "Unable to load the parent dashboard.", 500


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )