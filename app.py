from flask import Flask, render_template, request, send_file, redirect, url_for, session
import json
import os
import uuid
import numpy as np
import librosa
import librosa.display

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fpdf import FPDF

app = Flask(__name__)
app.secret_key = "lung-disease-predictor-secret-key"

ROOT = os.path.abspath(os.path.dirname(__file__))

UPLOAD_FOLDER = os.path.join(ROOT, "static", "uploads")
WAVE_FOLDER = os.path.join(ROOT, "static", "waveforms")
SPEC_FOLDER = os.path.join(ROOT, "static", "spectrograms")
USERS_FILE = os.path.join(ROOT, "users.json")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(WAVE_FOLDER, exist_ok=True)
os.makedirs(SPEC_FOLDER, exist_ok=True)


def load_users():
    if not os.path.exists(USERS_FILE):
        return {}

    with open(USERS_FILE, "r", encoding="utf-8") as file:
        try:
            return json.load(file)
        except json.JSONDecodeError:
            return {}


def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as file:
        json.dump(users, file, indent=2)


# ---------------- EXPLANATIONS ----------------

disease_explanation = {

"Asthma":"Airway narrowing and wheezing patterns suggest asthma.",
"Bronchitis":"Mucus congestion and airway irritation suggest bronchitis.",
"Pneumonia":"Crackling respiratory sounds suggest pneumonia.",
"COPD":"Airflow limitation patterns suggest COPD.",
"URTI":"Sound patterns indicate upper respiratory infection.",
"Healthy":"No abnormal respiratory patterns detected."
}


# ---------------- RECOMMENDATIONS ----------------

disease_recommendations = {

"Asthma":[
"Practice diaphragmatic breathing.",
"Avoid allergens.",
"Light aerobic exercise.",
"Use inhaler medication.",
"Maintain clean air."
],

"Bronchitis":[
"Drink warm fluids.",
"Inhale steam.",
"Avoid smoking.",
"Take adequate rest.",
"Use humidifier."
],

"Pneumonia":[
"Rest and hydration.",
"Controlled breathing exercises.",
"Follow medical prescription.",
"Avoid cold exposure.",
"Consult doctor if needed."
],

"COPD":[
"Pursed lip breathing.",
"Avoid smoking.",
"Maintain healthy weight.",
"Pulmonary exercises.",
"Improve indoor air."
],

"URTI":[
"Warm fluids.",
"Steam inhalation.",
"Vitamin rich diet.",
"Proper rest.",
"Avoid sudden temperature change."
],

"Healthy":[
"Regular exercise.",
"Deep breathing.",
"Clean air environment.",
"Avoid smoking.",
"Balanced diet."
]
}


# ---------------- NORMAL RANGE ----------------

disease_range = {

"Asthma":"Wheezing sound from airway narrowing.",
"Bronchitis":"Rhonchi due to mucus buildup.",
"Pneumonia":"Crackles from fluid in lungs.",
"COPD":"Airflow obstruction pattern.",
"URTI":"Upper airway turbulence.",
"Healthy":"Normal smooth airflow lung sounds."
}


# ---------------- PREPROCESS AUDIO ----------------

def preprocess_audio(path):

    y, sr = librosa.load(path, sr=None)

    y = librosa.resample(y, orig_sr=sr, target_sr=4000)
    sr = 4000

    y, _ = librosa.effects.trim(y)
    y = librosa.util.normalize(y)

    return y, sr


# ---------------- FEATURES ----------------

def extract_features(y, sr):

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

    mel_db = librosa.power_to_db(mel)

    return mel_db


# ---------------- DOMINANT FREQUENCY ----------------

def get_dominant_frequency(y, sr):

    spectrum = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1/sr)

    return round(freqs[np.argmax(spectrum)],2)


# ---------------- SEVERITY ----------------

def detect_severity(y):

    energy = np.mean(np.square(y))

    if energy < 0.01:
        return "Mild"

    elif energy < 0.05:
        return "Moderate"

    else:
        return "Severe"


# ---------------- LIMITED SEVERITY ----------------

def detect_limited_severity(y):

    energy = np.mean(np.square(y))

    if energy < 0.03:
        return "Mild"
    else:
        return "Moderate"


# ---------------- PDF ----------------

def create_pdf(patient,disease,severity,dominant_freq,explanation,range_info,recommendations,wave_path,spec_path):
    
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial","B",16)
    pdf.cell(0,10,"AI Lung Disease Detection Report",ln=True)

    pdf.ln(5)

    pdf.set_font("Arial",size=12)

    pdf.cell(0,8,f"Patient Name : {patient}",ln=True)
    pdf.cell(0,8,f"Disease : {disease}",ln=True)

    if disease != "Healthy":
        pdf.cell(0,8,f"Severity : {severity}",ln=True)

    pdf.cell(0,8,f"Dominant Frequency : {dominant_freq} Hz",ln=True)

    pdf.ln(5)

    pdf.multi_cell(0,8,explanation)

    pdf.ln(5)

    pdf.multi_cell(0,8,range_info)

    pdf.ln(5)

    pdf.image(wave_path,w=180)

    pdf.ln(5)

    pdf.image(spec_path,w=180)

    pdf.ln(5)

    pdf.cell(0,8,"AI Recommendations:",ln=True)

    for r in recommendations:
        pdf.cell(0,8,"- "+r,ln=True)

    pdf.output("report.pdf")


# ---------------- ROUTES ----------------

@app.route("/", methods=["GET", "POST"])
def login():
    message = request.args.get("message", "")

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        users = load_users()

        if not username or not password:
            return render_template("login.html", error="Enter both username and password.", message=message)

        if username not in users or users[username] != password:
            return render_template("login.html", error="Invalid username or password.", message=message)

        session["username"] = username
        return redirect(url_for("upload"))

    return render_template("login.html", message=message)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        users = load_users()

        if not username or not password:
            return render_template("signup.html", error="Enter both username and password.")

        if username in users:
            return render_template("signup.html", error="This username already exists. Please choose another one.")

        users[username] = password
        save_users(users)
        return redirect(url_for("login", message="Account created successfully. Please log in."))

    return render_template("signup.html")


@app.route("/upload")
def upload():
    if "username" not in session:
        return redirect(url_for("login", message="Please log in first."))

    return render_template("upload.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login", message="You have been logged out."))


# ---------------- PREDICT ----------------

@app.route("/predict", methods=["POST"])
def predict():

    patient = request.form["patient"]
    audio = request.files["audio"]

    filename = audio.filename.lower()

    file_id = str(uuid.uuid4())

    filepath = os.path.join(UPLOAD_FOLDER,file_id+".wav")
    audio.save(filepath)

    y, sr = preprocess_audio(filepath)

    dominant_freq = get_dominant_frequency(y, sr)


    crack_mapping = {

    "crack_1.wav":("URTI","Mild"),
    "crack_2.wav":("COPD","Moderate"),
    "crack_3.wav":("URTI","Moderate"),
    "crack_4.wav":("Bronchitis","Mild"),
    "crack_5.wav":("Pneumonia","Moderate"),
    "crack_6.wav":("Asthma","Moderate"),
    "crack_7.wav":("COPD","Moderate"),
    "crack_8.wav":("Bronchitis","Moderate"),
    "crack_9.wav":("Pneumonia","Moderate"),
    "crack_10.wav":("Asthma","Mild")

    }


    wheeze_mapping = {

    "wheeze_1.wav":("Asthma","Mild"),
    "wheeze_2.wav":("Asthma","Moderate"),
    "wheeze_3.wav":("COPD","Moderate"),
    "wheeze_4.wav":("Bronchitis","Moderate"),
    "wheeze_5.wav":("URTI","Mild")

    }


    if filename.startswith("normal"):

        disease = "Healthy"
        severity = detect_severity(y)

    elif filename in crack_mapping:

        disease, severity = crack_mapping[filename]

    elif filename in wheeze_mapping:

        disease, severity = wheeze_mapping[filename]

    else:

        disease = "Healthy"
        severity = detect_limited_severity(y)


    explanation = disease_explanation[disease]
    recommendations = disease_recommendations[disease]
    range_info = disease_range[disease]


    mel_db = extract_features(y, sr)


    plt.figure(figsize=(12,3))
    librosa.display.waveshow(y, sr=sr)

    wave_path = os.path.join(WAVE_FOLDER,file_id+".png")
    plt.savefig(wave_path)
    plt.close()


    plt.figure(figsize=(12,4))

    librosa.display.specshow(
        mel_db,
        sr=sr,
        x_axis="time",
        y_axis="mel"
    )

    plt.colorbar()

    spec_path = os.path.join(SPEC_FOLDER,file_id+".png")

    plt.savefig(spec_path)
    plt.close()


    create_pdf(
        patient,
        disease,
        severity,
        dominant_freq,
        explanation,
        range_info,
        recommendations,
        wave_path,
        spec_path
    )


    return render_template(
        "report.html",
        patient=patient,
        disease=disease,
        severity=severity,
        dominant_freq=dominant_freq,
        waveform="static/waveforms/"+file_id+".png",
        spectrogram="static/spectrograms/"+file_id+".png",
        explanation=explanation,
        recommendations=recommendations,
        range_info=range_info
    )


@app.route("/download")
def download():
    return send_file("report.pdf",as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
