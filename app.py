from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_file
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from bson.objectid import ObjectId
from pymongo import MongoClient
from datetime import datetime
from fpdf import FPDF
import os
import re

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "change_this_secret_key")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY if PINECONE_API_KEY else ""
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY if OPENAI_API_KEY else ""

client = MongoClient(MONGO_URI)
db = client["medical_chatbot_db"]

users_collection = db["users"]
chat_sessions_collection = db["chat_sessions"]
chat_messages_collection = db["chat_messages"]

users_collection.create_index("email", unique=True)
chat_sessions_collection.create_index("user_id")
chat_messages_collection.create_index("session_id")

embeddings = download_hugging_face_embeddings()

index_name = "aethercuraai"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

chatModel = ChatOpenAI(
    model="openai/gpt-oss-120b",
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENAI_API_KEY
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))

        user = users_collection.find_one({"_id": ObjectId(session["user_id"])})
        if not user or user.get("role") != "admin":
            flash("Access denied. Admin only.")
            return redirect(url_for("chat_page"))
        return f(*args, **kwargs)
    return decorated_function


def create_chat_session(user_id, title="New Chat"):
    result = chat_sessions_collection.insert_one({
        "user_id": ObjectId(user_id),
        "title": title,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    })
    return str(result.inserted_id)


def save_message(chat_session_id, sender, message):
    chat_messages_collection.insert_one({
        "session_id": ObjectId(chat_session_id),
        "sender": sender,
        "message": message,
        "created_at": datetime.utcnow()
    })
    chat_sessions_collection.update_one(
        {"_id": ObjectId(chat_session_id)},
        {"$set": {"updated_at": datetime.utcnow()}}
    )


def get_user_sessions(user_id):
    return list(
        chat_sessions_collection.find(
            {"user_id": ObjectId(user_id)}
        ).sort("updated_at", -1)
    )


def get_session_messages(chat_session_id):
    return list(
        chat_messages_collection.find(
            {"session_id": ObjectId(chat_session_id)}
        ).sort("created_at", 1)
    )


def detect_mode(message: str) -> str:
    text = message.lower()
    if any(word in text for word in ["medicine", "tablet", "dose", "drug", "paracetamol", "ibuprofen"]):
        return "Medicine Info"
    if any(word in text for word in ["diet", "exercise", "sleep", "water", "lifestyle", "weight"]):
        return "Lifestyle Advice"
    if any(word in text for word in ["emergency", "urgent", "chest pain", "breathing", "unconscious", "bleeding"]):
        return "Emergency Precautions"
    if any(word in text for word in ["symptom", "fever", "cough", "pain", "headache", "rash", "vomit"]):
        return "Symptoms Check"
    return "General Health"


def detect_emergency(text: str) -> bool:
    text = text.lower()
    emergency_terms = [
        "chest pain", "shortness of breath", "difficulty breathing", "breathing trouble",
        "unconscious", "seizure", "stroke", "severe bleeding", "heart attack",
        "suicidal", "passed out", "fainted", "blue lips"
    ]
    return any(term in text for term in emergency_terms)


def detect_severity(user_message: str, severity_input: str = "", answer: str = ""):
    full_text = f"{user_message} {answer}".lower()
    s = severity_input.lower().strip()

    if detect_emergency(full_text):
        return "Emergency", "#ff4d6d"

    if s in ["severe", "very severe", "10", "9", "8"]:
        return "High", "#ff8c42"

    if s in ["moderate", "7", "6", "5"]:
        return "Moderate", "#ffd166"

    if s in ["mild", "low", "4", "3", "2", "1"]:
        return "Low", "#42d392"

    high_terms = ["high fever", "severe pain", "vomiting blood", "fainting", "persistent chest pain"]
    moderate_terms = ["fever", "vomiting", "rash", "headache", "cough", "body pain"]

    if any(term in full_text for term in high_terms):
        return "High", "#ff8c42"
    if any(term in full_text for term in moderate_terms):
        return "Moderate", "#ffd166"
    return "Low", "#42d392"


def build_summary(intake_data, user_message, answer, mode, severity_label):
    age = intake_data.get("age", "Not provided")
    gender = intake_data.get("gender", "Not provided")
    symptom = intake_data.get("symptom", "Not provided")
    duration = intake_data.get("duration", "Not provided")
    existing_conditions = intake_data.get("conditions", "Not provided")

    precautions = "Monitor symptoms, stay hydrated, rest well, and consult a doctor if symptoms worsen."
    doctor_advice = "Consult a doctor if symptoms persist, worsen, or new warning signs appear."

    if severity_label == "Emergency":
        precautions = "Seek emergency medical help immediately. Do not rely only on AI guidance."
        doctor_advice = "Contact emergency services or visit the nearest hospital now."
    elif severity_label == "High":
        precautions = "Avoid self-medication without guidance. Rest, hydrate, and seek prompt medical evaluation."
        doctor_advice = "Consult a doctor as soon as possible."
    elif severity_label == "Moderate":
        precautions = "Track symptoms carefully, rest, hydrate, and avoid triggers if known."
        doctor_advice = "Consult a doctor if symptoms continue or become more severe."

    return {
        "age": age,
        "gender": gender,
        "primary_concern": symptom if symptom != "Not provided" else user_message[:80],
        "duration": duration,
        "existing_conditions": existing_conditions,
        "mode": mode,
        "severity": severity_label,
        "precautions": precautions,
        "doctor_advice": doctor_advice
    }


def nearby_care_message(severity_label):
    if severity_label == "Emergency":
        return "Nearest-care advice: Visit the nearest hospital or emergency room immediately, or contact local emergency services."
    if severity_label == "High":
        return "Nearest-care advice: Consider visiting a nearby doctor, urgent care, or hospital for proper evaluation."
    return "Nearest-care advice: If needed, consult a nearby clinic or doctor for a medical examination."

def safe_pdf_text(text):
    if text is None:
        return ""
    text = str(text)

    replacements = {
        "\u202f": " ",
        "\u00a0": " ",
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2022": "-",
        "\u2026": "...",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text.encode("latin-1", "ignore").decode("latin-1")

@app.route('/')
def home():
    return render_template('index.html')


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()
        login_type = request.form.get("login_type", "user").strip().lower()

        user = users_collection.find_one({"email": email})

        if not user:
            flash("Invalid email or password.")
            return redirect(url_for("login"))

        if not user.get("is_active", True):
            flash("Your account is disabled.")
            return redirect(url_for("login"))

        if not check_password_hash(user["password_hash"], password):
            flash("Invalid email or password.")
            return redirect(url_for("login"))

        if login_type == "admin" and user.get("role") != "admin":
            flash("This account does not have admin access.")
            return redirect(url_for("login"))

        session["user_id"] = str(user["_id"])
        session["username"] = user["username"]
        session["email"] = user["email"]
        session["role"] = user.get("role", "user")

        existing_sessions = get_user_sessions(session["user_id"])
        if existing_sessions:
            session["chat_session_id"] = str(existing_sessions[0]["_id"])
        else:
            session["chat_session_id"] = create_chat_session(session["user_id"])

        if user.get("role") == "admin" and login_type == "admin":
            return redirect(url_for("admin_panel"))

        return redirect(url_for("chat_page"))

    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()

        if not username or not email or not password or not confirm_password:
            flash("All fields are required.")
            return redirect(url_for("signup"))

        if password != confirm_password:
            flash("Passwords do not match.")
            return redirect(url_for("signup"))

        existing_user = users_collection.find_one({"email": email})
        if existing_user:
            flash("Email already registered. Please login.")
            return redirect(url_for("login"))

        users_collection.insert_one({
            "username": username,
            "email": email,
            "password_hash": generate_password_hash(password),
            "role": "user",
            "is_active": True,
            "created_at": datetime.utcnow()
        })

        flash("Signup successful. Please login.")
        return redirect(url_for("login"))

    return render_template("signup.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.")
    return redirect(url_for("home"))

@app.route("/chat")
@login_required
def chat_page():
    user_sessions = get_user_sessions(session["user_id"])
    current_session_id = session.get("chat_session_id")

    messages = []
    if current_session_id:
        messages = get_session_messages(current_session_id)

    return render_template(
        "chat.html",
        username=session.get("username"),
        chat_sessions=user_sessions,
        messages=messages,
        current_session_id=current_session_id
    )


@app.route("/new_chat")
@login_required
def new_chat():
    new_session_id = create_chat_session(session["user_id"], "New Chat")
    session["chat_session_id"] = new_session_id
    return redirect(url_for("chat_page"))


@app.route("/chat/<session_id>")
@login_required
def load_chat(session_id):
    chat_obj = chat_sessions_collection.find_one({
        "_id": ObjectId(session_id),
        "user_id": ObjectId(session["user_id"])
    })

    if not chat_obj:
        flash("Chat session not found.")
        return redirect(url_for("chat_page"))

    session["chat_session_id"] = session_id
    return redirect(url_for("chat_page"))


@app.route("/history")
@login_required
def history():
    user_sessions = get_user_sessions(session["user_id"])
    return render_template("history.html", chat_sessions=user_sessions)


@app.route("/admin")
@admin_required
def admin_panel():
    users = list(users_collection.find().sort("created_at", -1))
    total_sessions = chat_sessions_collection.count_documents({})
    total_messages = chat_messages_collection.count_documents({})

    return render_template(
        "admin.html",
        users=users,
        total_sessions=total_sessions,
        total_messages=total_messages
    )


@app.route("/get", methods=["POST"])
@login_required
def chat():
    msg = request.form.get("msg", "").strip()
    age = request.form.get("age", "").strip()
    gender = request.form.get("gender", "").strip()
    symptom = request.form.get("symptom", "").strip()
    duration = request.form.get("duration", "").strip()
    severity_input = request.form.get("severity", "").strip()
    conditions = request.form.get("conditions", "").strip()

    if not msg and not symptom:
        return jsonify({"error": "Please enter your symptom or message."}), 400

    current_session_id = session.get("chat_session_id")
    if not current_session_id:
        current_session_id = create_chat_session(session["user_id"])
        session["chat_session_id"] = current_session_id

    intake_parts = []
    if age:
        intake_parts.append(f"Age: {age}")
    if gender:
        intake_parts.append(f"Gender: {gender}")
    if symptom:
        intake_parts.append(f"Main Symptom: {symptom}")
    if duration:
        intake_parts.append(f"Duration: {duration}")
    if severity_input:
        intake_parts.append(f"Severity: {severity_input}")
    if conditions:
        intake_parts.append(f"Existing Conditions: {conditions}")
    if msg:
        intake_parts.append(f"User Query: {msg}")

    final_input = "\n".join(intake_parts)

    save_message(current_session_id, "user", final_input)

    response = rag_chain.invoke({"input": final_input})
    answer = str(response["answer"])

    current_chat = chat_sessions_collection.find_one({"_id": ObjectId(current_session_id)})
    if current_chat and current_chat.get("title") == "New Chat":
        base_title = symptom if symptom else msg
        title = base_title[:40] + ("..." if len(base_title) > 40 else "")
        chat_sessions_collection.update_one(
            {"_id": ObjectId(current_session_id)},
            {"$set": {"title": title, "updated_at": datetime.utcnow()}}
        )

    save_message(current_session_id, "bot", answer)

    mode = detect_mode(final_input)
    severity_label, severity_color = detect_severity(final_input, severity_input, answer)

    intake_data = {
        "age": age,
        "gender": gender,
        "symptom": symptom,
        "duration": duration,
        "conditions": conditions
    }

    summary = build_summary(intake_data, msg or symptom, answer, mode, severity_label)
    emergency = severity_label == "Emergency"
    care_message = nearby_care_message(severity_label)

    return jsonify({
        "answer": answer,
        "mode": mode,
        "severity": severity_label,
        "severity_color": severity_color,
        "summary": summary,
        "emergency": emergency,
        "care_message": care_message,
        "session_id": current_session_id
    })


@app.route("/download_report/<session_id>")
@login_required
def download_report(session_id):
    chat_obj = chat_sessions_collection.find_one({
        "_id": ObjectId(session_id),
        "user_id": ObjectId(session["user_id"])
    })

    if not chat_obj:
        flash("Report not found.")
        return redirect(url_for("chat_page"))

    messages = get_session_messages(session_id)
    if not messages:
        flash("No consultation found to download.")
        return redirect(url_for("chat_page"))

    latest_user_message = ""
    latest_bot_message = ""
    for m in messages:
        if m["sender"] == "user":
            latest_user_message = m["message"]
        elif m["sender"] == "bot":
            latest_bot_message = m["message"]

    mode = detect_mode(latest_user_message)
    severity_label, _ = detect_severity(latest_user_message, "", latest_bot_message)

    summary = {
        "primary_concern": "Consultation Summary",
        "severity": severity_label,
        "precautions": "Follow medical precautions and consult a doctor if symptoms persist.",
        "doctor_advice": "Consult a licensed doctor for proper diagnosis and treatment."
    }

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 10, "AetherCura AI Consultation Report", ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, safe_pdf_text(f"Patient/User: {session.get('username')}"), ln=True)
    pdf.cell(0, 8, safe_pdf_text(f"Session Title: {chat_obj.get('title', 'Consultation')}"), ln=True)
    pdf.cell(0, 8, safe_pdf_text(f"Generated On: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"), ln=True)
    pdf.cell(0, 8, safe_pdf_text(f"Consultation Mode: {mode}"), ln=True)
    pdf.cell(0, 8, safe_pdf_text(f"Severity Level: {severity_label}"), ln=True)
    pdf.ln(6)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, "AI Guidance", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, safe_pdf_text(latest_bot_message if latest_bot_message else "No AI guidance available."))
    pdf.ln(4)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, "Precautions", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, safe_pdf_text(summary["precautions"]))
    pdf.ln(4)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, "Doctor Advice", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, safe_pdf_text(summary["doctor_advice"]))
    pdf.ln(4)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, "Disclaimer", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(
    0, 7,
    safe_pdf_text(
        "This is an AI-generated consultation report and not a legal medical prescription. "
        "Use it only as informational support. For diagnosis, prescriptions, emergencies, or treatment, consult a licensed medical professional."
    )
)

    safe_title = re.sub(r"[^a-zA-Z0-9_-]+", "_", chat_obj.get("title", "consultation")).strip("_") or "consultation_report"
    file_path = f"{safe_title}.pdf"
    pdf.output(file_path)

    return send_file(file_path, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)