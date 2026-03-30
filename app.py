from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, Response
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

index_name = "medical-chatbot"

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


def build_insight(user_message: str, answer: str, mode: str) -> dict:
    msg = user_message.strip()
    short_topic = msg[:60] + ("..." if len(msg) > 60 else "")
    focus_area = mode
    next_step = "Share duration, severity, age, and any related symptoms for better guidance."
    safety_note = "This is general AI-assisted guidance, not a replacement for a licensed doctor."

    if mode == "Medicine Info":
        next_step = "Ask about usage, precautions, side effects, or interactions."
    elif mode == "Lifestyle Advice":
        next_step = "Ask for a daily plan, diet tips, hydration, sleep, or exercise suggestions."
    elif mode == "Emergency Precautions":
        next_step = "Seek urgent medical help or contact local emergency services if symptoms are severe."
        safety_note = "Possible urgent concern detected. Please do not rely only on AI in emergencies."
    elif mode == "Symptoms Check":
        next_step = "Ask about causes, warning signs, home care, and when to see a doctor."

    return {
        "topic": short_topic,
        "focus_area": focus_area,
        "next_step": next_step,
        "safety_note": safety_note
    }


@app.route("/")
def home():
    if "user_id" in session:
        return redirect(url_for("chat_page"))
    return redirect(url_for("login"))


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

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.")
    return redirect(url_for("login"))


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


@app.route("/get", methods=["POST"])
@login_required
def chat():
    msg = request.form.get("msg", "").strip()
    if not msg:
        return jsonify({"error": "Please enter a message."}), 400

    current_session_id = session.get("chat_session_id")
    if not current_session_id:
        current_session_id = create_chat_session(session["user_id"])
        session["chat_session_id"] = current_session_id

    save_message(current_session_id, "user", msg)

    response = rag_chain.invoke({"input": msg})
    answer = str(response["answer"])

    current_chat = chat_sessions_collection.find_one({"_id": ObjectId(current_session_id)})
    if current_chat and current_chat.get("title") == "New Chat":
        title = msg[:40] + ("..." if len(msg) > 40 else "")
        chat_sessions_collection.update_one(
            {"_id": ObjectId(current_session_id)},
            {"$set": {"title": title, "updated_at": datetime.utcnow()}}
        )

    save_message(current_session_id, "bot", answer)

    mode = detect_mode(msg)
    emergency = detect_emergency(msg) or detect_emergency(answer)
    insight = build_insight(msg, answer, mode)

    return jsonify({
        "answer": answer,
        "mode": mode,
        "emergency": emergency,
        "insight": insight,
        "session_id": current_session_id
    })


@app.route("/api/history/<session_id>")
@login_required
def api_history(session_id):
    chat_obj = chat_sessions_collection.find_one({
        "_id": ObjectId(session_id),
        "user_id": ObjectId(session["user_id"])
    })

    if not chat_obj:
        return jsonify({"error": "Not found"}), 404

    messages = get_session_messages(session_id)
    formatted_messages = [
        {
            "sender": m["sender"],
            "message": m["message"],
            "created_at": m["created_at"].isoformat() if m.get("created_at") else ""
        }
        for m in messages
    ]
    return jsonify(formatted_messages)


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

    mode = detect_mode(latest_user_message or "")
    insight = build_insight(latest_user_message or "", latest_bot_message or "", mode)
    emergency = detect_emergency(latest_user_message + " " + latest_bot_message)

    lines = []
    lines.append("AETHERCURA AI CONSULTATION REPORT")
    lines.append("=" * 42)
    lines.append(f"Patient/User: {session.get('username')}")
    lines.append(f"Session Title: {chat_obj.get('title', 'Consultation')}")
    lines.append(f"Generated On: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"Consultation Mode: {mode}")
    lines.append("")

    if emergency:
        lines.append("URGENT SAFETY FLAG:")
        lines.append("Possible urgent concern detected. Seek immediate medical attention if symptoms are severe.")
        lines.append("")

    lines.append("SESSION SUMMARY:")
    lines.append(f"Focus Area: {insight['focus_area']}")
    lines.append(f"Suggested Next Step: {insight['next_step']}")
    lines.append(f"Safety Note: {insight['safety_note']}")
    lines.append("")

    lines.append("LATEST AI GUIDANCE:")
    lines.append(latest_bot_message if latest_bot_message else "No AI guidance available.")
    lines.append("")
    lines.append("FULL CHAT TRANSCRIPT:")
    lines.append("-" * 42)

    for m in messages:
        sender = "USER" if m["sender"] == "user" else "AETHERCURA AI"
        lines.append(f"{sender}:")
        lines.append(m["message"])
        lines.append("")

    lines.append("DISCLAIMER:")
    lines.append("This file is an AI-generated consultation report and not a legal medical prescription.")
    lines.append("Use it as general informational support only. Consult a licensed doctor for diagnosis, prescriptions, or emergencies.")

    report_text = "\n".join(lines)
    safe_title = re.sub(r"[^a-zA-Z0-9_-]+", "_", chat_obj.get("title", "consultation")).strip("_") or "consultation_report"

    return Response(
        report_text,
        mimetype="text/plain",
        headers={"Content-Disposition": f"attachment;filename={safe_title}.txt"}
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)