from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")

client = MongoClient(MONGO_URI)
db = client["medical_chatbot_db"]

users_collection = db["users"]
chat_sessions_collection = db["chat_sessions"]
chat_messages_collection = db["chat_messages"]

users_collection.create_index("email", unique=True)
chat_sessions_collection.create_index("user_id")
chat_messages_collection.create_index("session_id")
chat_messages_collection.create_index([("session_id", 1), ("created_at", 1)])