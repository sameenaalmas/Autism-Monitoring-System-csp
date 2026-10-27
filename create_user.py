import os
from dotenv import load_dotenv
from pymongo import MongoClient
from werkzeug.security import generate_password_hash

# Load .env
load_dotenv()

# MongoDB connection
MONGO_URI = os.getenv("MONGODB_URI")

client = MongoClient(
    MONGO_URI,
    serverSelectionTimeoutMS=5000
)

db = client["autism_monitor"]
users_collection = db["users"]

# Test user
email = "teacher@test.com"
password = "Teacher@123"
role = "teacher"

# Check whether user already exists
existing_user = users_collection.find_one({
    "email": email
})

if existing_user:
    print("User already exists.")
else:
    user = {
        "email": email,
        "password": generate_password_hash(password),
        "role": role
    }

    users_collection.insert_one(user)

    print("Test user created successfully! ✅")
    print("Email:", email)
    print("Role:", role)