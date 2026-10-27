import os
from dotenv import load_dotenv
from pymongo import MongoClient

# Load variables from .env
load_dotenv()

# Get MongoDB connection string
MONGO_URI = os.getenv("MONGODB_URI")

print("Checking MongoDB connection...")

if not MONGO_URI:
    print("ERROR: MONGO_URI was not found in .env")
    exit()

try:
    client = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=5000
    )

    # Test connection
    client.admin.command("ping")

    print("MongoDB connected successfully! ✅")

    # Select database
    db = client["autism_monitor"]

    print("Database selected: autism_monitor")

    # Show collections
    print("Current collections:", db.list_collection_names())

except Exception as e:
    print("MongoDB connection failed ❌")
    print("Error:", e)