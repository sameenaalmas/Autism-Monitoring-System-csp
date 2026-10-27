import os
import pandas as pd

from dotenv import load_dotenv
from pymongo import MongoClient


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI")


# =========================================================
# CONNECT TO MONGODB
# =========================================================

print("Connecting to MongoDB...")

client = MongoClient(
    MONGO_URI,
    serverSelectionTimeoutMS=5000
)

client.admin.command("ping")

db = client["autism_monitor"]

observations_collection = db["observations"]

print("MongoDB connected successfully! ✅")


# =========================================================
# LOAD CSV
# =========================================================

DATA_FILE = "data/observations.csv"

print("Reading CSV...")

df = pd.read_csv(DATA_FILE)

print(
    "Total observations in CSV:",
    len(df)
)


# =========================================================
# SYNCHRONIZE OBSERVATIONS
# =========================================================

added_count = 0
existing_count = 0


for _, row in df.iterrows():

    observation = {

        "date":
            str(row["date"]),

        "student_id":
            str(row["student_id"]),

        "student_name":
            str(row["student_name"]),

        "communication":
            str(row["communication"]),

        "social_interaction":
            str(row["social_interaction"]),

        "learning":
            str(row["learning"]),

        "hyperactivity":
            str(row["hyperactivity"]),

        "emotional_state":
            str(row["emotional_state"]),

        "sound_sensitivity":
            str(row["sound_sensitivity"]),

        "attendance":
            str(row["attendance"]),

        "remarks":
            str(row["remarks"])
    }


    # =============================================
    # CHECK WHETHER THIS OBSERVATION ALREADY EXISTS
    # =============================================

    existing = observations_collection.find_one(
        observation
    )


    if existing:

        existing_count += 1

        print(
            "Already exists:",
            observation["student_id"],
            observation["date"]
        )

    else:

        observations_collection.insert_one(
            observation
        )

        added_count += 1

        print(
            "Added:",
            observation["student_id"],
            observation["date"]
        )


# =========================================================
# FINAL RESULT
# =========================================================

print()
print("==========================================")
print("Synchronization completed! ✅")
print("==========================================")

print(
    "Already in MongoDB:",
    existing_count
)

print(
    "Added to MongoDB:",
    added_count
)

print(
    "Total observations now in MongoDB:",
    observations_collection.count_documents({})
)

print("==========================================")