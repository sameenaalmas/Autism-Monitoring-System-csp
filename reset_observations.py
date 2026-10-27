import os
import pandas as pd

from dotenv import load_dotenv
from pymongo import MongoClient


# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI")


# ==========================================
# CONNECT TO MONGODB
# ==========================================

print("Connecting to MongoDB...")

client = MongoClient(
    MONGO_URI,
    serverSelectionTimeoutMS=5000
)

client.admin.command("ping")

db = client["autism_monitor"]

observations_collection = db["observations"]


print("MongoDB connected successfully! ✅")


# ==========================================
# READ CSV
# ==========================================

DATA_FILE = "data/observations.csv"

print("Reading CSV...")

df = pd.read_csv(DATA_FILE)

print(
    "Total observations in CSV:",
    len(df)
)


# ==========================================
# SAFETY CHECK
# ==========================================

if df.empty:

    print()
    print("CSV is empty ❌")
    print("No changes were made to MongoDB.")

    client.close()

    exit()


# ==========================================
# DELETE OLD OBSERVATIONS
# ==========================================

print()
print("Removing old MongoDB observations...")

delete_result = observations_collection.delete_many({})

print(
    "Deleted observations:",
    delete_result.deleted_count
)


# ==========================================
# PREPARE CSV RECORDS
# ==========================================

observations = []

for _, row in df.iterrows():

    observation = {
        "date": str(row["date"]),
        "student_id": str(row["student_id"]),
        "student_name": str(row["student_name"]),
        "communication": str(row["communication"]),
        "social_interaction": str(row["social_interaction"]),
        "learning": str(row["learning"]),
        "hyperactivity": str(row["hyperactivity"]),
        "emotional_state": str(row["emotional_state"]),
        "sound_sensitivity": str(row["sound_sensitivity"]),
        "attendance": str(row["attendance"]),
        "remarks": str(row["remarks"])
    }

    observations.append(observation)


# ==========================================
# INSERT CSV RECORDS INTO MONGODB
# ==========================================

print()
print("Adding CSV observations to MongoDB...")

insert_result = observations_collection.insert_many(
    observations
)

print(
    "Inserted observations:",
    len(insert_result.inserted_ids)
)


# ==========================================
# FINAL VERIFICATION
# ==========================================

final_count = observations_collection.count_documents({})

print()
print("==========================================")
print("Synchronization completed! ✅")
print("==========================================")
print("CSV observations:", len(df))
print("MongoDB observations:", final_count)
print("==========================================")


# ==========================================
# CHECK RESULT
# ==========================================

if final_count == len(df):

    print()
    print("SUCCESS ✅")
    print(
        "MongoDB now contains exactly the same"
        " number of observations as the CSV."
    )

else:

    print()
    print("WARNING ⚠️")
    print(
        "MongoDB count does not match CSV count."
    )


# ==========================================
# CLOSE CONNECTION
# ==========================================

client.close()

print()
print("MongoDB connection closed.")