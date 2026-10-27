
from getpass import getpass
from app import (
    users_collection,
    students_collection,
    MONGO_CONNECTED
)
from werkzeug.security import generate_password_hash

if not MONGO_CONNECTED or users_collection is None:
    print("MongoDB is not connected. Check app.py and try again.")
    raise SystemExit(1)

parent_email = input("Enter parent email: ").strip().lower()
parent_password = getpass("Create parent password: ")
student_id = input("Enter the student's ID (e.g., ST003): ").strip()

if not parent_email or not parent_password or not student_id:
    print("Email, password, and student ID are required.")
    raise SystemExit(1)

student = students_collection.find_one({"student_id": student_id})

if not student:
    print("Student ID not found. Check the ID and try again.")
    raise SystemExit(1)

existing_user = users_collection.find_one({"email": parent_email})

if existing_user:
    print("That email already exists. No account was changed.")
    raise SystemExit(1)

users_collection.insert_one({
    "email": parent_email,
    "password": generate_password_hash(parent_password),
    "role": "parent",
    "linked_student_id": student_id
})

print("Parent account created and linked successfully.")
