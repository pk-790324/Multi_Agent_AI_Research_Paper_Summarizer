# database.py

from pymongo import MongoClient
import os

MONGO_URL = os.getenv(
    "MONGO_URL",
    "mongodb://localhost:27017"
)

client = MongoClient(MONGO_URL)

database = client["my_app"]

user_collection = database["users"]

# Ensure email addresses are unique
user_collection.create_index("email", unique=True)