import os
import pandas as pd
from pymongo import MongoClient
import time

MONGO_URI = "mongodb://mongodb:27017/"
DB_NAME = "customer_dashboard"         
COLLECTION_NAME = "customer_feedback"  
DATASETS_DIR = "/updater/datasets"     

def connect_to_mongodb():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    return db[COLLECTION_NAME] 
def process_csv_file(file_path, collection): #read csv file and update mongodb
    try:
        print(f"Processing file: {file_path}")
        data = pd.read_csv(file_path)
        records = data.to_dict(orient="records")
        # insert or update records
        for record in records:
            unique_key = {"_id": hash(frozenset(record.items()))}
            collection.update_one(unique_key, {"$set": record}, upsert=True)

        print(f"Updated collection with data from: {file_path}")
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

def process_all_csvs():#Processes all CSV files
    collection = connect_to_mongodb()
    if not os.path.exists(DATASETS_DIR):
        print(f"Datasets directory '{DATASETS_DIR}' does not exist.")
        return

    for filename in os.listdir(DATASETS_DIR):
        if filename.endswith(".csv"):
            file_path = os.path.join(DATASETS_DIR, filename)
            process_csv_file(file_path, collection)

if __name__ == "__main__":
    print("Data Updater Service is running...")#monitors new files
    while True:
        process_all_csvs()
        time.sleep(10)
