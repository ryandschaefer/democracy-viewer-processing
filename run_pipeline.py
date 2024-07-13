print("Starting processing pipeline...")
from dotenv import load_dotenv
load_dotenv()
from sys import argv
from os import environ
from json import load
import requests

BASE_URL = "{}/datasets".format(environ.get("BACKEND_ENDPOINT"))

DATA_FILE = argv[1]
METADATA_FILE = argv[2]
metadata = load(METADATA_FILE)

print("Uploading data to server...")
headers = {
    "Authorization": "Bearer {}".format(metadata["token"])
}
response = requests.post(BASE_URL, files = { "file": open(DATA_FILE, "rb") })

print("Setting up dataset...")
params = {
    "table_name": response.json()["table_name"],
    "metadata": metadata["metadata"],
    "tags": metadata["tags"],
    "text": metadata["text"]
}
requests.post("{}/upload".format(BASE_URL), params)

# Run preprocessing
print("Running processing...")
from preprocessing import *