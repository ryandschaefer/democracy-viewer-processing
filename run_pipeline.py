print("Starting processing pipeline...")
from dotenv import load_dotenv
load_dotenv()
from sys import argv
from os import environ, listdir, remove, path
from json import load
import requests
from subprocess import run

BASE_URL = "{}/datasets".format(environ.get("BACKEND_ENDPOINT"))

DATA_FILE = argv[1]
METADATA_FILE = argv[2]
with open(METADATA_FILE) as file:
    metadata = load(file)

print("Uploading data to server...")
headers = {
    "Authorization": "Bearer {}".format(metadata["token"])
}
response = requests.post(BASE_URL, files = { "file": open(DATA_FILE, "rb") }, headers = headers)

print("Configuring dataset...")
params = {
    "table_name": response.json()["table_name"],
    "metadata": metadata["metadata"],
    "tags": metadata["tags"],
    "text": metadata["text"]
}
requests.post("{}/upload".format(BASE_URL), json = params, headers = headers)

print("Running processing...")
run(["python3", "preprocessing.py", params["table_name"]])

print("Deleting local files...")
def delete_dir(dir_name = "./files"):
    dir_files = listdir(dir_name)
    for item in dir_files:
        file_path = path.join(dir_name, item)
        if path.isdir(file_path):
            delete_dir(file_path)
        elif file_path.endswith(".parquet") or file_path.endswith(".pkl"):
            print("Deleting:", file_path)
            remove(path.join(dir_name, item))
            
delete_dir()