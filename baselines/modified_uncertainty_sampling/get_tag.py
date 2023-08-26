import os
import pandas as pd
import requests

if False:
    target_tag = "/m/03p1r4"
    output_file = "cupcake.txt"
if False:
    target_tag = "/m/0fp7c"
    output_file = "hawk.txt"
if True:
    target_tag = "/m/07030"
    output_file = "sushi_human.txt"


# ------------------------------------------
# FUNCTIONS
# -----------------------------------------
def download_csv_from_url(url, local_filename):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(local_filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"File downloaded and saved as: {local_filename}")
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
    except requests.exceptions.RequestException as e:
        print(f"Request Exception: {e}")


# ------------------------------------------
# LOAD MACHINE-HUMAN LABELS
# -----------------------------------------

output_path = "output_repo/"  # Change
machine_human_labels_path = os.path.join(
    output_path, "oidv6-train-annotations-human-imagelabels.csv"
)  # DO NOT CHANGE
try:
    mhl = pd.read_csv(machine_human_labels_path)
except FileNotFoundError:
    print("loading human-verified labels...")
    url = "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-human-imagelabels.csv"
    download_csv_from_url(url, machine_human_labels_path)
    mhl = pd.read_csv(machine_human_labels_path)


human_labels = []

for line in open("annotations-human.csv", "r").readlines():
    image_id, _, tag, value = line.strip().split(",")

    if tag == target_tag and float(value) >= 0.5:
        human_labels.append(image_id)


machine_labels = []
for line in open("annotations-machine.csv", "r").readlines():
    image_id, _, tag, value = line.strip().split(",")

    if tag == target_tag and float(value) >= 0.5:
        machine_labels.append(image_id)


print(len(human_labels))
print(len(machine_labels))

print(len(set(human_labels).intersection(set(machine_labels))))

if True:
    f = open(output_file, "w")
    for image_id in set(human_labels):
        f.write(image_id + "\n")
    f.close()

if False:
    f = open(output_file, "w")
    for image_id in set(human_labels + machine_labels):
        f.write(image_id + "\n")
    f.close()
