# Author: Paolo Climaco
# Author email: climaco@ins.uni-bonn.de

import numpy as np
from operator import itemgetter
import torch
from dgl.geometry import farthest_point_sampler
from tqdm import tqdm
import pandas as pd
import numpy as np
import shutil
from typing import Dict, List
import sklearn
import sklearn.model_selection
import sklearn.ensemble
import sklearn.linear_model
import tqdm
import os
import csv
import json
from sklearn.metrics import f1_score
import warnings
import requests

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

# ------------------------------------------
# PATHS
# -----------------------------------------

embeddings_path = "../../data/embeddings/train_emb_256_dataperf.parquet"
examples_path = "../../data/examples/"
output_path = "output_repo/"  # Change
machine_human_labels_path = os.path.join(
    output_path, "oidv6-train-annotations-human-imagelabels.csv"
)  # DO NOT CHANGE

# Create output directory if does not exist
if not os.path.exists(output_path):
    os.makedirs(output_path)


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


def read_parquet(path=str):
    return pd.read_parquet(path, engine="pyarrow", use_threads=True)


def get_labelled_samples(task=None, mhl=None, embeddings=None, examples_path=None):
    # load examples
    examples_df = pd.read_csv(
        os.path.join(examples_path, f"alpha_example_set_{task}.csv")
    )
    # get label name examples
    LabelName = examples_df["LabelName"][0]

    mydf = mhl[mhl["ImageID"].isin(embeddings["ImageID"])]
    labelled_samples_df = mydf[mydf["LabelName"] == LabelName]
    labelled_samples_pd = pd.DataFrame(
        embeddings[embeddings["ImageID"].isin(labelled_samples_df["ImageID"])]
    )
    labelled_samples_pd.loc[:, "Confidence"] = np.asarray(
        [
            labelled_samples_df[labelled_samples_df["ImageID"] == image][
                "Confidence"
            ].item()
            for image in labelled_samples_pd["ImageID"]
        ]
    )
    labelled_samples_vect = np.stack(labelled_samples_pd["embedding"].to_numpy())
    # return only those with confidence 1
    return (
        labelled_samples_vect[labelled_samples_pd["Confidence"] == 1],
        labelled_samples_pd[labelled_samples_pd["Confidence"] == 1],
    )


def csv_to_json(csv, json_file_path):
    data = []

    data = csv
    dictio = {}
    for i, c in zip(data["ImageID"], data["Confidence"]):
        dictio[i] = int(c)

    # Write JSON data to a file
    with open(json_file_path, "w") as json_file:
        json_file.write(json.dumps(dictio, indent=4))


# ------------------------------------------
# DOWNLOAD EMBEDDINGS
# -----------------------------------------
print("Loading embeddings")
embeddings = read_parquet(embeddings_path)
embeddings_vect = np.stack(embeddings["embedding"].to_numpy())
print("Embeddings loaded")


# ------------------------------------------
# LOAD MACHINE-HUMAN LABELS
# -----------------------------------------

try:
    mhl = pd.read_csv(machine_human_labels_path)
except FileNotFoundError:
    print("loading human-verified labels...")
    url = "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-human-imagelabels.csv"
    download_csv_from_url(url, machine_human_labels_path)
    mhl = pd.read_csv(machine_human_labels_path)


# ------------------------------------------
# FARTHEST POINT SAMPLING: it takes around 17 minutes but it is a one time effort. SAVE AND LOAD
# -----------------------------------------

try:
    idx_fps = np.load(
        os.path.join(output_path, "FPS_CrossV_init_0.npy")
    )  # np.load('/local/hdd/climaco/dataperf-vision-selection/data/selections_folder/FPS_CrossV_init_0.npy')
except FileNotFoundError:
    print("FPS running...it takes a few minutes but it is a one-time effort")
    t_emb = torch.from_numpy(embeddings_vect).unsqueeze_(0)
    idx_fps = farthest_point_sampler(t_emb, 1000, start_idx=0).numpy()[0].tolist()
    np.save(os.path.join(output_path, "FPS_CrossV_init_0.npy"), idx_fps)
    idx_fps = np.asarray(idx_fps)


# ------------------------------------------
# SELECTION ALGORITHM
# ------------------------------------------


def Selection(
    task="None",
    n_confidence1=5,
    n_confidence0=40,
    n_folds=5,
    mhl=mhl,
    examples_path=examples_path,
    embeddings=embeddings,
    output_path=output_path,
    idx_fps=idx_fps,
):
    """
    Parameters
    ----------

    task : string
        The name of the target class, e.g. Hawk, Cupcake or Sushi.

    n_confidence1 : int
        Number of points in the final selected set with confidence 1

    n_confidence0 : int
        Number of points in the final selected set with confidence 0

    n_folds : int
        Number of folds in the nested cross validation

    mhl : pandas data frame
        Dataframe containing machine-generated labels and human annotations of images in oidv6 training set

    examples_path : string
        Path the the classes examples provided within the challenge

    embeddings :  pandas data frame
        Data frame containing the available pool of data points provided within the challenge

    output_path : string
        Path to folder where algorithm's output is stored

    idx_fps : ndarray

    Returns
    ----------
    csv and .json files containing selected images IDs and associated confidence (0 or 1)


    """

    # get elements confidence 1
    confidence1_samples_vect, confidence1_samples_df = get_labelled_samples(
        task, mhl, embeddings, examples_path
    )

    # Idxs selected elements with FPS sapling algorithm

    # Candidate with confidence 0 and candidates with confiddence 1
    confidence1_samples = confidence1_samples_vect
    confidence0_samples = embeddings_vect[idx_fps]
    confidence1_labels = np.ones(confidence1_samples.shape[0])
    confidence0_labels = np.zeros(confidence0_samples.shape[0])

    # Cross validation
    best_score = 0
    best_confidence1_train_ixs = None
    best_confidence0_train_ixs = None
    confidence0_class_size = n_confidence0
    confidence1_class_size = n_confidence1

    crossfold_confidence1 = sklearn.model_selection.StratifiedShuffleSplit(
        n_splits=n_folds,
        train_size=confidence1_class_size,
        random_state=23,
    )

    for confidence1_train_ixs, confidence1_val_ixs in tqdm.tqdm(
        crossfold_confidence1.split(confidence1_samples, confidence1_labels),
        desc="k-fold cross validation",
        total=n_folds,
    ):
        crossfold_confidence0 = sklearn.model_selection.StratifiedShuffleSplit(
            n_splits=n_folds,
            train_size=confidence0_class_size,
            random_state=23,
        )
        for confidence0_train_ixs, confidence0_val_ixs in crossfold_confidence0.split(
            confidence0_samples, confidence0_labels
        ):
            train_Xs = np.vstack(
                [
                    confidence1_samples[confidence1_train_ixs],
                    confidence0_samples[confidence0_train_ixs],
                ]
            )
            train_ys = np.concatenate(
                [
                    confidence1_labels[confidence1_train_ixs],
                    confidence0_labels[confidence0_train_ixs],
                ]
            )

            clf = sklearn.ensemble.VotingClassifier(
                estimators=[
                    ("lr", sklearn.linear_model.LogisticRegression()),
                ],
                voting="soft",
                weights=None,
                n_jobs=-1,
            )
            clf.fit(train_Xs, train_ys)

            val_Xs = np.vstack(
                [
                    confidence1_samples[confidence1_val_ixs],
                    confidence0_samples[confidence0_val_ixs],
                ]
            )
            val_ys = np.concatenate(
                [
                    confidence1_labels[confidence1_val_ixs],
                    confidence0_labels[confidence0_val_ixs],
                ]
            )

            pred_Ys = clf.predict(val_Xs)

            score = f1_score(val_ys, pred_Ys, labels=[0, 1], average="binary")
            if score > best_score:
                best_score = score
                best_confidence1_train_ixs = confidence1_train_ixs
                best_confidence0_train_ixs = confidence0_train_ixs

    # store selected elements in csv
    submission_confidence1 = pd.DataFrame(
        confidence1_samples_df.iloc[best_confidence1_train_ixs]["ImageID"]
    )
    submission_confidence1.loc[:, "Confidence"] = np.ones(
        len(submission_confidence1), dtype=int
    ).tolist()

    submission_confidence0 = pd.DataFrame(
        embeddings.iloc[idx_fps[best_confidence0_train_ixs]]["ImageID"]
    )
    submission_confidence0.loc[:, "Confidence"] = np.zeros(
        len(submission_confidence0), dtype=int
    ).tolist()

    submission = pd.concat([submission_confidence1, submission_confidence0])

    # seve selected elements in csv and json format
    submission.to_csv(os.path.join(output_path, f"{task}_fpscv_ft2.csv"), index=False)
    csv_to_json(submission, os.path.join(output_path, f"{task}_fpscv_ft2.json"))


# ------------------------------------------
# RUN SELECTION ALGORITHM
# ------------------------------------------

# fpscv average F1 score  78.10%
"""
print('Hawk')
Selection('Hawk')

print('Cupcake')
Selection('Cupcake')

print('Sushi')
Selection('Sushi')
"""
# fpscv_ft average F1 score  79.95%
"""
print('Hawk')
Selection('Hawk', n_confidence1=5, n_confidence0=20)

print('Cupcake')
Selection('Cupcake', n_confidence1=5, n_confidence0=40)

print('Sushi')
Selection('Sushi', n_confidence1=10, n_confidence0=20)
"""

# fpscv_ft2 average F1 score  81%
print("Hawk")
Selection("Hawk", n_confidence1=5, n_confidence0=90, n_folds=4)

print("Cupcake")
Selection("Cupcake", n_confidence1=5, n_confidence0=40)

print("Sushi")
Selection("Sushi", n_confidence1=10, n_confidence0=20)


# ----------------------------------------------
#  how to improve fpscv algorithm
# ----------------------------------------------

"""
Notice that the fpscv, fpscv_ft and fpscv_ft2 are implemented using the same algorithm that has been initialized
considering different values for the parameters n_confidence1, n_confidence0, and n_folds. Different choices of these 
parameters lead to different results.


The main issue of the developed approach is its sensibility to the choices of the parameters
n_confidence1, n_confidence0  and n_folds. Such choices have been made heuristically. Can we do better than heuristic? 
Developing a principled approach for the optimization of the mentioned parameters may lead to a substantial improvement in terms  
of the F1 score on each of the considered tasks.

Please if you do develop such a principled approach do not hesitate to contact me. I will be curious to hear about it.

"""
