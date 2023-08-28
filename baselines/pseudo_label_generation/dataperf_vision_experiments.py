import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier

import sys
sys.path.append("dataperf-vision-selection/")
from main import run_tasks
from utils import SubmissionCreator, EnsembleOOD


# Load the data
data = pd.read_parquet('./dataperf-vision-selection/data/embeddings/train_emb_256_dataperf.parquet', engine='pyarrow')
data_np = np.vstack(data['embedding'].values)
data_ids = data["ImageID"]

base_folder = './dataperf-vision-selection/data/'

data_cupcake = pd.read_csv(base_folder + 'examples/alpha_example_set_Cupcake.csv')
cupcake_ids = data_cupcake['ImageID'].values
cupcake_idx = np.where([id_ in cupcake_ids for id_ in data_ids])
cupcake_np = data_np[cupcake_idx]
print(cupcake_np.shape)

data_hawk = pd.read_csv(base_folder + 'examples/alpha_example_set_Hawk.csv')
hawk_ids = data_hawk['ImageID'].values
hawk_idx = np.where([id_ in hawk_ids for id_ in data_ids])
hawk_np = data_np[hawk_idx]
print(hawk_np.shape)

data_sushi = pd.read_csv(base_folder + 'examples/alpha_example_set_Sushi.csv')
sushi_ids = data_sushi['ImageID'].values
sushi_idx = np.where([id_ in sushi_ids for id_ in data_ids])
sushi_np = data_np[sushi_idx]
print(sushi_np.shape)

# Experiment 1: baseline, the provided examples
example_cupcake_ids = data_cupcake["ImageID"].values
example_sushi_ids = data_sushi["ImageID"].values
example_hawk_ids = data_hawk["ImageID"].values

sc = SubmissionCreator(example_cupcake_ids, example_hawk_ids, example_sushi_ids)

sc.write(base_folder + 'train_sets/cupcake.csv', 0,
         base_folder + 'train_sets/cupcake_baseline.json')

sc.write(base_folder + 'train_sets/hawk.csv', 1,
         base_folder + 'train_sets/hawk_baseline.jso')

sc.write(base_folder + 'train_sets/sushi.csv', 2,
         base_folder + 'train_sets/sushi_baseline.json')

run_tasks("task_setup_new.yaml")

# Experiment 2: Random selection of 333 images per class with pseudo-labels from sklearn classifier

data_train = np.vstack([cupcake_np, hawk_np, sushi_np])
labels_train = np.hstack([np.ones(20) * 0, np.ones(20) * 1, np.ones(20) * 2])

mlp = MLPClassifier(25)
mlp.fit(data_train, labels_train)
mlp.score(data_train, labels_train)

pseudo_labels = mlp.predict(data_np)

sushi_ids = np.random.choice(data_ids[pseudo_labels == 2], 333, replace=False)
hawk_ids = np.random.choice(data_ids[pseudo_labels == 1], 333, replace=False)
cupcake_ids = np.random.choice(data_ids[pseudo_labels == 0], 333, replace=False)

sc = SubmissionCreator(cupcake_ids, hawk_ids, sushi_ids)

sc.write(base_folder + "train_sets/cupcake.csv", 0, base_folder + "train_sets/cupcake_nn.json")
sc.write(base_folder + "train_sets/hawk.csv", 1, base_folder + "train_sets/hawk_nn.json")
sc.write(base_folder + "train_sets/sushi.csv", 2, base_folder + "train_sets/sushi_nn.json")

# Experiment 3: Select samples based on uncertainty
ood = EnsembleOOD(mlp, data_train, labels_train, n_models=50)
pseudo_labels = ood.predict(data_np)
uncertainty = ood.predict_value(data_np, None)
prob = uncertainty / uncertainty.sum()

sushi_ids = np.random.choice(data_ids[pseudo_labels == 2], 333, replace=False,
                             p=prob[pseudo_labels == 2] / np.sum(prob[pseudo_labels == 2]))
hawk_ids = np.random.choice(data_ids[pseudo_labels == 1], 333, replace=False,
                            p=prob[pseudo_labels == 1] / np.sum(prob[pseudo_labels == 1]))
cupcake_ids = np.random.choice(data_ids[pseudo_labels == 0], 333, replace=False,
                               p=prob[pseudo_labels == 0] / np.sum(prob[pseudo_labels == 0]))

sc = SubmissionCreator(cupcake_ids, hawk_ids, sushi_ids)
sc.write(base_folder + "train_sets/cupcake.csv", 0, base_folder + "train_sets/cupcake_ood.json")
sc.write(base_folder + "train_sets/hawk.csv", 1, base_folder + "train_sets/hawk_ood.json")
sc.write(base_folder + "train_sets/sushi.csv", 2, base_folder + "train_sets/sushi_ood.json")
