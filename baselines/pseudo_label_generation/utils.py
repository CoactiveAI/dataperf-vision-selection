import json
import numpy as np
import pandas as pd
import sklearn


def make_json(train_file, file_name, data_class_name):
    """"""
    df = pd.read_csv(train_file)
    dct_ = {data_class_name: {}}
    dct = dct_[data_class_name]

    for index, row in df.iterrows():
        dct.update({row["ImageID"]: row["Confidence"]})

    json_string = json.dumps(dct_)
    json_file = open(file_name, "w")
    json_file.write(json_string)
    json_file.close()


class SubmissionCreator:

    def __init__(self, cupcake_ids, hawk_ids, sushi_ids):
        """"""
        self.cupcake = cupcake_ids
        self.hawk = hawk_ids
        self.sushi = sushi_ids

        self.names = ["Cupcake", "Hawk", "Sushi"]

    def write(self, file_name, target_class, file_name_json=""):
        """"""
        data = np.hstack((self.cupcake, self.hawk, self.sushi))
        labels = np.zeros(data.shape[0])

        if target_class == 0:
            labels[:self.cupcake.shape[0]] = 1

        if target_class == 1:
            labels[self.cupcake.shape[0]:self.cupcake.shape[0] + self.hawk.shape[0]] = 1

        if target_class == 2:
            labels[-self.sushi.shape[0]:] = 1

        df = pd.DataFrame(data={"ImageID": data, "Confidence": labels.astype(int)})
        df.to_csv(file_name, index=False)

        if file_name_json != "":
            make_json(file_name, file_name_json, self.names[target_class])


class EnsembleOOD:

    def __init__(self, model, X, y, n_models=10):  # noqa
        """"""
        self.model = model
        self.num_models = n_models

        self.models = [sklearn.base.clone(self.model).fit(X, y) for _ in range(self.num_models)]

    def predict_value(self, X, _y):  # noqa
        predictions = np.array([model.predict(X) for model in self.models])
        std = np.std(predictions, axis=0)
        return std

    def predict(self, X):  # noqa
        """Compute predictions with majority voting for the ensemble."""
        predictions = np.array([model.predict(X) for model in self.models])
        mean = np.mean(predictions, axis=0)
        return np.round(mean)
