import fire
import numpy as np
from selection.selection import TrainingSetSelection
from selection.serialization import deserialize


def main(train_file):
    train_data = deserialize.deserialize_from_pb(train_file)
    target_vectors = train_data["target_mswc_vectors"]
    target_ids = train_data["target_ids"]
    nontarget_vectors = train_data["nontarget_mswc_vectors"]
    nontarget_ids = train_data["nontarget_ids"]

    selection = TrainingSetSelection(
        target_vectors=target_vectors,
        target_ids=target_ids,
        nontarget_vectors=nontarget_vectors,
        nontarget_ids=nontarget_ids,
    )

    train_x, train_y = selection.select()

    np.save("train_x.npy", train_x)
    np.save("train_y.npy", train_y)


if __name__ == "__main__":
    fire.Fire(main)
