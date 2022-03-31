import os
import json
from datetime import datetime

import yaml
from pyspark.sql import SparkSession, DataFrame

import constants as c


def _get_spark_session(spark_driver_memory: str) -> SparkSession:
    return SparkSession.builder\
        .config('spark.driver.memory', spark_driver_memory)\
        .getOrCreate()


def _get_emb_dim(df: DataFrame) -> int:
    return len(df.select(c.EMB_COL).take(1)[0][0])


def _load_yaml(path: str) -> dict:
    with open(path, 'r') as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return yaml_dict


def _load_emb_df(ss: SparkSession, path: str, dim: int) -> DataFrame:
    df = ss.read.parquet(path)

    for col in [c.EMB_COL, c.ID_COL]:
        assert col in df.columns, \
            f'Embedding file does not have "{col}" column'

    actual_dim = _get_emb_dim(df)
    assert actual_dim == dim, \
        f'Embedding file dim={actual_dim}, but setup file specifies dim={dim}'

    return df


def _load_train_df(ss: SparkSession, path: str) -> DataFrame:
    df = ss.read.option('header', True).csv(path)

    for col in [c.LABEL_COL, c.ID_COL]:
        assert col in df.columns, \
            f'{path}: Train file does not have "{col}" column'

    return df


def _add_emb_col(df: DataFrame, emb_df: DataFrame) -> DataFrame:
    emb_df = emb_df.select(c.ID_COL, c.EMB_COL)
    return df.join(emb_df, c.ID_COL)


def _load_test_df(ss: SparkSession, path: str, dim: int) -> DataFrame:
    df = ss.read.parquet(path)

    for col in [c.LABEL_COL, c.ID_COL]:
        assert col in df.columns, \
            f'{path}: Train file does not have "{col}" column'

    actual_dim = _get_emb_dim(df)
    assert actual_dim == dim, \
        f'Test file dim={actual_dim}, but setup file specifies dim={dim}'

    return df


def _save_results(data: dict, save_dir: str, verbose=False) -> None:
    dt = datetime.utcnow().strftime("UTC-%Y-%m-%d-%H-%M-%S")
    filename = f'{c.RESULT_FILE_PREFIX}_{dt}.json'
    path = os.path.join(save_dir, filename)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

    if verbose:
        print(f'Results saved in {path}')
        with open(path, 'r') as f:
            print(json.dumps(json.load(f), indent=4))
