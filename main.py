import fire

import constants as c
import utils as utils
import eval as eval


def main(setup_yaml_path=c.DEFAULT_SETUP_YAML_PATH):
    task_setup = utils.load_yaml(setup_yaml_path)
    dim = task_setup[c.SETUP_YAML_DIM_KEY]
    emb_path = task_setup[c.SETUP_YAML_EMB_KEY]

    ss = utils.get_spark_session(task_setup[c.SETUP_YAML_SPARK_MEM_KEY])
    emb_df = utils.load_emb_df(ss=ss, path=emb_path, dim=dim)

    task_paths = {
        task: task_setup[task] for task in task_setup[c.SETUP_YAML_TASKS_KEY]}
    task_scores = {}
    for task, paths in task_paths.items():
        train_path, test_path = paths
        train_df = utils.load_train_df(ss=ss, path=train_path)
        train_df = utils.add_emb_col(df=train_df, emb_df=emb_df)
        test_df = utils.load_test_df(ss=ss, path=test_path, dim=dim)

        clf = eval.get_trained_classifier(df=train_df)
        task_scores[task] = eval.score_classifier(df=test_df, clf=clf)

    save_dir = task_setup[c.SETUP_YAML_RESULTS_KEY]
    utils.save_results(data=task_scores, save_dir=save_dir, verbose=True)


if __name__ == "__main__":
    fire.Fire(main)
