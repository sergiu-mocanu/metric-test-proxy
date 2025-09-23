import os
from pathlib import Path

from glob import glob
from random import choice

from metric_test_proxy.metric_measurement.enum import CodeDataset
from metric_test_proxy.classifiers.enum import Classifier

def find_project_root(marker='requirements.txt'):
    path = Path().resolve()
    for parent in [path] + list(path.parents):
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Could not find {marker} in any parent directories")

project_root_parent = find_project_root().parent

exp_data_path = os.path.join(project_root_parent, 'metric_exp_data')

code_path = os.path.join(exp_data_path, 'code')
exp_results_path = os.path.join(exp_data_path, 'exp_results')


def get_functionality_test_path(code_dataset: CodeDataset):
    dataset_name = str(code_dataset.value)
    target_path = os.path.join(exp_results_path, dataset_name, 'functionality_tests')
    return target_path


def get_metric_score_path(code_dataset: CodeDataset):
    dataset_name = str(code_dataset.value)
    target_path = os.path.join(exp_results_path, dataset_name, 'metrics_score')
    return target_path


def get_classification_results_path(code_dataset: CodeDataset, classifier: Classifier, iterations=False,
                                    confusion_matrix=False, folder_date=None):
    dataset_name = str(code_dataset.value)
    classifier_name = str(classifier.value)

    if folder_date is None:
        classifier_res_path = os.path.join(exp_results_path, dataset_name, classifier_name)
    else:
        if classifier == Classifier.LR:
            date_concat = f'{folder_date}-LR'
        else:
            date_concat = f'{folder_date}-DT'

        classifier_res_path = os.path.join(exp_results_path, dataset_name, date_concat)

    if iterations:
        classifier_res_path = os.path.join(classifier_res_path, 'training_iterations')
    elif confusion_matrix:
        classifier_res_path = os.path.join(classifier_res_path, 'confusion_matrix')

    return classifier_res_path


def get_ai_code_path(code_dataset: CodeDataset):
    dataset_name = str(code_dataset.value)
    target_path = os.path.join(code_path, dataset_name)
    return target_path


def get_humaneval_baseline_path():
    humaneval_baseline_path = os.path.join(code_path, 'humaneval_baseline')
    return humaneval_baseline_path


def get_baseline_by_index(task_index: int):
    baseline_path = get_humaneval_baseline_path()
    baseline_scripts = sorted(os.listdir(baseline_path))
    target_script = baseline_scripts[task_index]
    target_baseline_path = os.path.join(baseline_path, target_script)
    return target_baseline_path


def get_python_corpus_path():
    python_corpus_path = os.path.join(code_path, 'python_corpus', 'python_data.txt')
    return python_corpus_path


def get_rand_ai_script_path():
    ai_code_path = get_ai_code_path(CodeDataset.original)
    random_script_path = choice(glob(f'{ai_code_path}/**/**/*.py'))
    return random_script_path
