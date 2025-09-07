import os
from pathlib import Path

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


def get_functionality_test_path(dataset_name):
    target_path = os.path.join(exp_results_path, dataset_name, 'functionality_tests')
    return target_path


def get_metric_score_path(dataset_name):
    target_path = os.path.join(exp_results_path, dataset_name, 'metrics_score')
    return target_path


def get_classification_results_path(dataset_name, classifier_name, iterations=False, confusion_matrix=False):
    classifier_res_path = os.path.join(exp_results_path, dataset_name, classifier_name)

    if iterations:
        classifier_res_path = os.path.join(classifier_res_path, 'training_iterations')
    elif confusion_matrix:
        classifier_res_path = os.path.join(classifier_res_path, 'confusion_matrix')

    return classifier_res_path


def get_ai_code_path(dataset_name):
    target_path = os.path.join(code_path, dataset_name)
    return target_path


def get_humaneval_baseline_path():
    humaneval_baseline_path = os.path.join(code_path, 'humaneval_baseline')
    return humaneval_baseline_path


def get_python_corpus_path():
    python_corpus_path = os.path.join(code_path, 'python_corpus', 'python_data.txt')
    return python_corpus_path