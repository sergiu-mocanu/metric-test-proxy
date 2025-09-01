import json
import os
import pandas as pd
from pathing import get_path as gp
import time


def custom_sort_key(s):
    # A sorting key used to sort strings in a length-lexicographic order (length and alphabetical order)
    return len(s), s


# Section used to remove Task_145 from the LLM-generated scripts

def remove_task_ai_code():
    # Removing Task_145 from all the LLM-generated code
    ai_code_path = '../../ai_code'

    for path, folders, files in os.walk(ai_code_path):
        for folder_name in folders:
            if 'HumanEval_145' in folder_name:
                folder_path = os.path.join(path, folder_name)
                for script in os.listdir(folder_path):
                    script_path = os.path.join(folder_path, script)
                    os.remove(script_path)
                os.removedirs(folder_path)


def remove_task_funct_test():
    # Removing Task_145 from the .json files holding the functionality tests of LLM-generated code
    funct_test_path = '../exp_results/ai_code/functionality_tests'
    for path, folders, files in os.walk(funct_test_path):
        for file_name in files:
            if 'HumanEval_145' in file_name:
                file_path = os.path.join(path, file_name)
                os.remove(file_path)


def remove_task_metrics():
    # Removing Task_145 from all the code-quality metric measurements for LLM-generated code
    metrics_path = '../exp_results?ai_code/metrics_calc'

    for path, folders, files in os.walk(metrics_path):
        for file_name in files:
            if 'HumanEval_145' in file_name:
                file_path = os.path.join(path, file_name)
                os.remove(file_path)


# Section used to remove LLM-generated duplicate scripts as well as clean the exp results

def remove_duplicate_scripts(folder_path):
    distinct_scripts = set()

    num_total_scripts = 0
    num_duplicate_scripts = 0
    duplicate_scripts = {}

    list_models_and_temps = sorted(os.listdir(folder_path))
    for model_and_temp in list_models_and_temps:
        print(f'Analyzing model: {model_and_temp}')
        current_model_path = os.path.join(folder_path, model_and_temp)
        list_tasks = sorted(os.listdir(current_model_path), key=custom_sort_key)
        for task in list_tasks:
            current_task_path = os.path.join(current_model_path, task)
            list_scripts = sorted(os.listdir(current_task_path), key=custom_sort_key)
            for script in list_scripts:
                current_script_path = os.path.join(current_task_path, script)
                script_content = open(current_script_path, 'r').read()
                if script_content not in distinct_scripts:
                    distinct_scripts.add(script_content)
                else:
                    os.remove(current_script_path)

                    path_segments = current_script_path.split('/')
                    script_name = path_segments[-1]
                    task = path_segments[-2]
                    model_and_temp = path_segments[-3]

                    insert_nested_dict(duplicate_scripts, (model_and_temp, task), script_name)
                    num_duplicate_scripts += 1

                num_total_scripts += 1

    dict_duplicate_scripts = {'num_total_scripts': num_total_scripts,
                              'num_duplicate_scripts': num_duplicate_scripts,
                              'duplicate_scripts': duplicate_scripts}

    dict_path = '../exp_results/duplicate_scripts/duplicate_scripts.json'
    with open(dict_path, 'w') as f:
        json.dump(dict_duplicate_scripts, f)


def delete_empty_folders(folder_path):
    for path, folders, files in os.walk(folder_path, topdown=False):
        if not folders and not files:
            print(f"Deleting empty folder: {path}")
            os.rmdir(path)


def insert_nested_dict(d, keys, value):
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    if keys[-1] not in d or not isinstance(d[keys[-1]], list):
        d[keys[-1]] = []
    d[keys[-1]].append(value)


def clean_functionality_tests():
    functionality_tests_path = '../exp_results/ai_code_distinct/functionality_tests'

    duplicate_scripts_path = '../exp_results/duplicate_scripts/duplicate_scripts.json'
    with open(duplicate_scripts_path, 'r') as f:
        dict_duplicate_scripts = json.load(f)

    duplicate_scripts = dict_duplicate_scripts['duplicate_scripts']

    for model_and_temp in duplicate_scripts.keys():
        model_name = model_and_temp.split('_')[0]
        model_temp = model_and_temp[-8:]
        model_path = os.path.join(functionality_tests_path, model_name, model_temp)
        for task in duplicate_scripts[model_and_temp].keys():
            task_file_name = f'{task}.json'
            target_path = os.path.join(model_path, task_file_name)

            with open(target_path, 'r') as f:
                dict_funct_test = json.load(f)

            list_duplicate_scripts = duplicate_scripts[model_and_temp][task]

            for script in list_duplicate_scripts:
                del dict_funct_test[script]

            if len(dict_funct_test.keys()) > 1:
                with open(target_path, 'w') as f:
                    json.dump(dict_funct_test, f)
            else:
                os.remove(target_path)


def clean_metric_measurements():
    metric_measurements_path = '../exp_results/ai_code_distinct/metrics_calc'

    duplicate_scripts_path = '../exp_results/duplicate_scripts/duplicate_scripts.json'
    with open(duplicate_scripts_path, 'r') as f:
        dict_duplicate_scripts = json.load(f)

    duplicate_scripts = dict_duplicate_scripts['duplicate_scripts']

    for metric_folder in sorted(os.listdir(metric_measurements_path)):
        metric_folder_path = os.path.join(metric_measurements_path, metric_folder)
        for model_and_temp in list(duplicate_scripts.keys()):
            print(f'Cleaning {model_and_temp}')
            for task in duplicate_scripts[model_and_temp].keys():
                task_file_name = f'{task}-complete.csv'
                target_csv_path = os.path.join(metric_folder_path, task_file_name)
                df_metrics = pd.read_csv(target_csv_path)

                df_metrics['model&temp'] = df_metrics['model&temp'].astype('category')
                df_metrics['script'] = df_metrics['script'].astype('category')

                list_duplicate_scripts = duplicate_scripts[model_and_temp][task]
                for script in list_duplicate_scripts:
                    df_metrics.drop(df_metrics[(df_metrics['model&temp'] == model_and_temp) &
                                               (df_metrics['script'] == script)].index, inplace=True)

                df_metrics.to_csv(target_csv_path, index=False)
