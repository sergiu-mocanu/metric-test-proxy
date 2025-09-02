import os
import signal
import json

import pandas as pd
from enum import Enum

from pathing import get_path as gp

from codebleu import calc_codebleu

import io
import contextlib

stderr = io.StringIO()
with contextlib.redirect_stderr(stderr):
    # Supress warning about missing installation of a deeplearning framework
    import evaluate as ev

import re
from collections import Counter
from nltk.util import ngrams
from crystalbleu import corpus_bleu


class TextMetric(str, Enum):
    BL = 'bleu'
    CB = 'codebleu'
    RG = 'rouge'
    MT = 'meteor'
    CH = 'chrf'
    CR = 'crystalbleu'


##################### CrystalBLEU #####################
def tokenize(raw_string):
    return re.findall(r"\w+|[^\w\s]", raw_string)


def get_python_corpus():
    corpus_path = gp.get_python_corpus_path()
    with open(corpus_path) as f:
        python_corpus = f.read()

    tokenized_corpus = tokenize(python_corpus)
    return tokenized_corpus


def extract_shared_ngrams(corpus):
    k = 500
    all_ngrams = []
    for n in range(1, 5):
        all_ngrams.extend(list(ngrams(corpus, n)))
    # Calculate frequencies of all n-grams
    frequencies = Counter(all_ngrams)
    shared_ngrams = dict(frequencies.most_common(k))
    return shared_ngrams

#######################################################
# noinspection PyUnusedLocal
def timeout_handler(signum, frame):
    # Custom TimeOut exception used in 'test_functionality()' function
    raise TimeoutError('Execution timeout!')


# Initializing TimeOut exception
signal.signal(signal.SIGALRM, timeout_handler)


def custom_sort_key(s):
    # A sorting key used to sort strings in a length-lexicographic order (length and alphabetical order)
    return len(s), s


def code_cleanup(script, remove_assert=False, remove_exit=False):
    # Function that removes any unnecessary components of a given script (comments & tests), leaving only the code lines

    # Removing the test component of HumanEval implementation following 'METADATA' information
    if 'METADATA' in script:
        script = script.split('METADATA', 1)[0]
    elif 'def check(candidate)' in script:
        script = script.split('def check(candidate)', 1)[0]

    script_lines = script.splitlines()

    multi_line_comment = False
    comment_index = []
    assert_index = []
    empty_line_index = []
    exit_line_index = []

    for index, line in enumerate(script_lines):

        # Indexing any assert statement
        if remove_assert and 'assert' in line:
            line_elements = tokenize(line)
            if line_elements[0] == 'assert':
                assert_index.append(index)
            continue

        if remove_exit and 'exit(' in line:
            exit_line_index.append(index)
            continue

        if not multi_line_comment:
            if '#' in line:
                # Indexing single-line comments
                if line.strip()[0] == '#':
                    comment_index.append(index)
                # Removing comment component of the line
                else:
                    cleaned_up_line = line.split('#', 1)[0]
                    script_lines[index] = cleaned_up_line
                continue

            # Indexing the first line of multi-line comments
            if '"""' in line or "'''" in line:
                comment_index.append(index)
                if line.count('"""') == 1 or line.count("'''") == 1:
                    multi_line_comment = True
                continue

        # Adding indexes for multi-line comments
        if multi_line_comment and ('"""' not in line and "'''" not in line):
            comment_index.append(index)
            continue

        # Indexing the last line of multi-line comments
        if multi_line_comment and ('"""' in line or "'''" in line):
            multi_line_comment = False
            comment_index.append(index)
            continue

        # Indexing new lines and blank lines
        if len(line) == 0 or line.isspace():
            empty_line_index.append(index)
            continue

    # Merging indexes for comments, empty lines and assert statements
    [comment_index.extend(indexes) for indexes in (empty_line_index, assert_index, exit_line_index)]

    # Removing all the unnecessary parts of code
    for index in sorted(comment_index, reverse=True):
        del script_lines[index]

    # Concatenating the list of script lines
    clean_script = '\n'.join(script_lines)
    return clean_script


def list_non_hidden_files(dir_path):
    # Function that returns the list of visible files from a given directory
    return [f for f in os.listdir(dir_path) if not f.startswith('.')]


# TODO: update the docstring with the new `metric` datatype
def calculate_metric(metric, baseline, generated_script, metric_calc=None, shared_ngrams=None):
    """
    Function that measures the LLM-script score of a given metric against the HumanEval implementation

    :param metric: integer that represents the desired metric to be used
    :param baseline: HumanEval script
    :param generated_script: LLM-generated script
    :param metric_calc: preloaded metric module
    :param shared_ngrams: dictionary of most common ngrams used for CrystalBLEU score measurement
    :return: metric score
    """
    score = {}

    if not generated_script:
        if metric != TextMetric.CB.value:
            return 0
        else:
            return {"codebleu": 0.0,
                    "ngram_match_score": 0.0,
                    "weighted_ngram_match_score": 0.0,
                    "syntax_match_score": 0.0,
                    "dataflow_match_score": 0.0}

    if metric == TextMetric.CB.value:
        metric_complete = False
        signal.alarm(2)
        while not metric_complete:
            try:
                score = calc_codebleu(predictions=[generated_script], references=[baseline], lang='python')
                signal.alarm(0)
                metric_complete = True
            except TimeoutError:
                print('Timeout Error')
                signal.alarm(2)

    else:
        if metric == TextMetric.RG.value:
            results = metric_calc.compute(predictions=[generated_script], references=[baseline], rouge_types=['rougeL'])
        elif metric == TextMetric.CR.value:
            tokenized_baseline = tokenize(baseline)
            tokenized_generated_script = tokenize(generated_script)
            results = corpus_bleu([[tokenized_baseline]], [tokenized_generated_script],
                                  ignoring=shared_ngrams)
        else:
            results = metric_calc.compute(predictions=[generated_script], references=[baseline])

        if metric == TextMetric.RG.value:
            score = results['rougeL'].item()
        elif metric == TextMetric.MT.value:
            score = results[metric].item()
        elif metric == TextMetric.CH.value:
            score = results['score'] / 100
        elif metric == TextMetric.CR.value:
            score = results
        else:
            score = results[metric]
    return score


def full_metric_measurement(dataset_name):
    """
    Function that iterates over the LLM-generated scripts and measures the metric score all the studied metrics
    :return: writes a csv file with the obtained score as well as pass/fail label for each AI-script
    """
    ai_code_path = gp.get_ai_code_path(dataset_name)
    metric_folder_path = gp.get_metric_score_path(dataset_name)
    functionality_test_path = gp.get_functionality_test_path(dataset_name)
    humaneval_baseline_path = gp.get_humaneval_baseline_path()

    list_models_and_temps = sorted(os.listdir(ai_code_path))
    humaneval_scripts = sorted(os.listdir(humaneval_baseline_path))

    list_metrics = [e for e in TextMetric]

    # Experiment-resumption mechanism
    if not os.path.exists(metric_folder_path):
        os.mkdir(metric_folder_path)
        metric_file_exists = False
        script_starting_index = model_and_temp_starting_index = task_starting_index = metric_starting_index = 0

    else:
        # Obtaining the starting point of exp-resumption
        metric_file_exists = True

        list_dir = os.listdir(metric_folder_path)
        list_metric_results = list(filter(lambda x: not x.endswith('.csv'), list_dir))

        metric_starting_index = len(list_metric_results)-1
        last_tested_metric = list_metrics[metric_starting_index]
        metric_name = last_tested_metric.value

        metric_folder_name = f'{metric_name}_tasks'
        last_tested_metric_path = os.path.join(metric_folder_path, metric_folder_name)
        list_tested_tasks = sorted(list_non_hidden_files(last_tested_metric_path), key=custom_sort_key)
        task_starting_index = len(list_tested_tasks)-1

        task_csv_name = list_tested_tasks[0]
        current_task_path = os.path.join(last_tested_metric_path, task_csv_name)
        task_metric_df = pd.read_csv(current_task_path)

        if 'complete' in list_tested_tasks[0]:
            # Skipping to the next task if current task was complete in the previous exp
            task_starting_index += 1

            if task_starting_index == 163:
                metric_starting_index += 1
                task_starting_index = 0

                if metric_starting_index == len(list_metrics):
                    print('Metric measurement complete')
                    exit(0)

            model_and_temp_starting_index = 0
            script_starting_index = 0

        else:
            last_row = task_metric_df.tail(1)
            last_row_series = last_row.iloc[0]
            last_model_and_temp = last_row_series['model&temp']
            last_script = last_row_series['script']

            task_name = task_csv_name.strip('.csv')
            model_and_temp_starting_index = list_models_and_temps.index(str(last_model_and_temp))
            current_model_and_temp_path = os.path.join(ai_code_path, str(last_model_and_temp), task_name)

            list_scripts = sorted(os.listdir(str(current_model_and_temp_path)), key=custom_sort_key)
            script_starting_index = list_scripts.index(str(last_script)) + 1

            if script_starting_index == len(list_scripts):
                model_and_temp_starting_index += 1
                script_starting_index = 0

    exp_continuation_started = False

    for metric_index in range(metric_starting_index, len(list_metrics)):
        current_metric = list_metrics[metric_index]
        metric_name = current_metric.value

        target_folder_name = f'{metric_name}_tasks'
        current_metric_path = os.path.join(metric_folder_path, target_folder_name)
        if not os.path.exists(current_metric_path):
            os.mkdir(current_metric_path)

        # Preloading metric module for all metrics except CodeBLEU
        if metric_index != 1 and metric_index != 5:
            metric_calc = ev.load(metric_name)
            shared_ngrams = None

        elif metric_index == 5:
            python_corpus = get_python_corpus()
            shared_ngrams = extract_shared_ngrams(python_corpus)
            metric_calc = None

        else:
            metric_calc = None
            shared_ngrams = None

        for task_index in range(task_starting_index, 164):
            if task_index == 145:
                continue

            task_name = f'HumanEval_{task_index}'
            print(f'Measuring "{metric_name}" metric for task: {task_name}')
            task_csv_name = task_name + '.csv'
            task_csv_path = os.path.join(current_metric_path, task_csv_name)

            if os.path.exists(task_csv_path):
                task_metric_df = pd.read_csv(task_csv_path)
                task_metric = task_metric_df.to_dict('records')
            else:
                task_metric = []

            # Obtaining the HumanEval implementation as a comparison baseline
            target_humaneval = humaneval_scripts[task_index]
            target_humaneval_path = os.path.join(humaneval_baseline_path, target_humaneval)
            humaneval_content = open(target_humaneval_path, 'r').read()
            humaneval_script = code_cleanup(humaneval_content, remove_assert=True)

            num_models_and_temps = len(list_models_and_temps)

            for model_and_temp_index in range(model_and_temp_starting_index, num_models_and_temps):
                target_model_and_temp = list_models_and_temps[model_and_temp_index]
                # print(f'Analyzing model and temp: {target_model_and_temp}')

                target_model_and_temp_path = os.path.join(ai_code_path, target_model_and_temp)

                model_name = target_model_and_temp.split('_temp')[0]
                model_temp = target_model_and_temp[-8:]

                # Loading the functionality-test results for the current model/temp/task (used for the pass/fail label)
                target_functionality_test = os.path.join(functionality_test_path, model_name, model_temp,
                                                         f'{task_name}.json')
                with open(target_functionality_test, 'r') as f:
                    funct_test_results = json.load(f)

                file_write_counter = 100

                scripts_folder = os.path.join(target_model_and_temp_path, task_name)
                list_scripts = sorted(os.listdir(scripts_folder), key=custom_sort_key)

                for script_file in list_scripts[script_starting_index:]:
                    # Extracting and cleaning the LLM-generated script
                    target_script_path = os.path.join(target_model_and_temp_path, task_name, script_file)
                    script_content = open(target_script_path).read()
                    cleaned_script = code_cleanup(script_content)

                    script_test_pass = funct_test_results[script_file]['successful']

                    # Measuring the metric score of the current script
                    score = calculate_metric(metric_name, humaneval_script, cleaned_script, metric_calc, shared_ngrams)
                    dict_entry = {'model&temp': target_model_and_temp,
                                  'script': script_file,
                                  'pass': script_test_pass}

                    if metric_index != 1:
                        dict_entry.update({'score': score})

                    else:
                        entry_addition = {'codebleu': score['codebleu'],
                                          'ngram_match_score': score['ngram_match_score'],
                                          'weighted_ngram_match_score': score['weighted_ngram_match_score'],
                                          'syntax_match_score': score['syntax_match_score'],
                                          'dataflow_match_score': score['dataflow_match_score']}
                        dict_entry.update(entry_addition)
                    task_metric.append(dict_entry)

                    # Writing the results in a csv file every 100 iterations
                    file_write_counter -= 1
                    if script_file == list_scripts[-1]:
                        task_metric_df = pd.DataFrame.from_records(task_metric)
                        task_metric_df.to_csv(task_csv_path, index=False)
                        file_write_counter = 100

                # Experiment resumption mechanism (i.e., reinitializing the starting index after re-launching the exp)
                if metric_file_exists and not exp_continuation_started:
                    script_starting_index = 0

            if metric_file_exists and not exp_continuation_started:
                model_and_temp_starting_index = 0

            # Marking the resulting csv file as complete
            if os.path.exists(task_csv_path):
                os.remove(task_csv_path)

            task_csv_name = f'{task_name}-complete.csv'
            folder_path = task_csv_path.rsplit('/', 1)[0]
            task_csv_path = os.path.join(folder_path, task_csv_name)
            task_metric_df = pd.DataFrame.from_records(task_metric)
            task_metric_df.to_csv(task_csv_path, index=False)

        if metric_file_exists and not exp_continuation_started:
            task_starting_index = 0
            exp_continuation_started = True

    merge_metrics_results(dataset_name)


def merge_metrics_results(dataset_name):
    metric_results_path = gp.get_metric_score_path(dataset_name)
    for item in sorted(os.listdir(metric_results_path)):
        current_item_path = os.path.join(metric_results_path, item)

        if os.path.isdir(current_item_path):
            merged_df = pd.DataFrame()

            for metric_file in sorted(os.listdir(current_item_path), key=custom_sort_key):
                current_file_path = os.path.join(current_item_path, metric_file)
                current_df = pd.read_csv(current_file_path)

                humaneval_task = metric_file.split('-')[0]
                task_number = humaneval_task.split('_')[1]

                current_df.insert(1, 'task', '')
                current_df['task'] = task_number

                if merged_df.empty:
                    merged_df = current_df
                else:
                    merged_df = pd.concat([merged_df, current_df])

            metric_name = item.split('_')[0]
            merged_file_name = f'{metric_name}.csv'
            csv_path = os.path.join(metric_results_path, merged_file_name)
            merged_df.to_csv(csv_path, index=False)
