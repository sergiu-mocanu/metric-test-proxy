import os
import signal
import json
import time

import pandas as pd

from pathing import get_path as gp
from ai_code_testing import functional_testing as ft
from metric_measurement.enum import CodeDataset, TextMetric, metric_to_title

import io
import contextlib

stderr = io.StringIO()
with contextlib.redirect_stderr(stderr):
    # Supress warning about missing installation of a deeplearning framework
    import evaluate as ev

from codebleu import calc_codebleu

import re
from collections import Counter
from nltk.util import ngrams
from crystalbleu import corpus_bleu

##################### CrystalBLEU #####################
def tokenize(raw_string: str) -> list[str]:
    """Tokenize input code into a list of tokens."""
    return re.findall(r"\w+|[^\w\s]", raw_string)


def get_python_corpus():
    """Load and tokenize the Python corpus from CrystalBLEU dataset."""
    corpus_path = gp.get_python_corpus_path()
    with open(corpus_path) as f:
        python_corpus = f.read()

    tokenized_corpus = tokenize(python_corpus)
    return tokenized_corpus


def extract_shared_ngrams(corpus: list[str], k: int=500) -> dict[str, int]:
    """Extract the most common ngrams of size 1 to 4 from the corpus."""
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
    """Custom TimeOut exception used during CodeBLEU metric measurement.

    Due to CodeBLEU analyzing the AST of the input code, some AI-generated scripts with repetitive instructions lead
    to stack overflow during the metric measurement. The timeout limit avoids any undesired crashes.
    """
    raise TimeoutError('Execution timeout!')


# Initializing TimeOut exception
signal.signal(signal.SIGALRM, timeout_handler)


def custom_sort_key(s: str) -> tuple[int, str]:
    """Return a sort key for strings using length-lexicographic order.

    Used for ordering project's folders and files.
    """
    return len(s), s


def code_cleanup(script: str, remove_assert: bool=False, remove_exit: bool=False) -> str:
    """Remove unnecessary components of a script (e.g., comments, assert statements)."""
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

        # Index all assert statements
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
                # Index single-line comments
                if line.strip()[0] == '#':
                    comment_index.append(index)
                # Remove in-line comment component
                else:
                    cleaned_up_line = line.split('#', 1)[0]
                    script_lines[index] = cleaned_up_line
                continue

            # Index the first line of multi-line comments
            if '"""' in line or "'''" in line:
                comment_index.append(index)
                if line.count('"""') == 1 or line.count("'''") == 1:
                    multi_line_comment = True
                continue

        # Add indexes for multi-line comments
        if multi_line_comment and ('"""' not in line and "'''" not in line):
            comment_index.append(index)
            continue

        # Index the last line of multi-line comments
        if multi_line_comment and ('"""' in line or "'''" in line):
            multi_line_comment = False
            comment_index.append(index)
            continue

        # Index blank lines
        if len(line) == 0 or line.isspace():
            empty_line_index.append(index)
            continue

    # Merge indexes for comments, empty lines, assert and exit statements
    [comment_index.extend(indexes) for indexes in (empty_line_index, assert_index, exit_line_index)]

    # Remove all the unnecessary script components
    for index in sorted(comment_index, reverse=True):
        del script_lines[index]

    clean_script = '\n'.join(script_lines)
    return clean_script


def list_non_hidden_files(dir_path: str) -> list[str]:
    """Return a list of all non-hidden files in a directory."""
    return [f for f in os.listdir(dir_path) if not f.startswith('.')]


def calculate_metric(metric: TextMetric, baseline_script: str, ai_script: str, metric_calc=None,
                     shared_ngrams: dict[str, int]=None) -> dict | float:
    """
    Measure the textual-similarity metric score between an AI-generated script and the humaneval baseline.

    Args:
        metric (TextMetric): textual-similarity metric
        baseline_script (str): humaneval baseline script
        ai_script (str): AI-generated script
        metric_calc: preloaded textual metric module
        shared_ngrams (dict[str, int]): dictionary of most common ngrams used for CrystalBLEU metric measurement

    Returns:
        A dictionary containing the textual similarity score.
    """
    score = {}

    if not ai_script:
        if metric != TextMetric.CB:
            return 0.0
        else:
            return {"codebleu": 0.0,
                    "ngram_match_score": 0.0,
                    "weighted_ngram_match_score": 0.0,
                    "syntax_match_score": 0.0,
                    "dataflow_match_score": 0.0}

    if metric == TextMetric.CB:
        metric_complete = False
        signal.alarm(2)
        while not metric_complete:
            try:
                score = calc_codebleu(predictions=[ai_script], references=[baseline_script], lang='python')
                signal.alarm(0)
                metric_complete = True
            except TimeoutError:
                print('Timeout Error')
                signal.alarm(2)

    else:
        if metric == TextMetric.RG:
            results = metric_calc.compute(predictions=[ai_script], references=[baseline_script],
                                          rouge_types=['rougeL'])

        elif metric == TextMetric.CR:
            tokenized_baseline = tokenize(baseline_script)
            tokenized_ai_script = tokenize(ai_script)
            results = corpus_bleu([[tokenized_baseline]], [tokenized_ai_script],
                                  ignoring=shared_ngrams)

        else:
            results = metric_calc.compute(predictions=[ai_script], references=[baseline_script])

        metric_name = metric.value

        if metric == TextMetric.RG:
            score = results['rougeL'].item()
        elif metric == TextMetric.MT:
            score = results[metric_name].item()
        elif metric == TextMetric.CH:
            score = results['score'] / 100
        elif metric == TextMetric.CR:
            score = results
        else:
            score = results[metric_name]
    return score


def full_metric_measurement(code_dataset: CodeDataset):
    """Iterate over the dataset of AI-generated scripts and measure the textual-similarity score with the according
    humaneval baseline.

    The results are written to CSV files in the output directory.
    """
    ai_code_path = gp.get_ai_code_path(code_dataset)
    metric_folder_path = gp.get_metric_score_path(code_dataset)
    functionality_test_path = gp.get_functionality_test_path(code_dataset)
    humaneval_baseline_path = gp.get_humaneval_baseline_path()

    list_models_and_temps = sorted(os.listdir(ai_code_path))
    humaneval_scripts = sorted(os.listdir(humaneval_baseline_path))

    list_metrics = [e for e in TextMetric]

    # Experiment-resumption mechanism
    if not os.path.exists(metric_folder_path):
        os.mkdir(metric_folder_path)
        metric_file_exists = False
        script_starting_index = model_and_temp_starting_index = task_starting_index = metric_starting_index = 0

    # Obtain the starting point of exp-resumption
    else:
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

        # Skip to the next task if current task was complete in the previous exp
        if 'complete' in list_tested_tasks[0]:
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
        metric_name = str(current_metric.value)

        target_folder_name = f'{metric_name}_tasks'
        current_metric_path = os.path.join(metric_folder_path, target_folder_name)
        if not os.path.exists(current_metric_path):
            os.mkdir(current_metric_path)

        # Preload external textual-metric module for all metrics except CodeBLEU and CrystalBLEU
        if current_metric != TextMetric.CB and current_metric != TextMetric.CR:
            metric_calc = ev.load(metric_name)
            shared_ngrams = None

        elif current_metric == TextMetric.CR:
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
            print(f'Measuring {metric_to_title(current_metric)} metric for task: {task_name}')
            task_csv_name = task_name + '.csv'
            task_csv_path = os.path.join(current_metric_path, task_csv_name)

            if os.path.exists(task_csv_path):
                task_metric_df = pd.read_csv(task_csv_path)
                task_metric = task_metric_df.to_dict('records')
            else:
                task_metric = []

            # Get the humaneval baseline implementation
            target_humaneval = humaneval_scripts[task_index]
            target_humaneval_path = os.path.join(humaneval_baseline_path, target_humaneval)
            humaneval_content = open(target_humaneval_path, 'r').read()
            humaneval_script = code_cleanup(humaneval_content, remove_assert=True)

            num_models_and_temps = len(list_models_and_temps)

            for model_and_temp_index in range(model_and_temp_starting_index, num_models_and_temps):
                target_model_and_temp = list_models_and_temps[model_and_temp_index]

                target_model_and_temp_path = os.path.join(ai_code_path, target_model_and_temp)

                model_name = target_model_and_temp.split('_temp')[0]
                model_temp = target_model_and_temp[-8:]

                # Load the functional test results of the AI-generated code
                target_functionality_test = os.path.join(functionality_test_path, model_name, model_temp,
                                                         f'{task_name}.json')
                with open(target_functionality_test, 'r') as f:
                    funct_test_results = json.load(f)

                file_write_counter = 100

                scripts_folder = os.path.join(target_model_and_temp_path, task_name)
                list_scripts = sorted(os.listdir(scripts_folder), key=custom_sort_key)

                for script_file in list_scripts[script_starting_index:]:
                    # Extract and clean the AI-generated script
                    target_script_path = os.path.join(target_model_and_temp_path, task_name, script_file)
                    script_content = open(target_script_path).read()
                    cleaned_script = code_cleanup(script_content)

                    script_test_pass = funct_test_results[script_file]['successful']

                    score = calculate_metric(current_metric, humaneval_script, cleaned_script, metric_calc, shared_ngrams)
                    dict_entry = {'model&temp': target_model_and_temp,
                                  'script': script_file,
                                  'pass': script_test_pass}

                    if current_metric != TextMetric.CB:
                        dict_entry.update({'score': score})

                    else:
                        entry_addition = {'codebleu': score['codebleu'],
                                          'ngram_match_score': score['ngram_match_score'],
                                          'weighted_ngram_match_score': score['weighted_ngram_match_score'],
                                          'syntax_match_score': score['syntax_match_score'],
                                          'dataflow_match_score': score['dataflow_match_score']}
                        dict_entry.update(entry_addition)
                    task_metric.append(dict_entry)

                    # Write the results in a csv file every 100 iterations
                    file_write_counter -= 1
                    if script_file == list_scripts[-1]:
                        task_metric_df = pd.DataFrame.from_records(task_metric)
                        task_metric_df.to_csv(task_csv_path, index=False)
                        file_write_counter = 100

                # Experiment resumption mechanism (i.e., reinitialize the starting index)
                if metric_file_exists and not exp_continuation_started:
                    script_starting_index = 0

            if metric_file_exists and not exp_continuation_started:
                model_and_temp_starting_index = 0

            # Mark the resulting csv file as complete
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

    merge_metrics_results(code_dataset)


def merge_metrics_results(code_dataset: CodeDataset):
    """Merge the textual-similarity score files into one single CSV file. The results are initially separated per
    humaneval task and textual metric. The merge is done per textual metric.

    The results are written to CSV files in the output directory.
    """
    metric_results_path = gp.get_metric_score_path(code_dataset)
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


def random_ai_script_metrics(metric: TextMetric=None, functional_test: bool=False):
    if metric is None:
        list_metrics = [e for e in TextMetric]
    else:
        list_metrics = [metric]

    rand_script_path = gp.get_rand_ai_script_path()
    with open(rand_script_path, 'r') as f:
        rand_script_content = f.read()
    rand_script = code_cleanup(rand_script_content, remove_assert=True)

    humaneval_task = rand_script_path.split('/')[-2]
    task_index = int(humaneval_task.split('_')[1])
    baseline_path = gp.get_baseline_by_index(task_index)
    with open(baseline_path, 'r') as f:
        baseline_content = f.read()
    baseline_script = code_cleanup(baseline_content, remove_assert=True)

    print(f'Analyzing AI-script: {rand_script_path}')
    print(f'\n```\n{rand_script}\n```\n')
    time.sleep(3)
    print(f'Against the according HumanEval baseline script: {humaneval_task}')
    print(f'\n```\n{baseline_script}\n```')
    print('_' * 40)
    time.sleep(3)
    
    for current_metric in list_metrics:
        if current_metric != TextMetric.CB and current_metric != TextMetric.CR:
            metric_calc = ev.load(str(current_metric.value))
            shared_ngrams = None

        elif current_metric == TextMetric.CR:
            python_corpus = get_python_corpus()
            shared_ngrams = extract_shared_ngrams(python_corpus)
            metric_calc = None

        else:
            metric_calc = None
            shared_ngrams = None
            
        metric_result = calculate_metric(metric=current_metric, baseline_script=baseline_script, ai_script=rand_script,
                                         metric_calc=metric_calc, shared_ngrams=shared_ngrams)
        if current_metric == TextMetric.CB:
            metric_result = metric_result['codebleu']

        metric_title = metric_to_title(current_metric)

        print(f'Similarity score {metric_title}: {metric_result:.3f}')
        print('_' * 40)

    if functional_test:
        humaneval_test = ft.get_tests_by_index(task_index)
        merged_code = rand_script + '\n\n' + humaneval_test

        print('Executing humaneval test:')
        time.sleep(2)

        test_result = ft.execute_test(merged_code)
        ft.display_single_test_result(test_result)
