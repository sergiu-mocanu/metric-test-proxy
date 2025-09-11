import os
import json
import subprocess
import sys

from pathing import get_path as gp
from metric_measurement.textual_metrics import code_cleanup
from metric_measurement.enum import CodeDataset


def custom_sort_key(s: str) -> tuple[int, str]:
    """Return a sort key for strings using length-lexicographic order.

    Used for ordering project's folders and files.
    """
    return len(s), s


def extract_tests_from_script(humaneval_script: str) -> str:
    """Extract humaneval tests from a given script."""
    testing_funct_name = 'def check('

    extracted_checker = humaneval_script.split(testing_funct_name, 1)[1]
    test_funct = testing_funct_name + extracted_checker

    list_lines = test_funct.split('\n')

    del_index = []

    # Index empty lines, comments and unnecessary assert statements
    for index, line in enumerate(list_lines):
        if (len(line) == 0
                or line.isspace()
                or line.strip().startswith('#')
                or 'assert True' in line):
            del_index.append(index)

    # Remove unnecessary components
    for index in reversed(del_index):
        del list_lines[index]

    test_funct = '\n'.join(list_lines)
    return test_funct


def get_tests_by_index(task_index: int) -> str:
    """Extract humaneval tests by index."""
    humaneval_baseline_path = gp.get_humaneval_baseline_path()
    humaneval_scripts = sorted(os.listdir(humaneval_baseline_path))

    humaneval_file_name = humaneval_scripts[task_index]
    humaneval_file_path = os.path.join(humaneval_baseline_path, humaneval_file_name)

    humaneval_content = open(humaneval_file_path, 'r').read()

    tests = extract_tests_from_script(humaneval_content)

    return tests


def execute_test(merged_code: str) -> dict:
    """Execute code merged with tests in an isolated subprocess.

    Args:
        merged_code (str): AI-generated code merged with the according humaneval tests.

    Returns:
        A dictionary containing the functional test result.
    """
    test_result = {}

    try:
        subprocess.run(
            [sys.executable, '-c', merged_code],
            stderr=subprocess.PIPE,
            timeout=2,
            check=True
        )

        test_result['successful'] = True

    # Timeout error
    except subprocess.TimeoutExpired:
        test_result['successful'] = False
        test_result['error_type'] = 'TimeOut'

    # Runtime error
    except subprocess.CalledProcessError as e:
        test_result['successful'] = False

        error_name_and_message = e.stderr.decode().split('\n')[-2]

        if 'AssertionError' in error_name_and_message:
            test_result['error_type'] = 'AssertionError'

        elif ':' in error_name_and_message:
            error_name = error_name_and_message.split(':')[0]
            error_message = error_name_and_message.split(':')[1].strip()
            test_result['error_type'] = error_name
            test_result['error_message'] = error_message

        else:
            test_result['error_type'] = error_name_and_message

    return test_result


def full_functionality_test(code_dataset: CodeDataset):
    """Iterate over all AI-generated scripts, merge with humaneval tests and execute in an isolated subprocess.

    The results are written to JSON files in the output directory.
    """
    ai_code_path = gp.get_ai_code_path(code_dataset)
    funct_test_path = gp.get_functionality_test_path(code_dataset)

    exp_continuation_started = False

    test_file_write_counter = 50

    # Experiment-resumption mechanism
    if os.path.exists(funct_test_path):
        test_file_exists = True

        # Obtain the starting point of exp-resumption
        list_models = sorted(os.listdir(funct_test_path))
        last_tested_model = list_models[-1]
        last_model_path = os.path.join(funct_test_path, last_tested_model)

        list_model_temperatures = sorted(os.listdir(last_model_path))
        last_tested_temperature = list_model_temperatures[-1]
        tasks_folder_path = os.path.join(last_model_path, last_tested_temperature)

        list_tested_tasks = sorted(os.listdir(tasks_folder_path), key=custom_sort_key)
        last_task_name = list_tested_tasks[-1]
        last_task_path = os.path.join(tasks_folder_path, last_task_name)
        with open(last_task_path, 'r') as f:
            dict_test = json.load(f)
            if 'test_complete' in dict_test.keys() and dict_test['test_complete']:
                last_task_index = len(list_tested_tasks)
                last_task_name = f'HumanEval_{last_task_index}.json'
                script_starting_index = 0
            else:
                script_starting_index = len(dict_test.keys())-1

        model_name_and_temp = f'{last_tested_model}_{last_tested_temperature}'
        list_models = sorted(os.listdir(ai_code_path))
        model_temp_starting_index = list_models.index(model_name_and_temp)

        task_starting_index = int(last_task_name.split('_')[1].strip('.json'))

    else:
        test_file_exists = False
        script_starting_index = task_starting_index = model_temp_starting_index = 0

    list_models = sorted(os.listdir(ai_code_path))
    for model_index in range(model_temp_starting_index, len(list_models)):
        model_name_and_temp = list_models[model_index]
        model_path = os.path.join(ai_code_path, model_name_and_temp)

        print(f'Testing model: {model_name_and_temp}')

        list_tasks = sorted(os.listdir(model_path), key=custom_sort_key)

        for task_index in range(task_starting_index, len(list_tasks)):
            # Skip Task_145 due to lack of AI-code that accomplishes the said task
            if task_index == 145:
                continue

            task_name = f'HumanEval_{task_index}'
            model_name = model_name_and_temp.split('_')[0]
            model_temp = model_name_and_temp[-8:]

            test_file_path = os.path.join(funct_test_path, model_name, model_temp, f'{task_name}.json')
            if os.path.exists(test_file_path):
                with open(test_file_path, 'r') as f:
                    dict_test = json.load(f)
            else:
                dict_test: dict[str, bool | dict] = {'test_complete': False}

            test_folder_path = test_file_path.rpartition('/')[0]
            if not os.path.exists(test_folder_path):
                os.makedirs(test_folder_path)

            humaneval_test = get_tests_by_index(task_index)

            generated_scripts_path = os.path.join(model_path, task_name)

            list_generated_scripts = sorted(os.listdir(generated_scripts_path), key=custom_sort_key)

            for script_index in range(script_starting_index, len(list_generated_scripts)): # noqa
                # Clean and merge the AI-generated scripts with functionality test
                script_name = list_generated_scripts[script_index]
                script_path = os.path.join(generated_scripts_path, script_name)
                script_content = open(script_path, 'r').read()
                cleaned_script = code_cleanup(script_content, remove_exit=True)

                merged_code = cleaned_script + '\n\n' + humaneval_test

                test_result = execute_test(merged_code)

                dict_test[script_name] = test_result

                # Write the results in a json file every 50 iterations
                test_file_write_counter -= 1
                if not test_file_write_counter:
                    test_file_write_counter = 50
                    with open(test_file_path, 'w') as f:
                        json.dump(dict_test, f)

            dict_test['test_complete'] = True

            with open(test_file_path, 'w') as f:
                json.dump(dict_test, f)

            # Experiment resumption mechanism (i.e., reinitializing the starting index after re-launching the exp)
            if test_file_exists and not exp_continuation_started:
                script_starting_index = 0
        if test_file_exists and not exp_continuation_started:
            task_starting_index = 0
            exp_continuation_started = True


def display_test_results(code_dataset: CodeDataset):
    """Display the test results for AI-generated scripts."""
    funct_test_path = gp.get_functionality_test_path(code_dataset)

    total_tests_counter = 0
    failed_tests_counter = 0

    for path, folders, files in os.walk(funct_test_path):
        for file_name in files:
            test_file_path = os.path.join(path, file_name)
            with open(test_file_path, 'r') as f:
                model_dict = json.load(f)

            keys = model_dict.keys()
            total_tests_counter += len(keys)

            for key in list(model_dict.keys())[1:]:
                if not model_dict[key]['successful']:
                    failed_tests_counter += 1

    print(f'Total number of tests:  {total_tests_counter:,}')
    print(f'Number of failed tests: {failed_tests_counter:,}')
    print(f'Number of successful tests: {total_tests_counter - failed_tests_counter:,}')

    rate_failed_tests = (100 / total_tests_counter) * failed_tests_counter
    print(f'Rate of failed tests: {rate_failed_tests:.2f}%')
