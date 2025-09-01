import copy
import os
import json
import itertools
import numpy as np
import pandas as pd
from enum import Enum
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from pathlib import Path
from pathing import get_path as gp

test_pred_file_name = 'test_pred.json'
precision_recall_file_name = 'precision_recall.json'
avg_var_file_name = 'avg_var.json'

metric_names = ['bleu', 'codebleu', 'rouge', 'meteor', 'chrf', 'crystalbleu']


class Classifier(str, Enum):
    LR = 'logistic_regression'
    DT = 'decision_tree'


class Metric(Enum):
    bleu = 0
    codebleu = 1
    rouge = 2
    meteor = 3
    chrf = 4
    crystalbleu = 5


def get_metric_suffix(metric_name):
    if metric_name is None:
        return ''
    else:
        return f'{metric_name}_'


def metric_name_to_title(metric_name):
    # Function that returns the name of a metric used in the confusion matrix representation
    title = ''
    match metric_name:
        case 'bleu':
            title = 'BLEU'
        case 'codebleu':
            title = 'CodeBLEU'
        case 'rouge':
            title = 'ROUGE'
        case 'meteor':
            title = 'METEOR'
        case 'chrf':
            title = 'ChrF'
        case 'crystalbleu':
            title = 'CrystalBLEU'
        case None:
            title = 'DecisionTree'
    return title


def prepare_x_y(dataset_name, list_metrics):
    """
    Function that runs logistic regression over the LLM-generated script results (i.e., establish if a correlation
    exists between metric score and pass/fail label)
    :return: a json file with scores for precision, recall, f1, accuracy
    """
    gp.check_existing_dataset(dataset_name)

    metric_res_folder_path = gp.get_metric_score_path(dataset_name)

    x = pd.DataFrame()
    y = pd.DataFrame()
    label_df_initialized = False

    for metric_name in list_metrics:
        metric_file_name = f'{metric_name}.csv'
        metric_file_path = os.path.join(metric_res_folder_path, metric_file_name)
        metric_df = pd.read_csv(metric_file_path)

        if metric_name == 'codebleu':
            codebleu_scores = ['weighted_ngram_match_score', 'syntax_match_score', 'dataflow_match_score']
            x[codebleu_scores] = metric_df[codebleu_scores].values
        else:
            x[metric_name] = metric_df['score'].values

        if not label_df_initialized:
            y = metric_df['pass']
            label_df_initialized = True

    return x, y


def train_test_classifier(classifier, x, y, classification_results_dict, test_pred_dict, nb_iterations):
    start = datetime.now()

    # Run nb_iterations iterations of classifier training and testing with different split of train/test datasets
    for i in range(nb_iterations):
        if i % 10 == 0 or i == nb_iterations - 1:
            print(f'Iteration {i + 1}')

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)

        classification_results = classification_report(y_test, y_pred, target_names=['fail', 'pass'], output_dict=True)

        classification_results_dict[f'iter_{i+1}'] = classification_results
        test_pred_dict[f'iter_{i+1}'] = {}
        test_pred_dict[f'iter_{i+1}']['y_test'] = y_test.tolist()
        test_pred_dict[f'iter_{i+1}']['y_pred'] = y_pred.tolist()

    end = datetime.now()
    training_time = end - start
    print(f'Model training and testing time: {training_time}')
    print('_' * 80)


def save_classification_results(classification_results_dict, test_pred_dict, results_folder_path, metric_name=None):
    metric_suffix = get_metric_suffix(metric_name)
    current_p_r_name = metric_suffix + precision_recall_file_name
    current_t_p_name = metric_suffix + test_pred_file_name

    precision_recall_file_path = os.path.join(results_folder_path, current_p_r_name)
    test_pred_file_path = os.path.join(results_folder_path, current_t_p_name)

    with open(precision_recall_file_path, 'w') as f:
        json.dump(classification_results_dict, f)
    with open(test_pred_file_path, 'w') as f:
        json.dump(test_pred_dict, f)


def compute_logistic_regression(dataset_name, nb_iterations):
    """
    Function that runs logistic regression over the LLM-generated script results (i.e., establish if a correlation
    exists between metric score and pass/fail label)
    :return: a json file with scores for precision, recall, f1, accuracy
    """
    logreg_iterations_folder = gp.get_classification_results_path(dataset_name, Classifier.LR.value, iterations=True)

    os.makedirs(logreg_iterations_folder, exist_ok=True)

    logistic_regression = LogisticRegression()

    classification_results = {}
    test_pred = {}

    for metric_name in metric_names:
        print(f'Computing \'Logistic Regression\' classification for metric: {metric_name_to_title(metric_name)}')
        x, y = prepare_x_y(dataset_name, [metric_name])

        train_test_classifier(logistic_regression, x, y, classification_results, test_pred, nb_iterations)

        save_classification_results(classification_results, test_pred, logreg_iterations_folder, metric_name)


def compute_decision_tree(dataset_name, nb_iterations):
    print('Computing Decision Tree classification')
    dt_iterations_folder = gp.get_classification_results_path(dataset_name, Classifier.DT.value, iterations=True)

    os.makedirs(dt_iterations_folder, exist_ok=True)

    decision_tree = DecisionTreeClassifier()

    x, y = prepare_x_y(dataset_name, metric_names)

    classification_results = {}
    test_pred = {}

    train_test_classifier(decision_tree, x, y, classification_results, test_pred, nb_iterations)

    save_classification_results(classification_results, test_pred, dt_iterations_folder)


def divide_by(input_dict, value):
    # Function that divides the obtained scores for the average calculation
    input_dict['pass']['precision'] /= value
    input_dict['pass']['recall'] /= value
    input_dict['pass']['f1-score'] /= value

    input_dict['fail']['precision'] /= value
    input_dict['fail']['recall'] /= value
    input_dict['fail']['f1-score'] /= value

    input_dict['accuracy'] /= value

    input_dict['macro avg']['precision'] /= value
    input_dict['macro avg']['recall'] /= value
    input_dict['macro avg']['f1-score'] /= value

    input_dict['weighted avg']['precision'] /= value
    input_dict['weighted avg']['recall'] /= value
    input_dict['weighted avg']['f1-score'] /= value


def measure_average_variance(dataset_name, classifier_name):
    # Function that measures the average and variance values of the 100 logreg-iteration results
    iteration_results_folder = gp.get_classification_results_path(dataset_name, classifier_name, iterations=True)

    classification_metrics_template = {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0}
    global_template = {'pass': copy.deepcopy(classification_metrics_template),
                       'fail': copy.deepcopy(classification_metrics_template),
                       'accuracy': 0.0,
                       'macro avg': copy.deepcopy(classification_metrics_template),
                       'weighted avg': copy.deepcopy(classification_metrics_template)}

    for file_name in sorted(os.listdir(iteration_results_folder)):
        if file_name.endswith(precision_recall_file_name):
            current_file_path = os.path.join(iteration_results_folder, file_name)
            with open(current_file_path, 'r') as f:
                logreg_dict = json.load(f)

            metric_name = next((metric for metric in metric_names if file_name.startswith(metric)), None)
            metric_suffix = get_metric_suffix(metric_name)
            current_avg_var_name = metric_suffix + avg_var_file_name

            parent_folder = Path(iteration_results_folder).parent
            avg_var_file_path = os.path.join(parent_folder, current_avg_var_name)

            avg_var_dict = {
                'average': copy.deepcopy(global_template),
                'variance': copy.deepcopy(global_template)}

            nb_iterations = 0
            for iteration in list(logreg_dict.keys()):
                nb_iterations += 1

                avg_var_dict['average']['pass']['precision'] += logreg_dict[iteration]['pass']['precision']
                avg_var_dict['average']['pass']['recall'] += logreg_dict[iteration]['pass']['recall']
                avg_var_dict['average']['pass']['f1-score'] += logreg_dict[iteration]['pass']['f1-score']
                avg_var_dict['average']['pass']['support'] += logreg_dict[iteration]['pass']['support']

                avg_var_dict['average']['fail']['precision'] += logreg_dict[iteration]['fail']['precision']
                avg_var_dict['average']['fail']['recall'] += logreg_dict[iteration]['fail']['recall']
                avg_var_dict['average']['fail']['f1-score'] += logreg_dict[iteration]['fail']['f1-score']
                avg_var_dict['average']['fail']['support'] += logreg_dict[iteration]['fail']['support']

                avg_var_dict['average']['accuracy'] += logreg_dict[iteration]['accuracy']

                avg_var_dict['average']['macro avg']['precision'] += logreg_dict[iteration]['macro avg']['precision']
                avg_var_dict['average']['macro avg']['recall'] += logreg_dict[iteration]['macro avg']['recall']
                avg_var_dict['average']['macro avg']['f1-score'] += logreg_dict[iteration]['macro avg']['f1-score']
                avg_var_dict['average']['macro avg']['support'] += logreg_dict[iteration]['macro avg']['support']

                avg_var_dict['average']['weighted avg']['precision'] += (
                    logreg_dict)[iteration]['weighted avg']['precision']
                avg_var_dict['average']['weighted avg']['recall'] += (
                    logreg_dict)[iteration]['weighted avg']['recall']
                avg_var_dict['average']['weighted avg']['f1-score'] += (
                    logreg_dict)[iteration]['weighted avg']['f1-score']
                avg_var_dict['average']['weighted avg']['support'] += (
                    logreg_dict)[iteration]['weighted avg']['support']


            divide_by(avg_var_dict['average'], nb_iterations)

            for iteration in list(logreg_dict.keys()):
                avg_var_dict['variance']['pass']['precision'] += abs(avg_var_dict['average']['pass']['precision'] -
                                                                       logreg_dict[iteration]['pass']['precision'])
                avg_var_dict['variance']['pass']['recall'] += abs(avg_var_dict['average']['pass']['recall'] -
                                                                    logreg_dict[iteration]['pass']['recall'])
                avg_var_dict['variance']['pass']['f1-score'] += abs(avg_var_dict['average']['pass']['f1-score'] -
                                                                      logreg_dict[iteration]['pass']['f1-score'])
                avg_var_dict['variance']['pass']['support'] = avg_var_dict['average']['pass']['support']

                avg_var_dict['variance']['fail']['precision'] += abs(
                    avg_var_dict['average']['fail']['precision'] -
                    logreg_dict[iteration]['fail']['precision'])
                avg_var_dict['variance']['fail']['recall'] += abs(
                    avg_var_dict['average']['fail']['recall'] -
                    logreg_dict[iteration]['fail']['recall'])
                avg_var_dict['variance']['fail']['f1-score'] += abs(
                    avg_var_dict['average']['fail']['f1-score'] -
                    logreg_dict[iteration]['fail']['f1-score'])
                avg_var_dict['variance']['fail']['support'] = avg_var_dict['average']['fail']['support']

                avg_var_dict['variance']['accuracy'] += abs(avg_var_dict ['average']['accuracy'] -
                                                              logreg_dict[iteration]['accuracy'])

                avg_var_dict['variance']['macro avg']['precision'] += abs(
                    avg_var_dict['average']['macro avg']['precision'] -
                    logreg_dict[iteration]['macro avg']['precision'])
                avg_var_dict['variance']['macro avg']['recall'] += abs(
                    avg_var_dict['average']['macro avg']['recall'] -
                    logreg_dict[iteration]['macro avg']['recall'])
                avg_var_dict['variance']['macro avg']['f1-score'] += abs(
                    avg_var_dict['average']['macro avg']['f1-score'] -
                    logreg_dict[iteration]['macro avg']['f1-score'])
                avg_var_dict['variance']['macro avg']['support'] = avg_var_dict['average']['macro avg']['support']

                avg_var_dict['variance']['weighted avg']['precision'] += abs(
                    avg_var_dict['average']['weighted avg']['precision'] -
                    logreg_dict[iteration]['weighted avg']['precision'])
                avg_var_dict['variance']['weighted avg']['recall'] += abs(
                    avg_var_dict['average']['weighted avg']['recall'] -
                    logreg_dict[iteration]['weighted avg']['recall'])
                avg_var_dict['variance']['weighted avg']['f1-score'] += abs(
                    avg_var_dict['average']['weighted avg']['f1-score'] -
                    logreg_dict[iteration]['weighted avg']['f1-score'])
                avg_var_dict['variance']['weighted avg']['support'] = (
                    avg_var_dict)['average']['weighted avg']['support']

            divide_by(avg_var_dict['variance'], nb_iterations)

            with open(avg_var_file_path, 'w') as f:
                json.dump(avg_var_dict, f)


def format_logreg_results(logreg_dict):
    row_format = '{:<12} {:>10.2f} {:>10.2f} {:>10.2f} {:>10}'
    row_format_accuracy = '{:<33}  {:>10.2f} {:>10}'

    formated_rows = []

    for label in ['pass', 'fail']:
        row = row_format.format(
            label,
            logreg_dict[label]['precision'],
            logreg_dict[label]['recall'],
            logreg_dict[label]['f1-score'],
            int(logreg_dict[label]['support'])
        )
        formated_rows.append(row)

    accuracy_row = row_format_accuracy.format(
        'accuracy',
        logreg_dict['accuracy'],
        int(logreg_dict['pass']['support'] + logreg_dict['fail']['support'])
    )

    for avg_label in ['macro avg', 'weighted avg']:
        avg_row = row_format.format(
            avg_label,
            logreg_dict[avg_label]['precision'],
            logreg_dict[avg_label]['recall'],
            logreg_dict[avg_label]['f1-score'],
            int(logreg_dict[avg_label]['support'])
        )
        formated_rows.append(avg_row)
    formated_rows.insert(2, f'\n{accuracy_row}')
    return '\n'.join(formated_rows)


def display_classification_results(dataset_name, classifier_name, metric=None, iterations=False, num_iterations=5):
    metric_name = Metric(metric).name
    metric_title = metric_name_to_title(metric_name)
    classification_results_folder = gp.get_classification_results_path(dataset_name, classifier_name,
                                                                       iterations=iterations)

    if iterations:
        target_file_name = precision_recall_file_name
    else:
        target_file_name = avg_var_file_name

    target_file_path = os.path.join(classification_results_folder, target_file_name)

    with open(target_file_path, 'r') as f:
        logreg_dict = json.load(f)

    if iterations:
        logreg_dict = logreg_dict[:num_iterations]

    print(f'\nLogistic Regression results for \"{metric_title}\" metric (average and variance):\n')
    for key in list(logreg_dict.keys()):
        print(f'{key}:')
        print(f"{' ':<12} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}")
        print(format_logreg_results(logreg_dict[key]))
        print('\n' + '-' * 60 + '\n')
    print('\n')


def generate_confusion_matrix(dataset_name, classifier_name, nb_iterations, metric_name=None, font_size=14):
    # Generate the confusion matrix based on the ground truth and predicted labels of pass/fail
    dt_res_path = gp.get_classification_results_path(dataset_name, classifier_name, iterations=True)
    metric_suffix = get_metric_suffix(metric_name)
    target_file_name = metric_suffix + test_pred_file_name
    target_file_path = os.path.join(dt_res_path, target_file_name)

    matrix_folder_path = gp.get_classification_results_path(dataset_name, classifier_name, confusion_matrix=True)
    os.makedirs(matrix_folder_path, exist_ok=True)

    if metric_name is None:
        file_name = 'decision_tree.png'
    else:
        file_name = f'{metric_name}.png'

    matrix_file_path = os.path.join(matrix_folder_path, file_name)

    matrix_title = metric_name_to_title(metric_name)
    matrix_title += f' ({nb_iterations} iterations)'

    if metric_name is None:
        metric_name = ''

    print(f'Generating confusion matrix for metric: {metric_name_to_title(metric_name)} {classifier_name}')

    with open(target_file_path, 'r') as f:
        test_pred_dict = json.load(f)

    # Cumulate the 100 iteration confusion matrices
    cumulative_confusion_matrix = np.zeros((2, 2), dtype=int)

    for iteration in range(nb_iterations):
        current_iteration = f'iter_{iteration+1}'
        y_test = test_pred_dict[current_iteration]['y_test']
        y_pred = test_pred_dict[current_iteration]['y_pred']

        current_confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
        cumulative_confusion_matrix += current_confusion_matrix

    average_confusion_matrix = cumulative_confusion_matrix / 100

    # Normalize the confusion matrix to percentages
    total_predictions = cumulative_confusion_matrix.sum()
    percentage_confusion_matrix = (average_confusion_matrix / total_predictions) * 10000

    class_names = ['fail', 'pass']
    fig, ax = plt.subplots(figsize=(8, 6))
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, fontsize=font_size)
    plt.yticks(tick_marks, class_names, fontsize=font_size)

    # Create the heatmap
    sns.heatmap(pd.DataFrame(percentage_confusion_matrix), annot=True, cmap='YlGnBu', fmt='.2f',
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                annot_kws={"size": font_size+3})
    ax.xaxis.set_label_position('top')
    plt.title(f'{matrix_title}', y=1.05, fontsize=font_size + 2)  # Adjust the title position
    plt.ylabel('Actual label', fontsize=font_size)
    plt.xlabel('Predicted label', fontsize=font_size)

    plt.tight_layout()  # Adjust layout to prevent clipping

    # Save and display the figure
    fig.savefig(matrix_file_path, dpi=96)
    plt.close(fig)


def run_full_exp_protocol(dataset_name, classifier_name, nb_iterations=100):
    if classifier_name == Classifier.LR.value:
        compute_logistic_regression(dataset_name, nb_iterations)
        measure_average_variance(dataset_name, classifier_name)
        for metric in metric_names:
            generate_confusion_matrix(dataset_name, classifier_name, nb_iterations=nb_iterations, metric_name=metric)

    elif classifier_name == Classifier.DT.value:
        compute_decision_tree(dataset_name, nb_iterations)
        measure_average_variance(dataset_name, classifier_name)
        generate_confusion_matrix(dataset_name, classifier_name, nb_iterations)

    else:
        raise Exception(f'Unknown classifier "{classifier_name}"')
