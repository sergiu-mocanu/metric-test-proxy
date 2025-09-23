import copy
import os
import json
from typing import LiteralString

import numpy as np
import pandas as pd
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from pathlib import Path

from metric_test_proxy.classifiers.enum import Classifier, classifier_to_title
from metric_test_proxy.metric_measurement.enum import TextMetric, CodeDataset, metric_to_title
from metric_test_proxy.pathing import get_path as gp

test_pred_file_name = 'test_pred.json'
precision_recall_file_name = 'precision_recall.json'
avg_var_file_name = 'avg_mad.json'

current_date = None


def initialize_current_date():
    """Initialize the current date and time. Doing so will prompt the creation of a separate directory for the
    classification results, therefore avoiding overriding previous data.
    """
    global current_date
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M")


def get_metric_suffix(metric: TextMetric):
    """Return the file-name suffix for classification results."""
    if metric is None:
        return ''
    else:
        return f'{metric}_'


def prepare_x_y(code_dataset: CodeDataset, list_metrics: list[TextMetric]):
    """Load the textual-similarity score and the functional test results into train and test sets."""
    metric_res_folder_path = gp.get_metric_score_path(code_dataset)

    x = pd.DataFrame()
    y = pd.DataFrame()
    label_df_initialized = False

    for metric in list_metrics:
        metric_name = metric.value
        metric_file_name = f'{metric_name}.csv'
        metric_file_path = os.path.join(metric_res_folder_path, metric_file_name)
        metric_df = pd.read_csv(metric_file_path)

        if metric == TextMetric.CB:
            codebleu_scores = ['weighted_ngram_match_score', 'syntax_match_score', 'dataflow_match_score']
            x[codebleu_scores] = metric_df[codebleu_scores].values
        else:
            x[metric_name] = metric_df['score'].values

        if not label_df_initialized:
            y = metric_df['pass']
            label_df_initialized = True

    return x, y


def train_test_classifier(classifier, x, y, classification_results, test_predictions, nb_iterations):
    """Run `nb_iterations` iterations of classifier training and testing with different splits of train/test datasets.

    Args:
        classifier: preloaded scikit-learn classifier module
        x (DataFrame): training data (i.e., textual-similarity score)
        y (DataFrame): training labels (i.e., pass/fail label)
        classification_results (dict): dictionary of classification results
        test_predictions (dict): dictionary of test predictions
        nb_iterations (int): number of iterations to run from scratch

    The results are written in `classification_results` and `test_predictions`.
    """
    start = datetime.now()

    for i in range(nb_iterations):
        if i % 10 == 0 or i == nb_iterations - 1:
            print(f'Iteration {i + 1}')

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)

        class_report = classification_report(y_test, y_pred, target_names=['fail', 'pass'], output_dict=True)

        classification_results[f'iter_{i+1}'] = class_report
        test_predictions[f'iter_{i+1}'] = {}
        test_predictions[f'iter_{i+1}']['y_test'] = y_test.tolist()
        test_predictions[f'iter_{i+1}']['y_pred'] = y_pred.tolist()

    end = datetime.now()
    training_time = end - start
    print(f'Model training and testing time: {training_time}')
    print('_' * 80)


def save_classification_results(classification_results: dict, test_predictions: dict, results_dir_path: str,
                                metric: TextMetric=None):
    """Save classification results to JSON files.

    Args:
        classification_results (dict): dictionary of classification results (precision, recall, f1_score, accuracy)
        test_predictions (dict): dictionary of predictions made during model testing (pass/fail)
        results_dir_path (str): path to save classification results
        metric (TextMetric): target textual metric
    """
    metric_suffix = get_metric_suffix(metric)
    current_p_r_name = metric_suffix + precision_recall_file_name
    current_t_p_name = metric_suffix + test_pred_file_name

    precision_recall_file_path = os.path.join(results_dir_path, current_p_r_name)
    test_pred_file_path = os.path.join(results_dir_path, current_t_p_name)

    with open(precision_recall_file_path, 'w') as f:
        json.dump(classification_results, f)
    with open(test_pred_file_path, 'w') as f:
        json.dump(test_predictions, f)


def compute_logistic_regression(code_dataset: CodeDataset, nb_iterations: int):
    """Run logistic regression classification separately on each textual-similarity metric.
    Each iteration is executed from scratch in order to avoid model overfitting.
    """
    logreg_iterations_folder = gp.get_classification_results_path(code_dataset, Classifier.LR, iterations=True,
                                                                  folder_date=current_date)

    os.makedirs(logreg_iterations_folder, exist_ok=True)

    logistic_regression = LogisticRegression()

    classification_results = {}
    test_pred = {}

    for metric in TextMetric:
        print(f'Computing Logistic Regression for {metric_to_title(metric)} ({nb_iterations} iterations)')
        x, y = prepare_x_y(code_dataset, [metric])

        train_test_classifier(logistic_regression, x, y, classification_results, test_pred, nb_iterations)

        save_classification_results(classification_results, test_pred, logreg_iterations_folder, metric)


def compute_decision_tree(code_dataset: CodeDataset, nb_iterations: int):
    """Run decision tree classification on all textual-similarity metrics.
    Each iteration is executed from scratch in order to avoid model overfitting.
    """
    print(f'Computing Decision Tree with all textual metrics ({nb_iterations} iterations)')
    dt_iterations_folder = gp.get_classification_results_path(code_dataset, Classifier.DT, iterations=True,
                                                              folder_date=current_date)

    os.makedirs(dt_iterations_folder, exist_ok=True)

    decision_tree = DecisionTreeClassifier()

    list_metrics = [metric for metric in TextMetric]
    x, y = prepare_x_y(code_dataset, list_metrics)

    classification_results = {}
    test_pred = {}

    train_test_classifier(decision_tree, x, y, classification_results, test_pred, nb_iterations)

    save_classification_results(classification_results, test_pred, dt_iterations_folder)


def divide_by(input_dict: dict, value: int):
    """Divide the input dictionary elements by a certain value.

    The results are written in `input_dict`.
    """
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


def avg_mad(classification_res: dict) -> dict:
    """Measure the average and the mean absolute deviation of classification results."""
    classification_metrics_template = {'precision': 0.0, 'recall': 0.0, 'f1-score': 0.0, 'support': 0}
    global_template = {'pass': copy.deepcopy(classification_metrics_template),
                       'fail': copy.deepcopy(classification_metrics_template),
                       'accuracy': 0.0,
                       'macro avg': copy.deepcopy(classification_metrics_template),
                       'weighted avg': copy.deepcopy(classification_metrics_template)}

    avg_mad_dict = {
        'avg': copy.deepcopy(global_template),
        'mad': copy.deepcopy(global_template)}

    prediction_metrics = ['precision', 'recall', 'f1-score', 'support']
    prediction_labels = ['pass', 'fail', 'macro avg', 'weighted avg']
    iterations_counter = 0

    for iteration in list(classification_res.keys()):
        iterations_counter += 1

        for pred_metric in prediction_metrics:
            for label in prediction_labels:
                avg_mad_dict['avg'][label][pred_metric] += classification_res[iteration][label][pred_metric]

        avg_mad_dict['avg']['accuracy'] += classification_res[iteration]['accuracy']

    divide_by(avg_mad_dict['avg'], iterations_counter)

    for iteration in list(classification_res.keys()):
        for pred_metric in prediction_metrics[:-1]:
            for label in prediction_labels:

                avg_mad_dict['mad'][label][pred_metric] += abs(
                    avg_mad_dict['avg'][label][pred_metric] - classification_res[iteration][label][pred_metric])

                avg_mad_dict['mad'][label]['support'] = avg_mad_dict['avg'][label]['support']

        avg_mad_dict['mad']['accuracy'] += abs(
            avg_mad_dict['avg']['accuracy'] - classification_res[iteration]['accuracy'])

    divide_by(avg_mad_dict['mad'], iterations_counter)

    return avg_mad_dict


def measure_average_meandev(code_dataset: CodeDataset, classifier: Classifier):
    """Measure the average and the mean absolute deviation of classification results per code dataset and classifier.

    The results are written to JSON files to the output directory.
    """
    iteration_results_folder = gp.get_classification_results_path(code_dataset, classifier, iterations=True,
                                                                  folder_date=current_date)

    for file_name in sorted(os.listdir(iteration_results_folder)):
        if file_name.endswith(precision_recall_file_name):
            current_file_path = os.path.join(iteration_results_folder, file_name)
            with open(current_file_path, 'r') as f:
                logreg_dict = json.load(f)

            metric = next((metric for metric in TextMetric if file_name.startswith(str(metric.value))), None)
            metric_suffix = get_metric_suffix(metric)
            current_avg_var_name = metric_suffix + avg_var_file_name

            parent_folder = Path(iteration_results_folder).parent
            avg_var_file_path = os.path.join(parent_folder, current_avg_var_name)

            avg_mad_dict = avg_mad(logreg_dict)

            with open(avg_var_file_path, 'w') as f:
                json.dump(avg_mad_dict, f)


def format_classification_results(classification_res: dict, first_entry: bool) -> LiteralString:
    """Structure the classification results into a display-friendly format."""
    if first_entry:
        row_format = '{:<12} {:>10.2f} {:>10.2f} {:>10.2f} {:>10}'
        row_format_accuracy = '{:<33}  {:>10.2f} {:>10}'
    else:
        row_format = '{:<12} {:>10.4f} {:>10.4f} {:>10.4f} {:>10}'
        row_format_accuracy = '{:<33}  {:>10.4f} {:>10}'

    formated_rows = []

    for label in ['pass', 'fail']:
        row = row_format.format(
            label,
            classification_res[label]['precision'],
            classification_res[label]['recall'],
            classification_res[label]['f1-score'],
            int(classification_res[label]['support'])
        )
        formated_rows.append(row)

    accuracy_row = row_format_accuracy.format(
        'accuracy',
        classification_res['accuracy'],
        int(classification_res['pass']['support'] + classification_res['fail']['support'])
    )

    for avg_label in ['macro avg', 'weighted avg']:
        avg_row = row_format.format(
            avg_label,
            classification_res[avg_label]['precision'],
            classification_res[avg_label]['recall'],
            classification_res[avg_label]['f1-score'],
            int(classification_res[avg_label]['support'])
        )
        formated_rows.append(avg_row)
    formated_rows.insert(2, f'\n{accuracy_row}')
    return '\n'.join(formated_rows)


def display_classification_results(code_dataset: CodeDataset, classifier: Classifier, target_metric: TextMetric=None,
                                   iterations: bool=False, num_iterations: int=5):
    """Display the classification results per code dataset and classifier."""
    if target_metric is None:
        list_metrics = [metric for metric in TextMetric]
    else:
        list_metrics = [target_metric]

    for metric in list_metrics:
        metric_title = metric_to_title(metric)
        classification_results_folder = gp.get_classification_results_path(code_dataset, classifier,
                                                                    iterations=iterations, folder_date=current_date)

        if iterations:
            target_file_suffix = f'{precision_recall_file_name}'
        else:
            target_file_suffix = f'{avg_var_file_name}'

        if classifier == Classifier.LR:
            target_file_name = f'{metric.value}_{target_file_suffix}'
        else:
            target_file_name = f'{target_file_suffix}'

        target_file_path = os.path.join(classification_results_folder, target_file_name)

        with open(target_file_path, 'r') as f:
            logreg_dict = json.load(f)

        if iterations:
            logreg_dict = logreg_dict[:num_iterations]

        first_entry = True

        if classifier == Classifier.LR:
            print(f'\nLogistic Regression classification results for {metric_title} scores (average and mean absolute '
                  f'deviation):\n')
        else:
            print(f'\nDecision Tree classification results (average and mean absolute deviation):\n')

        for key in list(logreg_dict.keys()):
            print(f'{key}:')
            print(f"{' ':<12} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}")
            print(format_classification_results(logreg_dict[key], first_entry))

            if first_entry:
                print('\n' + '-' * 60 + '\n')
                first_entry = False
        print('\n' + '/' * 60)

        if classifier == Classifier.DT:
            exit(0)


def generate_confusion_matrix(code_dataset: CodeDataset, classifier: Classifier, nb_iterations: int,
                              metric: TextMetric=None, font_size=14):
    """Generate a confusion matrix for the classification results per code dataset and classifier.

    The image is saved to a PNG format in the output directory.
    """
    classification_res_dir = gp.get_classification_results_path(code_dataset, classifier, iterations=True,
                                                                folder_date=current_date)
    metric_suffix = get_metric_suffix(metric)
    target_file_name = metric_suffix + test_pred_file_name
    target_file_path = os.path.join(classification_res_dir, target_file_name)

    matrix_folder_path = gp.get_classification_results_path(code_dataset, classifier, confusion_matrix=True,
                                                            folder_date=current_date)
    os.makedirs(matrix_folder_path, exist_ok=True)

    if metric is None:
        file_name = 'decision_tree.png'
    else:
        file_name = f'{metric.value}.png'

    matrix_file_path = os.path.join(matrix_folder_path, file_name)

    if classifier == Classifier.DT:
        matrix_title = 'Decision Tree'
    else:
        matrix_title = metric_to_title(metric)

    matrix_title += f' ({nb_iterations} iterations)'

    print(f'Generating confusion matrix: {classifier_to_title(classifier)} {metric_to_title(metric)}')

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


def display_confusion_matrix(code_dataset: CodeDataset, classifier: Classifier):
    """Display the confusion matrices saved as a PNG file."""
    confusion_matrix_path = gp.get_classification_results_path(code_dataset, classifier, confusion_matrix=True,
                                                               folder_date=current_date)

    for path, _, files in os.walk(confusion_matrix_path):
        for file in files:
            file_path = os.path.join(path, file)
            image = Image.open(file_path)
            image.show()


def run_full_classification(code_dataset: CodeDataset, classifier: Classifier=None, nb_iterations: int=100,
                            display_results: bool=False, display_write_path: bool=False, display_cm: bool=False,
                            override_results: bool=False):

    if classifier is None:
        classifiers = [e for e in Classifier]
    else:
        classifiers = [classifier]

    if not override_results:
        # Date initialization will prompt the creation of a separate directory for the classification results
        initialize_current_date()

    for current_classifier in classifiers:
        if current_classifier == Classifier.LR:
            compute_logistic_regression(code_dataset, nb_iterations)
            measure_average_meandev(code_dataset, current_classifier)
            for metric in TextMetric:
                generate_confusion_matrix(code_dataset, current_classifier, nb_iterations=nb_iterations, metric=metric)

        elif current_classifier == Classifier.DT:
            compute_decision_tree(code_dataset, nb_iterations)
            measure_average_meandev(code_dataset, current_classifier)
            generate_confusion_matrix(code_dataset, current_classifier, nb_iterations)

        if display_results:
            display_classification_results(code_dataset, current_classifier)

        if display_write_path:
            classification_results_path = gp.get_classification_results_path(code_dataset, current_classifier,
                                                                             folder_date=current_date)
            print(f'\n!!!!The classification results will be saved to {classification_results_path}!!!!')

        if display_cm:
            display_confusion_matrix(code_dataset, current_classifier)

        print('/' * 80) # TODO: re-execute full classification in order to check the correct console delimiters
