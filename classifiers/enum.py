from enum import StrEnum


class Classifier(StrEnum):
    LR = 'logistic_regression'
    DT = 'decision_tree'


def classifier_to_title(classifier: Classifier):
    match classifier:
        case Classifier.LR:
            return 'Logistic Regression'
        case Classifier.DT:
            return 'Decision Tree'