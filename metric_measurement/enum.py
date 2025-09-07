from enum import StrEnum


class CodeDataset(StrEnum):
    original = 'ai_code'
    distinct = 'ai_code_distinct'


class TextMetric(StrEnum):
    BL = 'bleu'
    CB = 'codebleu'
    RG = 'rouge'
    MT = 'meteor'
    CH = 'chrf'
    CR = 'crystalbleu'


def metric_name_to_title(metric_name: str = None):
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
            title = ''
    return title