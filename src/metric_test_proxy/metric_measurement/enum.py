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


def metric_to_title(metric: TextMetric = None):
    # Function that returns the name of a metric used in the confusion matrix representation
    title = ''
    match metric:
        case TextMetric.BL:
            title = 'BLEU'
        case TextMetric.CB:
            title = 'CodeBLEU'
        case TextMetric.RG:
            title = 'ROUGE'
        case TextMetric.MT:
            title = 'METEOR'
        case TextMetric.CH:
            title = 'ChrF'
        case TextMetric.CR:
            title = 'CrystalBLEU'
        case None:
            title = ''
    return title