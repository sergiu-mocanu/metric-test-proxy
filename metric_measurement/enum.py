from enum import StrEnum


class TextMetric(StrEnum):
    BL = 'bleu'
    CB = 'codebleu'
    RG = 'rouge'
    MT = 'meteor'
    CH = 'chrf'
    CR = 'crystalbleu'


class CodeDataset(StrEnum):
    original = 'ai_code'
    distinct = 'ai_code_distinct'