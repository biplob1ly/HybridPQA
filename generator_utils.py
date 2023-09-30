import os
import shutil
import json
import string
import logging
import collections
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

logger = logging.getLogger()

GENERATORValResult = collections.namedtuple(
    'GENERATORValResult',
    [
        "val_id",
        "step",
        "metrics",
        "scores"
    ]
)


def format_generator_validation(val_result: GENERATORValResult):
    header = ['val_id', 'step'] + val_result.metrics
    fmt_header = ' | '.join([f"{item:->12}" for item in header])
    values = [val_result.val_id, val_result.step] + val_result.scores
    fmt_value = ' | '.join([f"{item: >12}" for item in values[:2]]) + ' | ' + ' | '.join([f"{item: >12.5f}" for item in values[2:]])
    return fmt_header, fmt_value


class BLEUScorer:
    def __init__(self):
        punctuations = string.punctuation.replace('%', '').replace('-', '')
        self.table = str.maketrans('', '', punctuations)
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update(['d', 'm', 're', 've'])

    def tokenize(self, txt):
        tokenized = [token.lower().translate(self.table) for token in word_tokenize(txt)]
        tokens = [self.stemmer.stem(word) for word in tokenized if word.isalpha() and word not in self.stop_words]
        return tokens

    def compute_bleu_score(self, references, hypothesis):
        # Computes percentage of non-stopwords in source found in target. Stemmed.
        ref_tokens_list = [self.tokenize(ref) for ref in references]
        hyp_tokens = self.tokenize(hypothesis)
        bleu = sentence_bleu(
            ref_tokens_list,
            hyp_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=SmoothingFunction().method1
        )
        return bleu


def save_combined_results(result_data, combined_result_path):
    with open(combined_result_path, 'w') as fout:
        json.dump(result_data, fout, indent=4)


def save_eval_metrics(metrics_dt, eval_metrics_path):
    with open(eval_metrics_path + '.json', 'w') as fout:
        json.dump(metrics_dt, fout, indent=4)

    col_dt = collections.defaultdict(list)
    for metric, score in metrics_dt.items():
        col_dt[metric].append(score)
    df = pd.DataFrame(col_dt)
    with open(eval_metrics_path + '.csv', 'w') as fout:
        df.to_csv(fout, index=False)


def delete(path):
    """path could either be relative or absolute. """
    # check if file or directory exists
    if os.path.isfile(path) or os.path.islink(path):
        # remove file
        os.remove(path)
    elif os.path.isdir(path):
        # remove directory and all its content
        shutil.rmtree(path)
    else:
        raise ValueError("Path {} is not a file or dir.".format(path))