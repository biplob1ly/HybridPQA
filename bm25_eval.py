import os
import sys
import csv
import json
import nltk
import string
import logging
import time
from timeit import default_timer as timer

import numpy as np
from rank_bm25 import BM25Okapi
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import argparse
import collections
from retriever.utils.data_utils import RetDataset
from torch.utils.data import DataLoader, SequentialSampler
from retriever_utils import get_ranked_ctxs, save_ranking_results, save_combined_results, save_eval_metrics, compute_metrics


RawRetBatch = collections.namedtuple(
    'RetBatch',
    [
        'qids', 'cids_per_qid', 'srcs_per_qid', 'pos_cids_per_qid',
        'questions', 'contexts'
    ]
)


class BasicTokenizer:
    def __init__(self):
        punctuations = string.punctuation
        self.table = str.maketrans('', '', punctuations)
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def tokenize(self, txt):
        tokens = [token.lower().translate(self.table) for token in word_tokenize(txt)]
        tokens = [self.stemmer.stem(word) for word in tokens if word.isalpha() and word not in self.stop_words]
        return tokens


class RawRetCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        batch_questions = []
        batch_contexts = []
        qids = []
        cids_per_qid = []
        srcs_per_qid = []
        pos_cids_per_qid = []
        for sample in batch:
            pos_neg_ctxs = sample.positive_ctxs + sample.negative_ctxs
            sample_cids = []
            sample_sources = []
            sample_ctxs = []
            for ctx in pos_neg_ctxs:
                sample_cids.append(ctx.cid)
                sample_ctxs.append(ctx.text)
                sample_sources.append(ctx.source)
            sample_pos_cids = [ctx.cid for ctx in sample.positive_ctxs]

            # Dim: Q
            batch_questions.append(sample.question)
            # extend instead of append: grouping all the ctx in single list
            # Dim: Q*C, e.g. C=5
            batch_contexts.extend(sample_ctxs)
            # Dim: Q
            qids.append(sample.qid)
            # Dim: Q x C
            cids_per_qid.append(sample_cids)
            # Dim: Q x C
            srcs_per_qid.append(sample_sources)
            # Dim: Q x PC
            pos_cids_per_qid.append(sample_pos_cids)

        tokenized_questions = [self.tokenizer.tokenize(question) for question in batch_questions]
        tokenized_contexts = [self.tokenizer.tokenize(context) for context in batch_contexts]
        return RawRetBatch(
            qids, cids_per_qid, srcs_per_qid, pos_cids_per_qid,
            tokenized_questions, tokenized_contexts
        )


def get_result_dt(batch):
    start = timer()
    bm25 = BM25Okapi(batch.contexts)
    question = batch.questions[0]
    ctx_scores = bm25.get_scores(question).tolist()
    ctx_scores_list, pred_ctx_ids, pred_ctx_srcs = get_ranked_ctxs(
        ctx_scores,
        batch.cids_per_qid[0],
        batch.srcs_per_qid[0]
    )
    end = timer()
    exec_time = end - start
    result_dt = {
        'qid': batch.qids[0],
        'pred_ctx_sources': pred_ctx_srcs,
        'scores': ctx_scores_list,
        'pred_ctx_ids': pred_ctx_ids,
        'actual_ctx_ids': batch.pos_cids_per_qid[0],
        'exec_time': exec_time
    }
    return result_dt


def evaluate_bm25(args):
    eval_dataset = RetDataset(
        file=args.eval_data_path,
        num_pos_ctx=None,
        num_total_ctx=None,
        normalize=args.normalize,
        flatten_attr=args.flatten_attribute,
        split="test"
    )
    eval_sampler = SequentialSampler(eval_dataset)
    tokenizer = BasicTokenizer()
    collator = RawRetCollator(tokenizer)
    eval_data_loader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=1,
        drop_last=False,
        collate_fn=collator
    )
    result_data = []
    exec_times = []
    ctx_times = []
    for iteration, batch in enumerate(eval_data_loader):
        result_dt = get_result_dt(batch)
        result_data.append(result_dt)
        exec_times.append(result_dt['exec_time'])
        ctx_times.append(result_dt['exec_time']/len(result_dt['pred_ctx_ids']))
    print(len(exec_times))
    mean_exec_time = (sum(exec_times) / len(exec_times)) * 1000
    mean_ctx_time = (sum(ctx_times) / len(ctx_times)) * 1000
    print(f"Mean exec time: {mean_exec_time} ms, total query count: {len(exec_times)}, mean ctx time: {mean_ctx_time}")
    return result_data


def load_result_list(ranking_result_path):
    result_list = []
    with open(ranking_result_path, 'r') as fin:
        for line in fin:
            json_dt = json.loads(line.strip())
            result_list.append(json_dt)
    return result_list


def load_src_weights(unweighted_eval_metrics_path, metric='map'):
    with open(unweighted_eval_metrics_path, 'r') as fin:
        dt = json.load(fin)
    weights = {}
    for src in dt:
        if src != 'all':
            weights[src] = dt[src][metric]
    print(f"Source {metric} as weights:")
    print(weights)
    return weights


def update_result_list(result_list, src_weights):
    for item in result_list:
        pred_ctx_srcs = item['pred_ctx_sources']
        scores = item['scores']
        weighted_scores = []
        for src, score in zip(pred_ctx_srcs, scores):
            weighted_scores.append(src_weights[src] * score)
        indices = np.argsort(weighted_scores)[::-1]
        item['sparse_scores'] = np.array(item['sparse_scores'])[indices].tolist()
        item['dense_scores'] = np.array(item['dense_scores'])[indices].tolist()
        item['scores'] = np.array(weighted_scores)[indices].tolist()
        sorted_ctx_ids, sorted_ctx_srcs = [], []
        for i in indices:
            sorted_ctx_ids.append(item['pred_ctx_ids'][i])
            sorted_ctx_srcs.append(item['pred_ctx_sources'][i])
        item['pred_ctx_ids'] = sorted_ctx_ids
        item['pred_ctx_sources'] = sorted_ctx_srcs


def run(args):
    result_list = evaluate_bm25(args)
    ranking_result_path = os.path.join(args.output_path, 'rank_score_ids.jsonl')
    save_ranking_results(result_list, ranking_result_path)
    logger.info('Rank and score saved in %s', ranking_result_path)
    combined_result_path = os.path.join(args.output_path, 'combined_score_ids.json')
    save_combined_results(result_list, args.eval_data_path, combined_result_path)
    logger.info('Combined score saved in %s', combined_result_path)
    eval_metrics = ["map", "r-precision", "mrr@5", "ndcg", "hit_rate@5", "precision@1"]
    metrics_dt = compute_metrics(result_list, eval_metrics, comp_separate=True)
    eval_metrics_path = os.path.join(args.output_path, 'eval_metrics')
    save_eval_metrics(metrics_dt, eval_metrics_path)
    logger.info('Evaluation done. Score per metric saved in %s', eval_metrics_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_data_path', default="./data/evidence_ranking/mixed/fixed_norm_test.json", help="Test data file path")
    parser.add_argument('--output_path', default="../output_bm25/", help="Eval result file path")
    parser.add_argument('--normalize', action='store_true', help="Whether to normalize test data")
    parser.add_argument('--flatten_attribute', action='store_true', help="Whether to flatten attribute source of test data")
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    logging.basicConfig(
        filename=os.path.join(args.output_path, 'bm25_log.log'),
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logger = logging.getLogger(__name__)
    run(args)
