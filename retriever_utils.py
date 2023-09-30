import collections
import json
import torch
import copy
import logging
import argparse
import pandas as pd
from ranx import Qrels, Run, evaluate, compare

logger = logging.getLogger()

RETRIEVERValResult = collections.namedtuple(
    'RETRIEVERValResult',
    [
        "val_id",
        "step",
        "metrics",
        "scores"
    ]
)


def parse_cmd_args():
    parser = argparse.ArgumentParser(description='hetPQA')
    parser.add_argument('--path_cfg_exp', type=str, default='', help='The path of experiment yaml file')
    parser.add_argument('--path_data', type=str, default='', help='The path of data files.')
    parser.add_argument('--path_model', type=str, default='', help='The path of model files.')
    parser.add_argument('--path_output', type=str, default='', help='The path of output.')
    parser.add_argument('--retriever_ckpt', type=str, default='', help='RETRIEVER checkpoint name.')
    parser.add_argument('--version', type=str, default='', help='Experiment version.')
    args = parser.parse_args()
    return args


def log_master(global_rank, text):
    if global_rank in [0, -1]:
        logger.info(text)


def format_retriever_run(ret_result: RETRIEVERValResult):
    header = ['val_id', 'step'] + ret_result.metrics
    fmt_header = ' | '.join([f"{item:->12}" for item in header])
    values = [ret_result.val_id, ret_result.step] + ret_result.scores
    fmt_value = ' | '.join([f"{item: >12}" for item in values[:2]]) + ' | ' + ' | '.join([f"{item: >12.5f}" for item in values[2:]])
    return fmt_header, fmt_value


def save_ranking_results(result_list, ranking_result_path):
    with open(ranking_result_path, 'w') as fout:
        for val_dt in result_list:
            json_line = json.dumps(val_dt)
            fout.write(json_line+'\n')


def save_combined_results(result_list, test_data_path, combined_result_path):
    with open(test_data_path, 'r') as fin:
        test_samples = json.load(fin)
    test_samples = [r for r in test_samples if len(r["positive_ctxs"]) > 0]
    for sample, result in zip(test_samples, result_list):
        assert sample['qid'] == result['qid']
        ctx_pred_score = {ctx_id: score for ctx_id, score in zip(result['pred_ctx_ids'], result['scores'])}
        all_ctxs = copy.deepcopy(sample['positive_ctxs'] + sample['negative_ctxs'])
        del sample['negative_ctxs']
        for ctx in all_ctxs:
            ctx['dp'] = ctx_pred_score[ctx['cid']]
        sample['pred_ctxs'] = sorted(all_ctxs, key=lambda x: x['dp'], reverse=True)
    with open(combined_result_path, 'w') as fout:
        json.dump(test_samples, fout, indent=4)


def save_eval_metrics(metrics_dt, eval_metrics_path):
    with open(eval_metrics_path + '.json', 'w') as fout:
        json.dump(metrics_dt, fout, indent=4)

    col_dt = collections.defaultdict(list)
    for source, dt in metrics_dt.items():
        col_dt['source'].append(source)
        for metric, score in dt.items():
            col_dt[metric].append(score)
    df = pd.DataFrame(col_dt)
    with open(eval_metrics_path + '.csv', 'w') as fout:
        df.to_csv(fout, index=False)


def compute_metrics(result_list, metrics, comp_separate=False):
    qrels_dt = collections.defaultdict(dict)
    run_dt = collections.defaultdict(dict)
    count = 0
    for q_dt in result_list:
        ctx_act_score = {ctx_id: 1 for ctx_id in q_dt['actual_ctx_ids']}
        ctx_pred_score = {ctx_id: score for ctx_id, score in zip(q_dt['pred_ctx_ids'], q_dt['scores'])}

        unq_qid = q_dt['qid'] + '_' + str(count)
        if comp_separate:
            ctx_source = {ctx_id: source for ctx_id, source in zip(q_dt['pred_ctx_ids'], q_dt['pred_ctx_sources'])}
            pos_sources = set([ctx_source[ctx_id] for ctx_id in q_dt['actual_ctx_ids']])
            for source in pos_sources:
                qrels_dt[source][unq_qid] = {ctx_id: rel for ctx_id, rel in ctx_act_score.items() if ctx_source[ctx_id]==source}
                run_dt[source][unq_qid] = {ctx_id: score for ctx_id, score in ctx_pred_score.items() if ctx_source[ctx_id]==source}

        qrels_dt['all'][unq_qid] = ctx_act_score
        run_dt['all'][unq_qid] = ctx_pred_score
        count += 1

    score_dict = {}
    for source in qrels_dt:
        qrels = Qrels(qrels_dt[source])
        run = Run(run_dt[source])
        score_dict[source] = evaluate(qrels, run, metrics)
        score_dict[source]['count'] = len(qrels_dt[source])
    return score_dict


def get_ranked_ctxs(scores, ctx_ids, ctx_srcs):
    assert len(scores) == len(ctx_ids)
    sorted_scores, sorted_indices = torch.tensor(scores).sort(descending=True)
    sorted_scores_list = sorted_scores.tolist()
    sorted_ctx_ids, sorted_ctx_srcs = [], []
    for i in sorted_indices.tolist():
        sorted_ctx_ids.append(ctx_ids[i])
        sorted_ctx_srcs.append(ctx_srcs[i])
    return sorted_scores_list, sorted_ctx_ids, sorted_ctx_srcs


def get_weighted_ranked_ctxs(scores, ctx_ids, ctx_srcs, alpha=0.5):
    # assert len(scores['dense']) == len(ctx_ids)
    if not scores['sparse']:
        weighted_scores = torch.tensor(scores['dense'])
    elif not scores['dense']:
        weighted_scores = torch.tensor(scores['sparse'])
    else:
        weighted_scores = alpha * torch.tensor(scores['dense']) + (1 - alpha) * torch.tensor(scores['sparse'])
    sorted_scores, sorted_indices = weighted_scores.sort(descending=True)
    final_scores = {'sparse': [], 'dense': [], 'weighted': []}
    final_scores['weighted'] = sorted_scores.tolist()
    sorted_ctx_ids, sorted_ctx_srcs = [], []
    for i in sorted_indices.tolist():
        sorted_ctx_ids.append(ctx_ids[i])
        sorted_ctx_srcs.append(ctx_srcs[i])
        if scores['sparse']:
            final_scores['sparse'].append(scores['sparse'][i])
        if scores['dense']:
            final_scores['dense'].append(scores['dense'][i])
    return final_scores, sorted_ctx_ids, sorted_ctx_srcs
