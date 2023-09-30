import csv
import logging
import glob
import random
import collections
from typing import List, Union

import torch
from torch.utils.data import Dataset
import re
import json
import unicodedata
from flatten_json import flatten

RetBatch = collections.namedtuple(
    'RetBatch',
    [
        'qids', 'cids_per_qid', 'srcs_per_qid', 'pos_cids_per_qid',
        'q_input_ids', 'q_attention_mask', 'q_token_type_ids',
        'ctx_input_ids', 'ctx_attention_mask', 'ctx_token_type_ids'
    ]
)

CrossTrainBatch = collections.namedtuple(
    'CrossTrainBatch',
    [
        'input_ids', 'attention_mask', 'token_type_ids', 'labels'
    ]
)

CrossEvalBatch = collections.namedtuple(
    'CrossEvalBatch',
    [
        'qids', 'cids_per_qid', 'srcs_per_qid', 'pos_cids_per_qid',
        'input_ids', 'attention_mask', 'token_type_ids'
    ]
)


RetContext = collections.namedtuple("Context", ["cid", "text", "source"])

logger = logging.getLogger()


def normalize_question(question: str) -> str:
    question = unicodedata.normalize('NFKD', question).encode('ascii', 'ignore').decode("utf-8", "ignore").strip()
    return question


def remove_key(json_dt):
    if isinstance(json_dt, dict):
        json_dt = {k.strip(): remove_key(v) for k, v in json_dt.items() if k != 'normalized_value'}
        if set(json_dt.keys()) == {'unit', 'value'}:
            return f"{json_dt['value']} {json_dt['unit']}"
        elif set(json_dt.keys()) == {'currency', 'value'}:
            return f"{json_dt['value']} {json_dt['currency']}"
        return json_dt
    else:
        return json_dt.strip() if isinstance(json_dt, str) else json_dt


def normalize_attr_context(text, do_flatten):
    s = text
    if ';' in s:
        prefix_len = s.index(':')+1
        s = s[:prefix_len] + '[' + re.sub(';', ',', s[prefix_len:]) + ']'
    try:
        t = re.sub('(\w+):', '"\g<1>":', s)
        json_dt = json.loads('{' + t + '}')
        json_dt = remove_key(json_dt)
        out = json.dumps(flatten(json_dt, ' ') if do_flatten else json_dt)
    except:
        s = re.sub('\"', '', s)
        s = re.sub(':([^[,}{]+)(,|})', ':"\g<1>"\g<2>', s)
        s = re.sub('([^}{\s\d"]+):', '"\g<1>":', s)
        try:
            json_dt = json.loads('{' + s + '}')
            json_dt = remove_key(json_dt)
            out = json.dumps(flatten(json_dt, ' ') if do_flatten else json_dt)
        except:
            out = s
    out = re.sub('["\[\]}{]', '', out)
    out = re.sub(':', ' : ', out)
    out = re.sub('_', ' ', out)
    out = re.sub('\s+', ' ', out)
    return out


def normalize_context(context, do_flatten=True):
    s = context['text']
    s = re.sub(u"(\u2018|\u2019)", "'", s)
    s = re.sub(u"(\u201c|\u201d)", '"', s)
    s = re.sub(u"\u00d7", 'x', s)
    s = re.sub(u"(\u2013|\u2014)", '-', s)
    s = re.sub('\s+', ' ', s)
    s = re.sub('(\d+\.)\s?}', '\g<1>0}', s)
    s = re.sub(r'(\d+(\.\d+)?)\s?[\\"|"|'']', '\g<1> inches ', s)
    s = re.sub(r'(\d+(\.\d+)?)\s?[\\′|′]', '\g<1> feet ', s)
    s = re.sub(r'"|\"|\\"', '', s)
    s = re.sub(r'(\d+(\.\d+)?)\s?lb', '\g<1> pound', s)
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode("utf-8", "ignore").strip()
    s = re.sub('\s+', ' ', s)
    s = re.sub(r'(\d+(\.\d+)?(\s\w+)?)\s?l? x (\d+(\.\d+)?(\s\w+)?)\s?w? x (\d+(\.\d+)?(\s\w+)?)\s?h?', 'length \g<1> x width \g<4> x height \g<7>', s)
    s = re.sub(r'(\d+(\.\d+)?(\s\w+)?)\s?h? x (\d+(\.\d+)?(\s\w+)?)\s?w? x (\d+(\.\d+)?(\s\w+)?)\s?d?', 'height \g<1> x width \g<4> x depth \g<7>', s)
    s = re.sub(r'(\d+(\.\d+)?(\s\w+)?)\s?l? x (\d+(\.\d+)?(\s\w+)?)\s?w?', 'length \g<1> x width \g<4>', s)
    s = re.sub(r'\d+(\.\d+)?', 'number', s)
    s = re.sub('\s+', ' ', s)
    context['text'] = s
    if context.get("source", None) == "attribute":
        context['text'] = normalize_attr_context(context['text'], do_flatten=do_flatten)
    return context


def read_data_from_json_files(paths: List[str]) -> List:
    results = []
    for i, path in enumerate(paths):
        with open(path, "r", encoding="utf-8") as f:
            logger.info("Reading file %s" % path)
            data = json.load(f)
            results.extend(data)
            logger.info("Aggregated data size: {}".format(len(results)))
    return results


def read_data_from_csv_files(paths: List[str]) -> List:
    results = []
    for i, path in enumerate(paths):
        with open(path, "r", encoding="utf-8") as f:
            logger.info("Reading file %s" % path)
            csv2dict = csv.DictReader(f, delimiter='\t')
            data = list(csv2dict)
            results.extend(data)
            logger.info("Aggregated data size: {}".format(len(results)))
    return results


class RetSample:
    qid: str
    question: str
    positive_ctxs: List[RetContext]
    negative_ctxs: List[RetContext]


class CrossSample:
    qid: str
    question: str
    ctx: RetContext
    label: int


# TODO: Apply OOP later
class CrossDataset(Dataset):
    def __init__(
            self,
            file: str,
            normalize: bool = False,
            flatten_attr: bool = False,
            insert_source: bool = True,
            split: str = "train",
            count: int = None
    ):
        self.file = file
        self.data = []
        self.data_files = []
        self.normalize = normalize
        self.flatten_attr = flatten_attr
        self.insert_source = insert_source
        self.split = split
        self.count = count
        self.load_data()

    def __len__(self):
        return len(self.data)

    def build_ret_context(self, ctx):
        if self.normalize:
            ctx = normalize_context(ctx, self.flatten_attr)
        cid = ctx['cid']
        text = (ctx['source'] + ' ' + ctx['text']) if self.insert_source else ctx['text']
        source = ctx['source']
        return RetContext(cid, text, source)

    def __getitem__(self, index):
        example = self.data[index]
        cross_sample = CrossSample()
        cross_sample.qid = example['qid']
        cross_sample.question = example['question']
        cross_sample.ctx = self.build_ret_context({'cid': example['cid'], 'source': example['source'], 'text': example['candidate']})
        cross_sample.label = int(example['label'])
        return cross_sample

    def get_example(self, index):
        return self.data[index]

    def load_data(self):
        if not self.data:
            logger.info("Loading all evidence ranking data")
            self.data_files = glob.glob(self.file)
            logger.info("Data files: %s", self.data_files)
            data = read_data_from_csv_files(self.data_files)
            self.data = data[:self.count]    # TODO: remove subscript here
            logger.info("Total cleaned data size: %d", len(self.data))


class RetDataset(Dataset):
    def __init__(
            self,
            file: str,
            num_pos_ctx: Union[int, None] = None,
            num_total_ctx: Union[int, None] = None,
            normalize: bool = False,
            flatten_attr: bool = False,
            insert_source: bool = True,
            split: str = "train",
            count: int = None
    ):
        self.file = file
        self.data = []
        self.data_files = []
        self.num_pos_ctx = num_pos_ctx
        self.num_total_ctx = num_total_ctx
        self.normalize = normalize
        self.flatten_attr = flatten_attr
        self.insert_source = insert_source
        self.split = split
        self.count = count
        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        ret_sample = RetSample()
        ret_sample.qid = example['qid']
        ret_sample.question = normalize_question(example['question'])
        if self.num_total_ctx:
            pos_ctxs = example['positive_ctxs'][:self.num_pos_ctx]
            if self.num_pos_ctx and len(pos_ctxs) < self.num_pos_ctx:
                pos_ctxs.extend(random.choices(example['positive_ctxs'], k=self.num_pos_ctx-len(pos_ctxs)))
            num_neg_ctx = self.num_total_ctx - len(pos_ctxs)
            neg_ctxs = example['negative_ctxs'][:num_neg_ctx]
            if len(neg_ctxs) < num_neg_ctx:
                neg_ctxs.extend(random.choices(example['negative_ctxs'], k=num_neg_ctx-len(neg_ctxs)))
        elif not self.num_total_ctx and not self.num_pos_ctx:
            pos_ctxs = example['positive_ctxs']
            neg_ctxs = example['negative_ctxs']
        else:
            raise ValueError("Number of total context can't be variable if not test set")
        ret_sample.positive_ctxs = [self.build_ret_context(ctx) for ctx in pos_ctxs]
        ret_sample.negative_ctxs = [self.build_ret_context(ctx) for ctx in neg_ctxs]
        return ret_sample

    def build_ret_context(self, ctx):
        if self.normalize:
            ctx = normalize_context(ctx, self.flatten_attr)
        cid = ctx.get('cid', None)
        text = (ctx['source'] + ' ' + ctx['text']) if self.insert_source and 'source' in ctx else ctx['text']
        source = ctx.get('source', 'review')
        return RetContext(cid, text, source)

    def load_data(self):
        if not self.data:
            logger.info("Loading all evidence ranking data")
            self.data_files = glob.glob(self.file)
            logger.info("Data files: %s", self.data_files)
            data = read_data_from_json_files(self.data_files)
            clean_data = []
            for sample in data:
                if len(sample["positive_ctxs"]) > 0:
                    if len(sample["negative_ctxs"]) <= 0 and self.split == "train":
                        sample["negative_ctxs"] = [random.choice(clean_data)["positive_ctxs"][0] for _ in range(5)]
                    clean_data.append(sample)
            self.data = clean_data[:self.count]    # TODO: remove subscript here
            logger.info("Total cleaned data size: %d", len(self.data))

    def get_example(self, index):
        return self.data[index]


class CrossTrainCollator:
    def __init__(self, tokenizer, question_max_len=None, ctx_max_len=None):
        self.tokenizer = tokenizer
        joint_max_len = question_max_len + ctx_max_len if question_max_len and ctx_max_len else None
        self.joint_max_len = min(joint_max_len, tokenizer.model_max_length) if joint_max_len else tokenizer.model_max_length

    def encode_batch(self, questions, contexts):
        enc = self.tokenizer(
            questions,
            contexts,
            max_length=self.joint_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return enc['input_ids'], enc['attention_mask'], enc['token_type_ids']

    def __call__(self, batch):
        questions, contexts, labels = [], [], []
        for sample in batch:
            questions.append(sample.question)
            contexts.append(sample.ctx.text)
            labels.append(sample.label)
        labels = torch.tensor(labels)
        # Dim: Q x S
        input_ids, attention_mask, token_type_ids = self.encode_batch(questions, contexts)
        return CrossTrainBatch(input_ids, attention_mask, token_type_ids, labels)


class CrossEvalCollator:
    def __init__(self, tokenizer, question_max_len=None, ctx_max_len=None):
        self.tokenizer = tokenizer
        joint_max_len = question_max_len + ctx_max_len if question_max_len and ctx_max_len else None
        self.joint_max_len = min(joint_max_len, tokenizer.model_max_length) if joint_max_len else tokenizer.model_max_length

    def encode_batch(self, questions, contexts):
        enc = self.tokenizer(
            questions,
            contexts,
            max_length=self.joint_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return enc['input_ids'], enc['attention_mask'], enc['token_type_ids']

    def __call__(self, batch):
        batch_questions = []
        batch_contexts = []
        qids = []
        cids_per_qid = []
        srcs_per_qid = []
        pos_cids_per_qid = []
        for sample in batch:
            pos_neg_ctxs = sample.positive_ctxs + sample.negative_ctxs
            # random.shuffle(pos_neg_ctxs)
            sample_cids = []
            sample_sources = []
            sample_ctxs = []
            for ctx in pos_neg_ctxs:
                sample_cids.append(ctx.cid)
                sample_ctxs.append(ctx.text)
                sample_sources.append(ctx.source)
            sample_pos_cids = [ctx.cid for ctx in sample.positive_ctxs]

            # Dim: Q
            batch_questions.extend([sample.question for _ in range(len(pos_neg_ctxs))])
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

        # Dim: Q x S
        input_ids, attention_mask, token_type_ids = self.encode_batch(batch_questions, batch_contexts)
        return CrossEvalBatch(
            qids, cids_per_qid, srcs_per_qid, pos_cids_per_qid,
            input_ids, attention_mask, token_type_ids
        )


class RetCollator:
    def __init__(self, tokenizer, question_max_len=None, ctx_max_len=None):
        self.tokenizer = tokenizer
        self.question_max_len = min(question_max_len, tokenizer.model_max_length) if question_max_len else tokenizer.model_max_length
        self.ctx_max_len = min(ctx_max_len, tokenizer.model_max_length) if ctx_max_len else tokenizer.model_max_length

    def encode_batch(self, batch_text, max_len):
        enc = self.tokenizer(
            batch_text,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return enc['input_ids'], enc['attention_mask'], enc['token_type_ids']

    def __call__(self, batch):
        batch_question = []
        batch_contexts = []
        qids = []
        cids_per_qid = []
        srcs_per_qid = []
        pos_cids_per_qid = []
        for sample in batch:
            pos_neg_ctxs = sample.positive_ctxs + sample.negative_ctxs
            random.shuffle(pos_neg_ctxs)
            sample_cids = []
            sample_sources = []
            sample_ctxs = []
            for ctx in pos_neg_ctxs:
                sample_cids.append(ctx.cid)
                sample_ctxs.append(ctx.text)
                sample_sources.append(ctx.source)
            sample_pos_cids = [ctx.cid for ctx in sample.positive_ctxs]

            # Dim: Q
            batch_question.append(sample.question)
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

        # Dim: Q x S
        q_input_ids, q_attention_mask, q_token_type_ids = self.encode_batch(batch_question, self.question_max_len)
        # Dim: (Q*C) x S
        ctx_input_ids, ctx_attention_mask, ctx_token_type_ids = self.encode_batch(batch_contexts, self.ctx_max_len)
        return RetBatch(
            qids, cids_per_qid, srcs_per_qid, pos_cids_per_qid,
            q_input_ids, q_attention_mask, q_token_type_ids,
            ctx_input_ids, ctx_attention_mask, ctx_token_type_ids
        )


def encode_batch(batch_data, tokenizer):
    enc = tokenizer(
        batch_data,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    return enc


def get_data_batch(batch, tokenizer):
    batch_question = []
    batch_contexts = []
    for sample in batch:
        pos_neg_ctxs = sample['positive_ctxs'] + sample['negative_ctxs']
        sample_ctxs = []
        for ctx in pos_neg_ctxs:
            sample_ctxs.append(re.sub(r'\d+(.\d+)?', 'number', ctx['text']))

        # Dim: Q
        batch_question.append(sample['question'])
        # extend instead of append: grouping all the ctx in single list
        # Dim: Q*C, e.g. C=5
        batch_contexts.extend(sample_ctxs)

    # Dim: Q x S
    query_enc = encode_batch(batch_question, tokenizer)
    # Dim: (Q*C) x S
    ctx_enc = encode_batch(batch_contexts, tokenizer)
    return query_enc, ctx_enc
