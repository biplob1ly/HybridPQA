import logging
import random
import glob
import collections
import torch
from torch.utils.data import Dataset
from typing import List
import re
import json
import unicodedata
from flatten_json import flatten

logger = logging.getLogger()

GenSample = collections.namedtuple(
    "GenSample",
    [
        "index",
        "question",
        "contexts",
        "answer"
    ]
)

GenBatch = collections.namedtuple(
    "GenBatch",
    [
        "indices",
        "prompt_ids",
        "prompt_masks",
        "target_ids"
    ]
)


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


class GenDataset(Dataset):
    def __init__(
            self,
            file: str,
            n_context: int,
            normalize: bool = False,
            flatten_attr: bool = False,
            insert_source: bool = True
    ):
        self.file = file
        self.data = []
        self.data_files = []
        self.n_context = n_context
        self.normalize = normalize
        self.flatten_attr = flatten_attr
        self.insert_source = insert_source
        self.question_prefix = 'question:'
        self.ctx_prefix = 'context:'
        self.source_prefix = 'source:'
        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        question = self.question_prefix + " " + example['question']
        answer = random.choice(example['answers'])
        ctxs = example['ctxs'][:self.n_context]
        while self.n_context and len(ctxs) < self.n_context:
            ctxs.append(random.choice(example['ctxs']))
        contexts = []
        for ctx in ctxs:
            if self.insert_source and ctx.get("source", None):
                context = self.source_prefix + " " + ctx["source"] + " " + self.ctx_prefix + " " + ctx["text"]
            else:
                context = self.ctx_prefix + " " + ctx["text"]
            contexts.append(context)
        return GenSample(
            index,
            question,
            contexts,
            answer
        )

    def load_data(self):
        if not self.data:
            logger.info("Loading all answer generation data")
            self.data_files = glob.glob(self.file)
            logger.info("Data files: %s", self.data_files)
            self.data = read_data_from_json_files(self.data_files)    # TODO: remove subscript here
            logger.info("Total cleaned data size: %d", len(self.data))

    def get_example(self, index):
        return self.data[index]


class GenCollator:
    def __init__(self, tokenizer, prompt_max_len=None, answer_max_len=None):
        self.tokenizer = tokenizer
        self.prompt_max_len = min(prompt_max_len, tokenizer.model_max_length) if prompt_max_len else tokenizer.model_max_length
        self.answer_max_len = min(answer_max_len, tokenizer.model_max_length) if answer_max_len else tokenizer.model_max_length

    def encode_targets(self, batch_target):
        enc = self.tokenizer(
            batch_target,
            max_length=self.answer_max_len,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )
        target_ids = enc["input_ids"]
        target_mask = enc["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)
        return target_ids

    def encode_prompts(self, batch_prompts):
        prompt_ids, prompt_masks = [], []
        for sample_prompts in batch_prompts:
            enc = self.tokenizer(
                sample_prompts,
                max_length=self.prompt_max_len,
                padding='max_length',
                return_tensors='pt',
                truncation=True
            )
            prompt_ids.append(enc['input_ids'].unsqueeze(0))
            prompt_masks.append(enc['attention_mask'].unsqueeze(0))
        # print(prompt_ids)
        prompt_ids = torch.cat(prompt_ids, dim=0)
        prompt_masks = torch.cat(prompt_masks, dim=0)
        return prompt_ids, prompt_masks.bool()

    def __call__(self, batch):
        indices = torch.tensor([sample.index for sample in batch])
        batch_target = [sample.answer for sample in batch]
        target_ids = self.encode_targets(batch_target)
        batch_prompts = [[f"{sample.question} {context}" for context in sample.contexts] for sample in batch]
        prompt_ids, prompt_masks = self.encode_prompts(batch_prompts)
        return GenBatch(
            indices,
            prompt_ids,
            prompt_masks,
            target_ids
        )
