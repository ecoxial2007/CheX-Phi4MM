import json
import argparse
import os
import sys
import random
import logging
import time
import torch
from tqdm import tqdm
import numpy as np

from utils.test_REXKG.shared.data_structures import Dataset
from utils.test_REXKG.shared.const import task_ner_labels, get_labelmap
from utils.test_REXKG.entity.utils import convert_dataset_to_samples, batchify, NpEncoder
from utils.test_REXKG.entity.models import EntityModel
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from torch.nn import CrossEntropyLoss
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from utils.test_REXKG.relation.models import BertForRelation, AlbertForRelation
from transformers import AutoTokenizer
from utils.test_REXKG.relation.utils import generate_relation_data, decode_sample_id
from utils.test_REXKG.shared.const import task_rel_labels, task_ner_labels

CLS = "[CLS]"
SEP = "[SEP]"

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger('root')


def output_ner_predictions(ner_id2label, model, batches, dataset):
    """
    Save the prediction as a json file
    """
    ner_result = {}
    span_hidden_table = {}

    for i in range(len(batches)):
        output_dict = model.run_batch(batches[i], training=False)
        pred_ner = output_dict['pred_ner']
        for sample, preds in zip(batches[i], pred_ner):
            sample['doc_key'] = str(sample['doc_key'])
            off = sample['sent_start_in_doc'] - sample['sent_start']
            k = sample['doc_key'] + '-' + str(sample['sentence_ix'])
            ner_result[k] = []
            for span, pred in zip(sample['spans'], preds):
                # print(span,pred)
                # span_id = '%s::%d::(%d,%d)'%(sample['doc_key'], sample['sentence_ix'], span[0]+off, span[1]+off)
                if pred == 0:
                    continue
                ner_result[k].append([span[0] + off, span[1] + off, ner_id2label[pred]])

    js = dataset.js
    for i, doc in enumerate(js):
        doc['doc_key'] = str(doc['doc_key'])
        doc["predicted_ner"] = []
        doc["predicted_relations"] = []
        for j in range(len(doc["sentences"])):
            k = doc['doc_key'] + '-' + str(j)
            if k in ner_result:
                doc["predicted_ner"].append(ner_result[k])
            else:
                logger.info('%s not in NER results!' % k)
                doc["predicted_ner"].append([])

            doc["predicted_relations"].append([])

        js[i] = doc

    return js


def evaluate_relation(model, device, eval_dataloader, eval_label_ids, num_labels, e2e_ngold=None, verbose=True):
    model.eval()
    preds = []
    for input_ids, input_mask, segment_ids, label_ids, sub_idx, obj_idx in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        sub_idx = sub_idx.to(device)
        obj_idx = obj_idx.to(device)
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None, sub_idx=sub_idx, obj_idx=obj_idx)
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)
    if len(preds) > 0:
        preds = np.argmax(preds[0], axis=1)
    else:
        print("Warning: No predictions returned by model")
        preds = np.array([])

    return preds


def print_pred_json(eval_data, eval_examples, preds, id2label):
    rels = dict()
    for ex, pred in zip(eval_examples, preds):
        doc_sent, sub, obj = decode_sample_id(ex['id'])
        if doc_sent not in rels:
            rels[doc_sent] = []
        if pred != 0:
            rels[doc_sent].append([sub[0], sub[1], obj[0], obj[1], id2label[pred]])

    js = eval_data.js
    for doc in js:
        doc['predicted_relations'] = []
        for sid in range(len(doc['sentences'])):
            k = '%s@%d' % (doc['doc_key'], sid)
            doc['predicted_relations'].append(rels.get(k, []))

    # logger.info('Output predictions to %s..'%(output_file))

    return js


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, sub_idx, obj_idx):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.sub_idx = sub_idx
        self.obj_idx = obj_idx


def add_marker_tokens(tokenizer, ner_labels):
    new_tokens = ['<SUBJ_START>', '<SUBJ_END>', '<OBJ_START>', '<OBJ_END>']
    for label in ner_labels:
        new_tokens.append('<SUBJ_START=%s>' % label)
        new_tokens.append('<SUBJ_END=%s>' % label)
        new_tokens.append('<OBJ_START=%s>' % label)
        new_tokens.append('<OBJ_END=%s>' % label)
    for label in ner_labels:
        new_tokens.append('<SUBJ=%s>' % label)
        new_tokens.append('<OBJ=%s>' % label)
    tokenizer.add_tokens(new_tokens)
    # logger.info('# vocab after adding markers: %d'%len(tokenizer))


def convert_examples_to_features(examples, label2id, max_seq_length, tokenizer, special_tokens, unused_tokens=True):
    """
    Loads a data file into a list of `InputBatch`s.
    unused_tokens: whether use [unused1] [unused2] as special tokens
    """

    def get_special_token(w):
        if w not in special_tokens:
            if unused_tokens:
                special_tokens[w] = "[unused%d]" % (len(special_tokens) + 1)
            else:
                special_tokens[w] = ('<' + w + '>').lower()
        return special_tokens[w]

    num_tokens = 0
    max_tokens = 0
    num_fit_examples = 0
    num_shown_examples = 0
    features = []
    for (ex_index, example) in enumerate(examples):

        tokens = [CLS]
        # SUBJECT_START = get_special_token("SUBJ_START")
        # SUBJECT_END = get_special_token("SUBJ_END")
        # OBJECT_START = get_special_token("OBJ_START")
        # OBJECT_END = get_special_token("OBJ_END")
        # SUBJECT_NER = get_special_token("SUBJ=%s" % example['subj_type'])
        # OBJECT_NER = get_special_token("OBJ=%s" % example['obj_type'])

        SUBJECT_START_NER = get_special_token("SUBJ_START=%s" % example['subj_type'])
        SUBJECT_END_NER = get_special_token("SUBJ_END=%s" % example['subj_type'])
        OBJECT_START_NER = get_special_token("OBJ_START=%s" % example['obj_type'])
        OBJECT_END_NER = get_special_token("OBJ_END=%s" % example['obj_type'])

        for i, token in enumerate(example['token']):
            if i == example['subj_start']:
                sub_idx = len(tokens)
                tokens.append(SUBJECT_START_NER)
            if i == example['obj_start']:
                obj_idx = len(tokens)
                tokens.append(OBJECT_START_NER)
            for sub_token in tokenizer.tokenize(token):
                tokens.append(sub_token)
            if i == example['subj_end']:
                tokens.append(SUBJECT_END_NER)
            if i == example['obj_end']:
                tokens.append(OBJECT_END_NER)
        tokens.append(SEP)

        num_tokens += len(tokens)
        max_tokens = max(max_tokens, len(tokens))

        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]
            if sub_idx >= max_seq_length:
                sub_idx = 0
            if obj_idx >= max_seq_length:
                obj_idx = 0
        else:
            num_fit_examples += 1

        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        try:
            label_id = label2id[example['relation']]
        except:
            print(example['relation'])
            label_id = 0
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id,
                          sub_idx=sub_idx,
                          obj_idx=obj_idx))

    return features


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def setseed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main_entity(args, data1):
    task = 'mimic01'

    setseed(args.seed2)

    ner_label2id, ner_id2label = get_labelmap(task_ner_labels[task])
    print('LOAD NER MODEL...')
    num_ner_labels = len(task_ner_labels[task]) + 1
    args.bert_model_dir = 'utils/test_REXKG/result/run_entity'
    model = EntityModel(args, num_ner_labels=num_ner_labels)

    test_data = Dataset(data1, is_augment=False)

    test_samples, test_ner = convert_dataset_to_samples(test_data, args.max_span_length, ner_label2id=ner_label2id,
                                                        context_window=args.context_window)
    test_batches = batchify(test_samples, args.eval_batch_size)
    js = output_ner_predictions(ner_id2label, model, test_batches, test_data)

    return js


def main_relation(args, js):
    model2 = 'utils/test_REXKG/BiomedNLP'

    gpu_id = 0
    torch.cuda.set_device(gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda2 else "cpu")

    test_dataset, test_examples, test_nrel = generate_relation_data(js, use_gold=args.eval_with_gold,
                                                                    context_window=args.context_window)

    setseed(args.seed2)

    # get label_list
    label_list = ["no_relation", "located_at", "suggestive_of", "modify"]

    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    tokenizer = AutoTokenizer.from_pretrained(model2, do_lower_case=True)

    special_tokens = {}

    eval_dataset = test_dataset
    eval_examples = test_examples
    eval_features = convert_examples_to_features(
        test_examples, label2id, args.max_seq_length, tokenizer, special_tokens, unused_tokens=False)
    eval_nrel = test_nrel

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    all_sub_idx = torch.tensor([f.sub_idx for f in eval_features], dtype=torch.long)
    all_obj_idx = torch.tensor([f.obj_idx for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_sub_idx, all_obj_idx)
    eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)
    eval_label_ids = all_label_ids
    model = BertForRelation.from_pretrained(args.output_dir_relation, num_rel_labels=num_labels)
    model.to(device)
    preds = evaluate_relation(model, device, eval_dataloader, eval_label_ids, num_labels, e2e_ngold=eval_nrel)

    relation_js = print_pred_json(eval_dataset, eval_examples, preds, id2label)
    return relation_js

