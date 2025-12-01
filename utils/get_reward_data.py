import os
import glob
import json 
import re
import random
import argparse
from tqdm import tqdm
import pandas as pd 
import nltk
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import PorterStemmer


punc = list(punctuation)
def divide_to_sentences(reports):
    """
    This function is used to divide reports into several sentences.

    Args:
        reports: list[str], each str is a report

    Return:
        reports_sentences: list[list[str]], each list[str] is the divided sentences of one report
    """

    reports_sentences = []

    for report in reports:
        text_list = []

        text_new = parse_decimal(report)
        text_sentences = text_new.split(".")

        for sentence in text_sentences:
            if len(sentence) > 0:
                text_list.append(sentence)

        reports_sentences.append(text_list)

    return reports_sentences


def clean_sentence(reports):
    """
    This function is used to clean the reports.
    For example: This image doesn't show some diseases. --> This image does not show some diseases.
    """

    clean = []
    for report in reports:
        report_list = []

        for text in report:
            text = re.sub(r"n't", " not ", text)
            text = re.sub(r"\'s", " ", text)
            text = re.sub(r"\'ve", " have ", text)
            text = re.sub(r"\'re", " are ", text)
            text = re.sub(r"\'d", " would ", text)
            text = re.sub(r"\'ll", " will ", text)
            # Keep letters only, and convet texts to lower case
            text = re.sub("[^a-z\s]", "", text.lower())
            # Remove punctuations
            text_nopunc = [char for char in text if char not in punc]
            text_nopunc = "".join(text_nopunc)
            wd = []
            for word in text_nopunc.split():
                wd.append(word)
            report_list.append(" ".join(wd))

        clean.append(report_list)

    return clean


def split_sentence(reports):
    """
    Split each sentence into a list of words.
    e.g.,  "a large hiatal hernia is noted" -> ['a', 'large', 'hiatal', 'hernia', 'is', 'noted', '.']
    """

    split_sen = []

    for report in reports:
        report_list = []

        for text in report:
            text_split = text.split()
            text_split.append(".")
            report_list.append(text_split)

        split_sen.append(report_list)

    return split_sen


def parse_decimal(text):
    """
    input: a sentence. e.g. "The size is 5.5 cm."
    return: a sentence. e.g. "The size is 5*5 cm."
    """

    find_float = lambda x: re.search("\d+(\.\d+)", x).group()
    text_list = []

    for word in text.split():
        try:
            decimal = find_float(word)
            new_decimal = decimal.replace(".", "*")
            text_list.append(new_decimal)
        except:
            text_list.append(word)

    return " ".join(text_list)


def find_word_indices(sen, target_word):
    target_words = re.sub('(?<! )(?=[/,-,:,.,!?()])|(?<=[/,-,:,.,!?()])(?! )', r' ',target_word).lower().split()
    start_index = -1
    end_index = -1
    for i, word in enumerate(sen):
        if word == target_words[0] and (start_index == -1 or end_index == -1):
            if sen[i:i+len(target_words)] == target_words:
                start_index = i
                end_index = i + len(target_words) - 1
    return start_index, end_index


def get_ner_list(sen,res_dict_id):
    return_ner_list = []
    for entity, entity_type in res_dict_id.items():
        start_index, end_index = find_word_indices(sen, entity)
        return_ner_list.append([start_index, end_index,entity_type])
    return return_ner_list


def get_sentence_list(ori_report):
    # 去掉换行符
    ori_report = str(ori_report).replace('\n', '')
    #ori_report=re.sub(r'^\[|\]$', '', ori_report)
    # 用句号分割句子
    sentence_list = str(ori_report).split('.')
    return_sentence_list = []
    
    idx = 0
    while idx < len(sentence_list):
        sentence_idx = sentence_list[idx]
        # 检查当前句子是否以数字结尾以及下一个句子是否以数字开头
        if (idx + 1 < len(sentence_list) and 
            re.search(r'\d$', sentence_idx) and 
            re.match(r'^\d', sentence_list[idx + 1])):
            # 将当前句子与下一个句子合并
            return_sentence_list.append(sentence_idx + '.' + sentence_list[idx + 1])
            idx += 1  # 跳过下一个句子
        else:
            return_sentence_list.append(sentence_idx)
        idx += 1
    
    return return_sentence_list
            
    
def preprocess_sentences_all(texts):
    report_list = texts

    final_list = []
    arr=[]
    for i in range(len(report_list)):
        sentence_list = get_sentence_list(report_list[i])
        count=0
        for sen_idx in range(len(sentence_list)):
            sentence = sentence_list[sen_idx]
            sen = re.sub('(?<! )(?=[/,-,:,.,!?()])|(?<=[/,-,:,.,!?()])(?! )', r' ',sentence).lower().split()
            if len(sen) < 2:
                count=count+1
            else:
                temp_dict = {}
                temp_dict["doc_key"] = str(i)+'_'+str(sen_idx)
                temp_dict["sentences"] = [sen]
                temp_dict["ner"] = [[]]
                temp_dict["relations"] = [[]]
                final_list.append(temp_dict)
        if count==len(sentence_list):
            arr.append(i)

    return final_list, arr
            
       
def preprocess_sentences(data):
    '''''
    with open(input_json_file, 'r', encoding='utf-8') as file:
        # 读取 JSON 文件
        data = [json.loads(line) for line in file]
    '''''
    processed_data = []
    for doc in data:
        # 合并句子

        sentences = ' '.join(doc['sentences'][0])
        
        predicted_entities = {}
        for entity_info in doc['predicted_ner'][0]:
            start, end, entity_type = entity_info
            entity_text = ' '.join(doc['sentences'][0][start:end + 1])
            predicted_entities[entity_text] = entity_type
        
        predicted_relations = []
        for relation_info in doc['predicted_relations'][0]:
            if relation_info:  # 确保关系信息不为空
                start1, end1, start2, end2, relation_type = relation_info
                entity1_text = ' '.join(doc['sentences'][0][start1:end1 + 1])
                entity2_text = ' '.join(doc['sentences'][0][start2:end2 + 1])
                predicted_relations.append({'source_entity': entity1_text, 'target_entity': entity2_text, 'type': relation_type})


        processed_doc = {
            'doc_key': doc['doc_key'],
            'sentences': sentences,
            'entities': predicted_entities,
            'relations': predicted_relations
        }

        processed_data.append(processed_doc)

    return processed_data
