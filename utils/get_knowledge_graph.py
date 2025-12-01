import argparse
import torch
from ReXKG.get_entity_relation import main_entity, main_relation
from utils.get_reward_data import preprocess_sentences_all, preprocess_sentences
from collections import defaultdict


def extract_knowledge_graph(completion, device=None):
    # 增加device参数
    if device is None:
        device = torch.device(
            f"cuda:{torch.cuda.current_device()}"
            if torch.cuda.is_available()
            else "cpu"
        )
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir_relation", default='./utils/test_REXKG/result/run_relation', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--max_span_length', type=int, default=8, 
                        help="spans w/ length up to max_span_length are considered as candidates")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--negative_label", default="no_relation", type=str)
    parser.add_argument('--eval_batch_size', type=int, default=2048, help="batch size during inference")
    
    parser.add_argument("--eval_with_gold", action="store_true", help="Whether to evaluate the relation model with gold entities provided.")
    parser.add_argument("--no_cuda2", action='store_true',help="Whether not to use CUDA when available")
    parser.add_argument('--bert_model_dir', type=str, default=None, help="the base model directory")
    parser.add_argument('--seed2', type=int, default=0)
    parser.add_argument('--context_window', type=int, default=100, help="the context window size W for the entity model")
    args, _ = parser.parse_known_args()
    args.save_json_file = '/utils/train_100.json'

    data, arr = preprocess_sentences_all(completion)
    js = main_entity(args, data, device=device)
    relation_js = main_relation(args, js, device=device)
    data = preprocess_sentences(relation_js) 
    
    # 创建一个字典来存储合并后的数据
    merged_data = defaultdict(lambda: {"sentences":'' , "entities": {}, "relations": []})

    # 遍历数据，按 doc_key 进行合并
    for entry in data:
        num = entry["doc_key"].split('_')[0]
        merged_data[num]["doc_key"] = num
        merged_data[num]["sentences"] = str(merged_data[num]["sentences"]) + ' ' + entry["sentences"]
        merged_data[num]["entities"].update(entry["entities"])
        merged_data[num]["relations"].extend(entry["relations"])

    merged_data_list = list(merged_data.values())
    
    for i in arr:
        new_item = {
            "sentences": completion[i],
            "entities": {},
            "relations": [],
            "doc_key": str(i)
        }
        merged_data_list.insert(i, new_item)
        
    return merged_data_list