import json
from utils.test_REXKG.test_GRPO import main_entity, main_relation
from utils.test_REXKG.result1.run_relation.reverse_structure_data import preprocess_sentences
import argparse
from tqdm import tqdm
import numpy as np
from utils.get_reward_data import preprocess_sentences_all
from collections import defaultdict
import time
import re


def think_kg(completion):
    """
    注意：此函数与 01_MIMIC_kg.py 中的函数完全相同。
    (Note: This function is identical to the one in 01_MIMIC_kg.py)
    """
    parser1 = argparse.ArgumentParser()

    parser1.add_argument("--output_dir_relation", default='utils/test_REXKG/result/run_relation', type=str,
                         help="The output directory where the model predictions and checkpoints will be written.")
    parser1.add_argument('--max_span_length', type=int, default=8,
                         help="spans w/ length up to max_span_length are considered as candidates")
    parser1.add_argument("--max_seq_length", default=256, type=int,
                         help="The maximum total input sequence length after WordPiece tokenization. \n"
                              "Sequences longer than this will be truncated, and sequences shorter \n"
                              "than this will be padded.")
    parser1.add_argument("--negative_label", default="no_relation", type=str)
    parser1.add_argument('--eval_batch_size', type=int, default=2048, help="batch size during inference")

    # parser1.add_argument("--entity_predictions_test", type=str, default="ent_pre_your_test_file.json", help="The entity prediction file of the test set")
    parser1.add_argument("--eval_with_gold", action="store_true",
                         help="Whether to evaluate the relation model with gold entities provided.")
    parser1.add_argument("--no_cuda2", action='store_true', help="Whether not to use CUDA when available")
    parser1.add_argument('--bert_model_dir', type=str, default=None, help="the base model directory")
    parser1.add_argument('--seed2', type=int, default=0)
    parser1.add_argument('--context_window', type=int, default=100,
                         help="the context window size W for the entity model")

    args, unknown_args = parser1.parse_known_args()

    args.save_json_file = 'train_100.json'

    data1, arr = preprocess_sentences_all(completion)
    js = main_entity(args, data1)
    relation_js = main_relation(args, js)
    save = 'your_test_file.json'
    data = preprocess_sentences(relation_js, save)

    # 创建一个字典来存储合并后的数据
    merged_data = defaultdict(lambda: {"explanation": '', "entities": {}, "relations": []})

    # 遍历数据，按 doc_key 进行合并
    for entry in data:
        num = entry["doc_key"].split('_')[0]

        merged_data[num]["doc_key"] = num

        merged_data[num]["explanation"] = str(merged_data[num]["explanation"]) + ' ' + entry[
            "sentences"]  # 合并 sentences
        merged_data[num]["entities"].update(entry["entities"])  # 合并 ner#对实体进行去重，关系比较复杂，但是也可以不去重，体现多次提到的实体的重要性
        merged_data[num]["relations"].extend(entry["relations"])  # 合并 relations

    merged_data_list = list(merged_data.values())
    # print(len(merged_data_list))
    for i in arr:
        new_item = {
            "explanation": completion[i],
            "entities": {},
            "relations": [],
            "doc_key": str(i)
        }
        merged_data_list.insert(i, new_item)
    return merged_data_list


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Extract Ground Truth Knowledge Graph from 'explanation' field.")

    parser.add_argument(
        "--input_file",
        type=str,
        default='/root/autodl-fs/06_Phi4_GRPO/result/v111/checkpoint-990_MIMIC_combined_hybrid_base_image_trained_tokenizer_results.json',
        help="Path to the input JSON file (e.g., the 'results.json' file)."
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default='/root/autodl-fs/06_Phi4_GRPO/result/gt_kg.json',
        help="Path to save the extracted ground truth KG JSON file."
    )

    main_args = parser.parse_args()

    start_time = time.time()

    with open(main_args.input_file, 'r', encoding='utf-8') as f:
        js = json.load(f)

    js = js['results']
    chunk_size = 1000
    chunks = [js[i:i + chunk_size] for i in range(0, len(js), chunk_size)]
    all_data = []

    for j, chunk in enumerate(chunks):
        print(f"正在处理 (GT) 第 {j + 1} 个分块，包含 {len(chunk)} 个元素")

        text = []
        for i, content in enumerate(chunk):
            # --- 这是关键修改 ---
            # 直接从 "explanation" 字段读取
            ground_truth = content['explanation']
            # --- 修改结束 ---

            text.append(ground_truth)
        print(f"已提取 {len(text)} 条 explanation 文本")

        data = think_kg(text)

        print(f"已处理为 {len(data)} 条 KG 数据")

        for i, da in enumerate(data):
            da["doc_key"] = i + j * chunk_size
            all_data.append(da)

    with open(main_args.output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("程序运行时间：", elapsed_time, "秒")
    print(f"Ground Truth KG 文件已保存到: {main_args.output_file}")