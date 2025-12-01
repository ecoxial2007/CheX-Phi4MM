from collections import defaultdict
from tqdm import tqdm
import numpy as np
import json
import os
from typing import Any, Dict

sample_type = 'uniform'  # uniform hard
data_type = 'MIMIC-VQA-CoT'  # 'Answer-First'


def transform_item(item):
    """
    将旧格式的 item 字典转换为新格式的 new_item 字典。
    """
    solution_text = item.get("explanation", '')
    img_ids = item.get('image_ids', [])
    images = []
    for img_id in img_ids:
        image = os.path.join('./data/MIMIC-VQA-CoT/images/', f'{img_id}.png')
        images.append(image)
    new_item = {
        "images": images,
        "messages": [
            {
                "role": "user",
                "content": item.get('question')
            }
        ],
        "solution": solution_text,
        "idx": item.get('idx'),
        "subject_id": item.get('subject_id'),
        "study_id": item.get('study_id'),
        "answer": item.get('answer')
    }

    return new_item


def convert_non_list_values_to_list(data_dict: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    接收一个字典，将其所有非列表（list）类型的 value 转换为包含该 value 的列表。
    如果 value 本身已经是列表，则保持不变。

    Args:
        data_dict (Dict[Any, Any]): 输入的原始字典。

    Returns:
        Dict[Any, Any]: 一个新的字典，其中所有 value 都是列表。
    """
    new_dict = {}
    for key, value in data_dict.items():
        if isinstance(value, list):
            # 如果值已经是列表，直接赋值
            new_dict[key] = value
        else:
            # 如果值不是列表，将其放入一个新列表中
            new_dict[key] = [value]
    return new_dict


def keep_only_list_values(data_dict: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    接收一个字典，移除所有 value 不是列表（list）类型的键值对。

    Args:
        data_dict (Dict[Any, Any]): 输入的原始字典。

    Returns:
        Dict[Any, Any]: 一个新的字典，只包含 value 是列表的键值对。
    """
    # 使用字典推导式可以非常简洁地实现这个功能
    return {
        key: value for key, value in data_dict.items()
        if isinstance(value, list) or key == "solution" or key == "idx" or key == "study_id" or key == "subject_id"
    }


# all_json = 'train_all_vqa.json'
all_json = '/autodl-fs/data/06_Phi4_GRPO/data/Think-First/valid_all_vqa.json'
if os.path.exists(all_json):
    with open(all_json, 'r') as f:
        data = json.load(f)
else:
    print('No data for sampling.')
    quit()
    
for target_total in [1000, 2000, 5000]:
    qa_type_list = []
    new_data = []
    for item in tqdm(data):
        sft_is_correct = item['is_correct']
        try:
            qa_type = [item['semantic_type'], item['question_type'], item['answer_type']]
        except:
            qa_type = [item['question_type'], item['answer_type']]

        item = transform_item(item)

        if sample_type == 'hard_uniform':
            if sft_is_correct is True:
                continue
            else:
                new_data.append(item)
        elif sample_type == 'uniform':
            new_data.append(item)
        else:
            raise ValueError(f'Invalid sample type: {sample_type}')

        qa_type = '_'.join(qa_type)
        qa_type_list.append(qa_type)

    qa_type_indices = defaultdict(list)
    for idx, qa_type in enumerate(qa_type_list):
        qa_type_indices[qa_type].append(idx)

    original_total = len(qa_type_list)  # 原始测试集总量

    # 计算每个类型的配额（保持原始比例）
    qa_type_quotas = {}
    for qa_type, indices in qa_type_indices.items():
        proportion = len(indices) / original_total
        qa_type_quotas[qa_type] = int(round(proportion * target_total))

    # 调整配额总和确保等于目标总量
    quota_sum = sum(qa_type_quotas.values())
    if quota_sum != target_total:
        # 调整最大类型的配额来补偿误差
        diff = target_total - quota_sum
        largest_type = max(qa_type_quotas, key=lambda k: qa_type_quotas[k])
        qa_type_quotas[largest_type] += diff

    # 执行分层抽样
    selected_indices = []
    for qa_type, quota in qa_type_quotas.items():
        if quota == 0:
            continue
        indices = qa_type_indices[qa_type]
        if len(indices) < quota:
            raise ValueError(f"QA类型 {qa_type} 样本不足 ({len(indices)} < {quota})")

        # 使用固定随机种子确保可重复性
        selected = np.random.choice(indices, size=quota, replace=False)
        selected_indices.extend(selected)

    # print(selected_indices)
    save_data = [new_data[i] for i in selected_indices]
    dst_path = f'./data/{data_type}/annotations/{sample_type}/{target_total}.json'
    with open(dst_path, 'w') as f:
        json.dump(save_data, f, indent=4)


