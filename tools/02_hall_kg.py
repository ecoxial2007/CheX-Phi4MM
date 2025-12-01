import json
import argparse  # Added import

# --- Start of Modification ---
# Set up argument parser
parser = argparse.ArgumentParser(description="Calculate P, R, and F1 for knowledge graph extraction.")

parser.add_argument(
    "--gt_file",
    type=str,
    default="/root/autodl-fs/06_Phi4_GRPO/result/gt_kg.json",
    help="Path to the ground truth (gt) JSON file."
)

parser.add_argument(
    "--pred_file",
    type=str,
    default="/root/autodl-fs/06_Phi4_GRPO/result/v100_1000_0.828/kg1000.json",
    help="Path to the predictions JSON file."
)

# Parse the arguments
args = parser.parse_args()
# --- End of Modification ---


# Use arguments to open files
with open(args.gt_file, "r") as f:
    data = json.load(f)

with open(args.pred_file, "r") as f:
    data2 = json.load(f)

p0 = 0
r0 = 0
f0 = 0
for i, content in enumerate(data):
    entity_set1 = set(content["entities"].items())

    entity_set2 = set(data2[i]["entities"].items())

    # print(entity_set2)
    entity_over = entity_set2 & entity_set1
    # print(entity_over)

    if len(entity_set2) == 0:
        precision = 0
    else:
        precision = len(entity_over) / len(entity_set2)

    # 计算召回率 (Recall)
    if len(entity_set1) == 0:
        recall = 0
    else:
        recall = len(entity_over) / len(entity_set1)

    # 计算F1分数
    if precision + recall == 0:
        f = 0
    else:
        f = 2 * (precision * recall) / (precision + recall)

    relation_set1 = set((rel["source_entity"], rel["target_entity"], rel["type"]) for rel in content["relations"])
    relation_set2 = set((rel["source_entity"], rel["target_entity"], rel["type"]) for rel in data2[i]["relations"])
    relation_over = relation_set2 & relation_set1

    if len(relation_set2) == 0:
        precision1 = 0
    else:
        precision1 = len(relation_over) / len(relation_set2)

    # 计算召回率 (Recall)
    if len(relation_set1) == 0:
        recall1 = 0
    else:
        recall1 = len(relation_over) / len(relation_set1)

    # 计算F1分数
    if precision1 + recall1 == 0:
        f1 = 0
    else:
        f1 = 2 * (precision1 * recall1) / (precision1 + recall1)

    f0 = f0 + f
    p0 = p0 + precision
    r0 = r0 + recall

print('Precision:', p0 / len(data))
print('Recall:', r0 / len(data))
print('F1',f0 / len(data))