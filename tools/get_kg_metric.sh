#!/bin/bash

# --- 唯一需要修改的地方 ---
# (The only place you need to edit)
INPUT_JSON="./result/v114/checkpoint-500_MIMIC_combined_hybrid_base_image_trained_tokenizer_results.json"
# --- 修改结束 ---


# --- 自动路径推导 (请勿修改) ---
# (Automatic path derivation (Do not edit))

# 1. 从输入JSON的路径中获取目录名
#    例如: /root/autodl-fs/06_Phi4_GRPO/result/v113
INPUT_DIR=$(dirname "$INPUT_JSON")

# 2. 定义 "标准答案" (Ground Truth) KG 的输出路径
#    (由 create_gt_kg.py 生成)
GT_KG_JSON="${INPUT_DIR}/gt_kg.json"

# 3. 定义 "模型预测" (Prediction) KG 的输出路径
#    (由 01_MIMIC_kg.py 生成)
PRED_KG_JSON="${INPUT_DIR}/pred_kg.json"


# --- 脚本执行 ---

# --- 新增逻辑: 检查文件是否存在 ---
# (New logic: Check if files exist)
if [ -f "$GT_KG_JSON" ] && [ -f "$PRED_KG_JSON" ]; then
    echo "======================================================"
    echo "检测到文件已存在, 跳过步骤 1 和 2。"
    echo "(Files already exist, skipping steps 1 and 2.)"
    echo "  - $GT_KG_JSON"
    echo "  - $PRED_KG_JSON"
    echo "======================================================"
    echo ""
else
    # --- 至少有一个文件不存在, 执行生成步骤 ---
    # (At least one file is missing, execute generation steps)

    echo "======================================================"
    echo "步骤 1: 运行 create_gt_kg.py (生成 '标准答案' KG)"
    echo "======================================================"
    echo "  输入 (Input):     $INPUT_JSON"
    echo "  输出 (Output):    $GT_KG_JSON"
    echo "------------------------------------------------------"

    python 00_GT_kg.py \
        --input_file "$INPUT_JSON" \
        --output_file "$GT_KG_JSON"

    if [ $? -ne 0 ]; then
        echo "错误: 步骤 1 (create_gt_kg.py) 执行失败。"
        exit 1
    fi
    echo "步骤 1 完成。"
    echo ""


    echo "======================================================"
    echo "步骤 2: 运行 01_MIMIC_kg.py (生成 '模型预测' KG)"
    echo "======================================================"
    echo "  输入 (Input):     $INPUT_JSON"
    echo "  输出 (Output):    $PRED_KG_JSON"
    echo "------------------------------------------------------"

    python 01_MIMIC_kg.py \
        --input_file "$INPUT_JSON" \
        --output_file "$PRED_KG_JSON"

    if [ $? -ne 0 ]; then
        echo "错误: 步骤 2 (01_MIMIC_kg.py) 执行失败。"
        exit 1
    fi
    echo "步骤 2 完成。"
    echo ""

fi # --- 文件检查逻辑结束 (End of file check logic) ---


echo "======================================================"
echo "步骤 3: 运行 02_hall_kg.py (比较两者并评估)"
echo "======================================================"
echo "  标准答案 (GT):    $GT_KG_JSON"
echo "  预测文件 (Pred):  $PRED_KG_JSON"
echo "------------------------------------------------------"

python 02_hall_kg.py \
    --gt_file "$GT_KG_JSON" \
    --pred_file "$PRED_KG_JSON"

if [ $? -ne 0 ]; then
    echo "错误: 步骤 3 (02_hall_kg.py) 执行失败。"
    exit 1
fi

echo "======================================================"
echo "评估流程执行完毕。"
echo "======================================================"