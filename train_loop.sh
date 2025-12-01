#!/bin/bash

# =============================================================================
# 训练参数配置
# =============================================================================
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 基础配置
NPROC_PER_NODE=1
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/root/autodl-fs/CheX-Phi4MM/:$PYTHONPATH
export FREEZE_NON_VISION_PARAMS=1

# 模型配置
MODEL_NAME="restore_MIMIC_Think_phi4_ep1_lr6e-5"
MODEL_PATH="checkpoints/${MODEL_NAME}"


# 训练参数
LEARNING_RATES=(4e-06) # 5e-06
NUM_GENERATIONS=(8) # 16 32
BETA_VALUES=(0.5) # 0.4 0.3

# 数据集配置
DATASET_SIZES=(1000) # 2000 3000 5000

SAMPLE_TYPES=("hard_uniform" ) # "uniform" "hard_random" #random
BASE_DATASET_PATH="./data/MIMIC-VQA-CoT"

# 其他固定参数
NUM_EPOCHS=1
PER_DEVICE_TRAIN_BATCH_SIZE=8
PER_DEVICE_EVAL_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=1
MAX_COMPLETION_LENGTH=256
WARMUP_RATIO=0.1
SAVE_STEPS=1000
SAVE_TOTAL_LIMIT=2

# =============================================================================
# 生成时间戳
# =============================================================================
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# =============================================================================
# 遍历所有参数组合
# =============================================================================

for sample_type in "${SAMPLE_TYPES[@]}"; do
    for dataset_size in "${DATASET_SIZES[@]}"; do
        for lr in "${LEARNING_RATES[@]}"; do
            for num_gen in "${NUM_GENERATIONS[@]}"; do
                for beta in "${BETA_VALUES[@]}"; do
                # for base_dataset_path in "${BASE_DATASET_PATHS[@]}"; do

                    # 构建数据集路径
                    DATASET_PATH="${BASE_DATASET_PATH}/${sample_type}/valid_sample_vqa_${dataset_size}.json"

                    # 检查数据集文件是否存在
                    if [ ! -f "$DATASET_PATH" ]; then
                        echo "警告: 数据集文件不存在: $DATASET_PATH"
                        continue
                    fi

                    # 构建日志文件名
                    LOG_NAME="${TIMESTAMP}_${MODEL_NAME}_lr${lr}_gen${num_gen}_beta${beta}_${sample_type}_${dataset_size}.log"

                    echo "=========================================="
                    echo "开始训练:"
                    echo "  时间戳: $TIMESTAMP"
                    echo "  模型: $MODEL_NAME"
                    echo "  学习率: $lr"
                    echo "  生成数量: $num_gen"
                    echo "  Beta: $beta"
                    echo "  采样类型: $sample_type"
                    echo "  数据集大小: $dataset_size"
                    echo "  数据集路径: $DATASET_PATH"
                    echo "  日志文件: $LOG_NAME"
                    echo "=========================================="

                    # 执行训练命令   reward  external_kg_acc
                    nohup torchrun --standalone --nproc_per_node=${NPROC_PER_NODE} \
                      -m swift.cli.rlhf \
                      --model_kwargs '{"device_map": null}' \
                      --rlhf_type grpo \
                      --use_vllm false \
                      --save_only_model false \
                      --external_plugins plugin.py \
                      --reward_funcs  external_r1v_acc external_kg_jac \
                      --model "$MODEL_PATH" \
                      --ref_model "$MODEL_PATH" \
                      --dataset "$DATASET_PATH" \
                      --do_train true \
                      --num_train_epochs $NUM_EPOCHS \
                      --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
                      --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
                      --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
                      --num_generations $num_gen \
                      --max_completion_length $MAX_COMPLETION_LENGTH \
                      --beta $beta \
                      --learning_rate $lr \
                      --warmup_ratio $WARMUP_RATIO \
                      --log_completions true \
                      --save_steps $SAVE_STEPS \
                      --save_total_limit $SAVE_TOTAL_LIMIT \
                      --train_type dummy \
                      --temperature 0.9 \
                      --epsilon 0.2 \
                      --torch_dtype bfloat16 \
                      --attn_impl flash_attn \
                      --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
                      --dynamic_sample true \
                      --max_resample_times 4 \
                      > "$LOG_NAME" 2>&1 &

                    # 获取进程ID
                    TRAIN_PID=$!
                    echo "训练进程已启动，PID: $TRAIN_PID"
                    echo "日志文件: $LOG_NAME"
                    echo ""

                    # 等待当前训练完成再开始下一个（可选）
                    # 如果希望并行训练，可以注释掉下面这行
                    wait $TRAIN_PID

                done
            done
        done
    done
done

echo "所有训练任务已完成！"
