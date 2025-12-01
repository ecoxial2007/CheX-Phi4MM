export CUDA_VISIBLE_DEVICES=0

python evaluate.py \
  --model_path ./output/restore_MIMIC_Think_phi4_ep1_lr6e-5/v1-20251109-084105/checkpoint-990 \
  --output_dir ./result/v1  \
  --dynamic_hd 4  \
  --use_flash_attention

python evaluate.py \
  --model_path ./output/restore_MIMIC_Think_phi4_ep1_lr6e-5/v1-20251109-084105/checkpoint-990 \
  --annotation_file ./data/MIMIC-VQA-CoT/annotations/eval/test_basic_vqa.json \
  --img_root ./data/MIMIC-VQA-CoT/images \
  --output_dir ./result/v1  \
  --dynamic_hd 4  \
  --use_flash_attention

python evaluate_token_prob.py \
  --model_path ./output/restore_MIMIC_Think_phi4_ep1_lr6e-5/v1-20251109-084105/checkpoint-990 \
  --annotation_file ./data/MIMIC-VQA-CoT/annotations/eval/test_basic_vqa.json \
  --img_root ./data/MIMIC-VQA-CoT/images \
  --output_dir ./result/v1  \
  --dynamic_hd 4  \
  --use_flash_attention