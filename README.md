# CheX-Phi4MM: GRPO-Based Multimodal RL Training

CheX-Phi4MM implements the CheXPO-v2 pipeline for chest X-ray multimodal reasoning. It combines GRPO reinforcement learning, knowledge-graph rewards, and customized Swift 3.3.0.post1 trainers to improve diagnostic accuracy for Phi-4 style models.

## 1. Prepare Data and Weights

All prerequisite assets must be arranged before running any script (see `README6.md` for the original notes).

1. **Datasets**
   - Place the evaluation annotations under `data/MIMIC-VQA-CoT/annotations/{eval,hard_uniform,uniform,...}`.
   - Images live in `data/MIMIC-VQA-CoT/images/`.
   - Download links: `https://drive.google.com/drive/folders/1p-GlXypee4UvnwMmfjn3KVuAh9SBuKz_`.

2. **SFT Checkpoints**
   - Store the restored Phi-4 SFT weights in `./checkpoints/
   restore_MIMIC_Think_phi4_ep1_lr6e-5/`.
   - Download links: `https://drive.google.com/file/d/1k_K1ALsdtSnQkM6RddLNtIqbdXeaof4p/view?usp=drive_link`.

3. **ReXKG assets**
   - Hugging Face model `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` should be downloaded and renamed into `./utils/test_REXKG/BiomedNLP`.
   - Pre-computed ReXKG outputs go under `./utils/test_REXKG/result1/run_entity` and `run_relation` (download from `https://drive.google.com/drive/folders/1sNZDT8bI97AJwBGq_O5Y8oMrTjmA0Fpe`).

4. **GRPO Checkpoints for evaluation**
   - Store the restored GRPO weights in `./checkpoints/anywhere_you_like/`.
   - Download links: `https://drive.google.com/drive/`.

## 2. Environment Setup

```bash
conda create -n chex-phi4mm python=3.10
conda activate chex-phi4mm
pip install -r requirements.txt
```

## 3. Launch Training (`train_loop.sh`)

`train_loop.sh` sweeps over dataset sizes, sampling strategies, and logging seeds. It automatically:
- Enables multi-GPU GRPO via Swift.
- Logs with timestamped filenames under `logs/`.
- Uses `FREEZE_NON_VISION_PARAMS` to keep only Phi-4 vision LoRA and `lm_head` trainable.

Typical usage:

```bash
chmod +x train_loop.sh
./train_loop.sh
```

Important knobs inside the script:
- Learning rate: `6e-5`
- Candidates per prompt: `8`
- KL beta: `0.5`
- Dataset sizes: `1000`, `2000`, `5000`
- Sampling types: `hard_sample`, `uniform_sample`

If you prefer a single configuration run, use `train.sh`.

## 4. Launch Evaluation (`evaluate.sh`)

`evaluate.sh` chains three evaluation modes so you can choose the level of analysis:

1. **Default sweep** – `evaluate.py` iterates over every JSON file defined in the script.
2. **Single annotation file** – pass `--annotation_file` and `--img_root` to focus on one split.
3. **Token-level probabilities** – `evaluate_token_prob.py` emits per-token log probabilities for error analysis.

Each stage shares flags such as `--dynamic_hd`, `--use_flash_attention`, output directory, and model checkpoint. Feel free to comment out unwanted blocks if you only need one mode.

## 5. Project Layout

```
CheX-Phi4MM/
├── train_loop.sh / train.sh        # training entry points
├── evaluate.sh / evaluate.py       # batch evaluation utilities
├── evaluate_token_prob.py          # token-level probability export
├── plugin.py                       # custom reward functions and freezing hooks
├── data/                           # datasets (annotations + images)
├── tools/                          # sampling & KG scripts
├── utils/                          # knowledge-graph helpers + ReXKG assets
├── swift/                          # Swift 3.3.0.post1 (patched)
├── ReXKG/                          # entity/relation modules
└── output/                         # checkpoints, completions, logs
```

## 6. Training Data Schema

Training items follow a multimodal chat format:

```json
{
  "messages": [
    {"role": "user", "content": "Question text"},
    {"role": "assistant", "content": "<think>...</think><answer>...</answer>"}
  ],
  "image": "relative path or base64 X-ray",
  "question": "Standalone question",
  "answer": "Canonical short answer"
}
```

Generated outputs (`output/completions.jsonl`) enrich each sample with:

```json
{
  "step": 1000,
  "prompt": "...",
  "completion": "<think>...</think><answer>...</answer>",
  "per_token_logps": [[-2.1, -1.8, ...]],
  "raw_completion_tokens": [["The", " answer", " is", ...]],
  "rewards": {
    "external_r1v_acc": 1.0,
    "external_kg_jac": 0.72,
    "external_code_reward": 0.0
  }
}
```

## 7. Swift Baseline and Customizations

The `swift/` directory contains the source code from `modelscope/ms-swift` (version `3.3.0.post1`), downloaded from [https://github.com/modelscope/ms-swift/tree/main](https://github.com/modelscope/ms-swift/tree/main). We patched it to better support Phi-4 GRPO by:
- Exposing per-token log probabilities and raw tokens in `swift/trainers/rlhf_trainer/grpo_trainer.py`.
- Adding hooks for reward logging and JSONL exports.
- Integrating environment-driven parameter freezing to avoid editing Swift internals for each run.

Refer to the upstream README for baseline usage plus our diffs for Phi-4 specifics.

## 8. Reward Plugins (`plugin.py`)

`plugin.py` registers multiple ORM rewards for Swift:

- `external_r1v_acc` (`MultiModalAccuracyORM`): parses `<answer>` spans or fallback sentences and returns binary accuracy based on overlap.
- `external_kg_jac`: compares predicted knowledge graphs with ground truth using entity/relation IoU.

When `FREEZE_NON_VISION_PARAMS=1`, the module dynamically patches `GRPOTrainer` so that only vision-related LoRA weights and `lm_head` remain trainable while also printing distributed states.

## 9. Sampling and Evaluation Tools (`tools/`)

- `sample_data.py`: builds validation subsets using `uniform` or `hard_sample` policies. You can extend it to mix Answer-First / Think-First pools.
- `0x_*_kg.py` scripts & `get_kg_metric.sh`: regenerate or audit knowledge-graph annotations.

⚠️ The evaluation helpers in `tools/` are prototypes and have not been fully regression-tested. Please review and adapt them to your workflow before large-scale use.

## 10. Citation

```
To be added.
```

## 11. Acknowledgements

We thank the ModelScope Swift team for releasing Swift, the authors of ReXKG and related datasets that make CheX-Phi4MM possible.

---

Ensure all environment variables and datasets are correctly configured before launching GRPO runs. Adjust hyperparameters to match your cluster budget and monitoring preferences.
