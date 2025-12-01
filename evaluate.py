import os
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from torch.utils.data import ConcatDataset

from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)

from utils.mimic import MiMiCEXTVQADataset

USER = '<|user|>'
ASSIST = '<|assistant|>'
END = '<|end|>'

def load_processor_no_special_tokens(base_model_path: str, trained_model_path: str, dynamic_hd: int = 4):
    base_proc = AutoProcessor.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        dynamic_hd=dynamic_hd,
    )
    try:
        trained_tok = AutoTokenizer.from_pretrained(
            trained_model_path,
            trust_remote_code=True,
            use_fast=False,
        )
        base_proc.tokenizer = trained_tok
        source = "hybrid_base_image_trained_tokenizer"
    except (OSError, ValueError):
        source = "base_image_tokenizer"
    return base_proc, source

def create_dataset_from_annotation(annotation_file: str, img_root: str, processor):
    return MiMiCEXTVQADataset(annotation_file=annotation_file, vis_root=img_root, processor=processor)

def create_dataset_multi(processor):
    parts = []
    dataset_info = {}
    for name in ['basic', 'compare', 'region']:
        img_root = './data/MIMIC-VQA-CoT/images/'
        ann = f'./data/MIMIC-VQA-CoT/annotations/eval/test_{name}_vqa.json'
        dataset = MiMiCEXTVQADataset(annotation_file=ann, vis_root=img_root, processor=processor)
        parts.append(dataset)
        dataset_info[name] = {'path': ann, 'samples': len(dataset)}
    combined = ConcatDataset(parts)
    print("\nDataset Summary:")
    for name, info in dataset_info.items():
        print(f"- {name}: {info['samples']} samples ({info['path']})")
    print(f"Total: {len(combined)} samples")
    return combined

@torch.no_grad()
def evaluate(model, processor, eval_dataset, save_path=None, disable_tqdm=False, max_samples=None, vis_root=''):
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'

    model.eval()
    all_results, all_acc = [], []

    N = len(eval_dataset) if max_samples is None else min(max_samples, len(eval_dataset))
    print(f"Evaluating on {N} samples (total: {len(eval_dataset)})")

    for i in tqdm(range(N), disable=(rank != 0) or disable_tqdm, desc="Evaluation progress"):
        sample = eval_dataset[i]
        q = sample['question']
        gts = sample['answer']
        exp = sample.get('explanation', '')
        img_ids = sample['image_ids']

        images = []
        image_paths = []
        for img_file in img_ids:
            path = os.path.join(vis_root, img_file + '.png')
            img = Image.open(path).convert('RGB')
            
            images.append(img)
            image_paths.append(os.path.abspath(path))

        refs = ''.join([f"<|image_{k+1}|>" for k in range(len(images))])
        prompt = f"{USER}{refs}{q}{END}{ASSIST}"

        inputs = processor(text=prompt, images=images, return_tensors='pt').to(device)

        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        pad_id = processor.tokenizer.pad_token_id

        gen_ids = model.generate(
            **inputs,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=pad_id,
            max_new_tokens=256,
            do_sample=False,
        )

        in_len = inputs['input_ids'].shape[1]
        resp = processor.batch_decode(
            gen_ids[:, in_len:], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0].strip()

        resp_norm = resp.lower()
        ok = any(str(ans).lower() in resp_norm for ans in gts)
        all_acc.append(1 if ok else 0)

        all_results.append({
            "id": i,
            "image_ids": img_ids,
            "question": q,
            "answer": gts,
            "explanation": exp,
            "generated_text": resp,
            "is_correct": ok,
            "image_paths": image_paths,
        })

        if rank == 0 and i < 5:
            print(f"\nSample {i}:")
            print(f"Question: {q}")
            print(f"Ground truth: {gts}")
            print(f"Model answer: {resp}")
            print(f"Correct: {ok}")

    if rank == 0:
        acc = float(np.mean(all_acc)) if all_acc else 0.0
        print(f"\nFinal accuracy: {acc:.4f}")
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w") as f:
                json.dump({"results": all_results, "accuracy": acc, "num_samples": len(all_results)},
                          f, indent=2, ensure_ascii=False)
            print(f"Results saved to {save_path}")
        return acc
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model weights')
    parser.add_argument('--base_model_path', type=str, default='../02_Phi4-MM/base_model/Phi-4-multimodal-instruct', help='Base model path for image processor')
    parser.add_argument('--annotation_file', type=str, default=None, help='Optional: single test json file')
    parser.add_argument('--img_root', type=str, default='./data/images/MIMIC_CXR_JPG', help='Optional: image root directory for annotation_file')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results/')
    parser.add_argument('--dataset_folder', type=str, default='MIMIC', help='Dataset folder')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--use_flash_attention', action='store_true')
    parser.add_argument('--dynamic_hd', type=int, default=4)
    parser.add_argument('--no_tqdm', dest='tqdm', action='store_false')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'

    processor, processor_source = load_processor_no_special_tokens(
        # base_model_path=args.base_model_path,
        base_model_path=args.model_path,
        trained_model_path=args.model_path,
        dynamic_hd=args.dynamic_hd
    )

    if rank == 0:
        print(f"\nProcessor Information:")
        print(f"Processor source: {processor_source}")
        print(f"Tokenizer type: {type(processor.tokenizer).__name__}")
        print(f"Vocabulary size: {len(processor.tokenizer)}")
        print(f"Additional special tokens: {processor.tokenizer.additional_special_tokens}")

    print(f"\nLoading model from: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.use_flash_attention else torch.float32,
        _attn_implementation='flash_attention_2' if args.use_flash_attention else 'sdpa',
        trust_remote_code=True,
    ).to(device)

    if rank == 0:
        cur_vocab = len(processor.tokenizer)
        emb_vocab = model.get_input_embeddings().weight.shape[0]
        print(f"\nVocabulary Validation:")
        print(f"Tokenizer vocabulary: {cur_vocab}")
        print(f"Model embedding rows: {emb_vocab}")
        print(f"Match status: {'Match' if cur_vocab == emb_vocab else 'Mismatch'}")

    if args.annotation_file:
        eval_dataset = create_dataset_from_annotation(args.annotation_file, args.img_root, processor)
        dataset_name = Path(args.annotation_file).stem
        if rank == 0:
            print(f"\nUsing specified test file:")
            print(f"File: {args.annotation_file}")
            print(f"Image root: {args.img_root}")
    else:
        eval_dataset = create_dataset_multi(processor)
        dataset_name = f"{args.dataset_folder}_combined"
        if rank == 0:
            print(f"\nUsing combined test set from folder: {args.dataset_folder}")

    if rank == 0:
        print(f"Evaluation dataset size: {len(eval_dataset)}")

    out_file = f"{Path(args.model_path).name}_{dataset_name}_{processor_source}_results.json"
    out_path = os.path.join(args.output_dir, out_file)

    print("\nStarting evaluation...")
    evaluate(
        model=model,
        processor=processor,
        eval_dataset=eval_dataset,
        save_path=out_path,
        disable_tqdm=not args.tqdm,
        max_samples=args.max_samples,
        vis_root=args.img_root
    )

    if rank == 0:
        print(f"\nEvaluation completed! Results saved to: {out_path}")

if __name__ == "__main__":
    main()