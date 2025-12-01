import os
import json
import torch
import re
import random
from torch.utils.data import Dataset
from transformers import BatchFeature
from PIL import Image

_IGNORE_INDEX = -100
_TRAIN_SIZE = 8000
_EVAL_SIZE = 500
_MAX_TRAINING_LENGTH = 8192



class MiMiCEXTVQADataset(Dataset):
    def __init__(self, annotation_file='', vis_root='', processor=None):
        """
        Initialize the dataset.

        Parameters:
            annotation_file (str): Path to the annotation file containing image IDs and captions.
            vis_root (str): Root directory where images are stored.
        """
        with open(annotation_file, 'r') as file:
            self.annotation = json.load(file)

        self.vis_root = vis_root
        self.processor = processor
       
    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.annotation)

    def __getitem__(self, index):
        """
        Retrieve a sample from the dataset at the specified index.
        """
        ann = self.annotation[index]
        question = ann["question"]
        explanation = ann["explanation"]
        short_answer = ann["answer"]
        if type(short_answer) is str:
            short_answer = [short_answer.lower()]
        img_files = ann["image_ids"]
        images = []
        for img_file in img_files:
            image_path = os.path.join(self.vis_root,  img_file+'.png')
            image = Image.open(image_path).convert('RGB')
            images.append(image)
        image_references = ''.join([f"<|image_{i + 1}|>\n" for i in range(len(images))])
        user_message = {
            'role': 'user',
            'content': f"{image_references}{question}",
        }
        prompt = self.processor.tokenizer.apply_chat_template(
            [user_message], tokenize=False, add_generation_prompt=True
        )
        answer = f'{explanation}<|end|>\n<|endoftext|>'
        inputs = self.processor(prompt, images, return_tensors='pt')
        
        answer_ids = self.processor.tokenizer(answer, return_tensors='pt').input_ids

        
        input_ids = torch.cat([inputs.input_ids, answer_ids], dim=1)
        labels = torch.full_like(input_ids, _IGNORE_INDEX)
        labels[:, -answer_ids.shape[1] :] = answer_ids

        if input_ids.size(1) > _MAX_TRAINING_LENGTH:
            input_ids = input_ids[:, :_MAX_TRAINING_LENGTH]
            labels = labels[:, :_MAX_TRAINING_LENGTH]
            if torch.all(labels == _IGNORE_INDEX).item():
                # workaround to make sure loss compute won't fail
                labels[:, -1] = self.processor.tokenizer.eos_token_id

        return {
            'input_ids': input_ids,
            'labels': labels,
            'input_image_embeds': inputs.input_image_embeds,
            'image_attention_mask': inputs.image_attention_mask,
            'image_sizes': inputs.image_sizes,
            'image_ids': img_files,
            'question': question,
            'answer': short_answer,
            'explanation': explanation,
        }



