import sys
sys.path.insert(0, '../dataloader')
from dataset import QADataset, collate_fn_pad
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random
import os
import json
import numpy as np

SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"
OOV_TOKEN = "<oov>"

def set_all_seeds(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

def decode_captions(captions, idx_to_word):
    singleton = False
    if captions.ndim == 1:
        singleton = True
        captions = captions[None]
    decoded = []
    N, T = captions.shape
    for i in range(N):
        words = []
        for t in range(T):
            word = idx_to_word[captions[i, t]]
            if word != PAD_TOKEN:
                words.append(word)
            if word == EOS_TOKEN:
                break
        decoded.append(" ".join(words))
    if singleton:
        decoded = decoded[0]
    return decoded

def decode_captions_bert(captions, tokenizer):
    singleton = False
    if captions.ndim == 1:
        singleton = True
        captions = captions[None]
    decoded = []
    N, T = captions.shape
    for i in range(N):
        words = []
        tmp = tokenizer.decode(captions[i], skip_special_tokens=True)
        decoded.append(tmp)
    if singleton:
        decoded = decoded[0]
    return decoded



def eval(model, model_name, loader, criterion, split='val', dump=''):
    all_image_ids = []
    all_questions = []
    all_targets = []
    all_predictions = []
    total_loss = 0
    for batch in loader:
        features, questions, answers, image_ids, questions_vec = batch
      
        gt_answers = decode_captions(answers.numpy(), model.idx_to_word)
        output, logits = model.sample(features.float().cuda(), questions.cuda(), max_length=answers.shape[1]-1)
        sample_answers = decode_captions(output, model.idx_to_word)
        gt_questions = decode_captions(questions_vec.numpy(), model.idx_to_word)

        loss = criterion(logits, answers[:, 1:].long().to("cuda"))
        total_loss += loss

        all_image_ids.extend(image_ids)
        all_questions.extend(gt_questions)
        all_targets.extend(gt_answers)
        all_predictions.extend(sample_answers)
      
    if dump:
        out = {
            "model_name": model_name,
            "metadata": {
                "mode": split,
                "modality": ["multimodal"]
            },
            "data": []
        }
        assert len(all_image_ids) == len(all_questions) == len(all_targets) == len(all_predictions)
        for img, q, tg, pred in zip(all_image_ids, all_questions, all_targets, all_predictions):
            out["data"].append({
                "image_id": img.replace(".jpg", ""),
                "question": q,
                "predicted_answer": pred,
                "target_answer": tg
            })

        with open(dump, 'w') as f:
            json.dump(out, f, indent=4)

    return total_loss / len(loader)
