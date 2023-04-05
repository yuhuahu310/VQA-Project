import torch
from utils import *
from torch.utils.data import DataLoader
from trainer2 import Trainer
from vit_bert_atten_transformer import TransformerDecoder
from matplotlib import pyplot as plt
import torchvision
import random
import numpy as np
import os
import sys
sys.path.insert(0, '../dataloader')
from dataset import VQA_mm2_Dataset, collate_fn_pad_mm2
from vqa import VQA
from transformers import BertTokenizer, BertModel
import argparse

SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"
OOV_TOKEN = "<oov>"

with open("vit_bert_att_HID_DIM_128_outputs_val.txt", 'r') as f:
    res = json.load(f)
for group in res['data']:
    if not group['predicted_answer'] == "unanswerable <eos>":
        print(group)


# train_loss = np.load('hidden64_layer2_train_loss.npy')
# val_loss = np.load('hidden64_layer2_val_loss.npy')
# print(train_loss)
# print(val_loss)

# device = 'cuda'
#
# # ds_path = "../data"  #Vicky
# ds_path = "../../../../net/projects/ranalab/kh310/vqa"
#
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#
# pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
# train_dataset = VQA_mm2_Dataset(ds_path, "train", img_transforms=pretrained_vit_weights.transforms(), tokenizer=tokenizer)
# qa_dataset_val = VQA_mm2_Dataset(ds_path, "val", img_transforms=pretrained_vit_weights.transforms(), tokenizer=tokenizer)
#
# train_dataloader = DataLoader(train_dataset, batch_size=30, shuffle=True, collate_fn=collate_fn_pad_mm2, num_workers=2)
#
# for batch in train_dataloader:
#
#     features, questions, answers = batch[0], batch[1], batch[2]
#
#     q_str = decode_captions_bert(questions.numpy(), tokenizer)
#     print(q_str)
#
#     break