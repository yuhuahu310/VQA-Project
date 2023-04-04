
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


parser = argparse.ArgumentParser()
parser.add_argument('--hidden', required=True)
parser.add_argument('--layers', required=True)
parser.add_argument('--exp', required=True)
parser.add_argument('--epoch', required=True)
args = parser.parse_args()

print(f"Hidden Dimension: {args.hidden}")
print(f"Number of Layers: {args.layers}")
print(f"Experiment Name: {args.exp}")
print(f"Epoch: {args.epoch}")

set_all_seeds(42)
exp_name = args.exp
ENC_EMB_DIM = 1000
DEC_EMB_DIM = 256
HID_DIM = args.hidden
N_LAYERS = args.layers
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
BATCH_SIZE = 64
NUM_EPOCH = args.epoch

device = 'cuda'

# ds_path = "../data"  #Vicky
ds_path = "../../../../net/projects/ranalab/kh310/vqa"

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print("old_token_size:", len(tokenizer))

pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
train_dataset = VQA_mm2_Dataset(ds_path, "train", img_transforms=pretrained_vit_weights.transforms(), tokenizer=tokenizer)
qa_dataset_val = VQA_mm2_Dataset(ds_path, "val", img_transforms=pretrained_vit_weights.transforms(), tokenizer=tokenizer)

VOCAB_SIZE = len(train_dataset.vocab)


train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_pad_mm2, num_workers=20)
val_dataloader = DataLoader(qa_dataset_val, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_pad_mm2, num_workers=20)

transformer = TransformerDecoder(
          tokenizer,
          input_dim=ENC_EMB_DIM,
          embed_dim=HID_DIM,
          new_token_size = new_token_size,
          vocab_size=VOCAB_SIZE,
          num_heads=4,
          num_layers=N_LAYERS,
          max_length=30,
          device=device
        )

trainer = Trainer(exp_name, transformer, train_dataloader, val_dataloader,
          word_to_idx=train_dataset.vocab,
          idx_to_word = train_dataset.reverse_vocab,
          num_epochs=NUM_EPOCH,
          learning_rate=1e-4,
          device = device
        )

trainer.train()

# torch.save(transformer.state_dict(), "./model.pth")

# transformer.load_state_dict(torch.load("/home/yizhi/VQA-Project/multimodal_baseline/vision/model_99.pth"))
# transformer.eval()


# # Plot the training losses.
# plt.plot(trainer.loss_history)
# plt.xlabel('Iteration')
# plt.ylabel('Loss')
# os.makedirs('plots', exist_ok=True)
# plt.title('Training loss history')
# plt.savefig('plots/' + exp_name + '_loss_out.png')
#
# # Plot the eval losses.
# plt.plot(trainer.val_loss_history)
# plt.xlabel('Iteration')
# plt.ylabel('Loss')
# os.makedirs('plots', exist_ok=True)
# plt.title('Val loss history')
# plt.savefig('plots/' + exp_name + '_loss_out.png')




# vis_imgs('train')
# vis_imgs('val')