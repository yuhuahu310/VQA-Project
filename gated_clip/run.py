
import torch
from utils import * 
from torch.utils.data import DataLoader
from trainer import Trainer
from transformer import TransformerDecoder
from matplotlib import pyplot as plt
import random
import numpy as np
import os
import sys
sys.path.insert(0, '../dataloader')
from dataset import VQA_mm_Dataset, collate_fn_pad_mm


SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"
OOV_TOKEN = "<oov>"


set_all_seeds(42)
exp_name = 'case1'

train_dataset = VQA_mm_Dataset("../../../../net/projects/ranalab/kh310/vqa", "train", subset=True)
qa_dataset_val = VQA_mm_Dataset("../../../../net/projects/ranalab/kh310/vqa", "val", subset=True)

VOCAB_SIZE = len(train_dataset.vocab)
ENC_EMB_DIM = 1000
DEC_EMB_DIM = 256
HID_DIM = 256
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
BATCH_SIZE = 128
NUM_EPOCH = 100


train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_pad_mm, num_workers=20)
val_dataloader = DataLoader(qa_dataset_val, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_pad_mm, num_workers=20)

device = 'cuda'
transformer = TransformerDecoder(
          word_to_idx=train_dataset.vocab,
          idx_to_word = train_dataset.reverse_vocab,
          input_dim=ENC_EMB_DIM,
          embed_dim=HID_DIM,
          num_heads=4,
          num_layers=6,
          max_length=30,
          device = device
        )

trainer = Trainer(transformer, train_dataloader, val_dataloader,
          word_to_idx=train_dataset.vocab,
          idx_to_word = train_dataset.reverse_vocab,
          num_epochs=100,
          learning_rate=1e-4,
          device = device
        )

trainer.train()

# torch.save(transformer.state_dict(), "/home/yizhi/VQA-Project/gated_clip/model.pth")

# transformer.load_state_dict(torch.load("/home/yizhi/VQA-Project/multimodal_baseline/vision/model_99.pth"))
# transformer.eval()


# # Plot the training losses.
# plt.plot(trainer.loss_history)
# plt.xlabel('Iteration')
# plt.ylabel('Loss')
# os.makedirs('plots', exist_ok=True)
# plt.title('Training loss history')
# plt.savefig('plots/' + exp_name + '_loss_out.png')




# vis_imgs('train')
# vis_imgs('val')