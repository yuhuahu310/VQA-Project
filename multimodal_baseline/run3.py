
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

SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"
OOV_TOKEN = "<oov>"


set_all_seeds(42)
exp_name = 'vit+bert+transformer'
ENC_EMB_DIM = 1000
DEC_EMB_DIM = 256
HID_DIM = 256
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
BATCH_SIZE = 64
NUM_EPOCH = 100

device = 'cuda'

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print("old_token_size:", len(tokenizer))
vqa = VQA(annotation_file=os.path.join("../data/Annotations/train.json"))
added = tokenizer.add_tokens(list(vqa.get_vocab()))
print(added, "tokens are added to tokenizer")
vqa = VQA(annotation_file=os.path.join("../data/Annotations/val.json"))
added = tokenizer.add_tokens(list(vqa.get_vocab()))
print(added, "tokens are added to tokenizer")
# add new, random embeddings for the new tokens
new_token_size = len(tokenizer)
print("new_token_size:", new_token_size)
pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
train_dataset = VQA_mm2_Dataset("../data", "train", img_transforms=pretrained_vit_weights.transforms(), tokenizer=tokenizer)
qa_dataset_val = VQA_mm2_Dataset("../data", "val", img_transforms=pretrained_vit_weights.transforms(), tokenizer=tokenizer)

VOCAB_SIZE = len(train_dataset.vocab)


train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_pad_mm2, num_workers=1)
val_dataloader = DataLoader(qa_dataset_val, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_pad_mm2, num_workers=1)

transformer = TransformerDecoder(
          tokenizer,
          input_dim=ENC_EMB_DIM,
          embed_dim=HID_DIM,
          new_token_size = new_token_size,
          vocab_size=VOCAB_SIZE,
          num_heads=4,
          num_layers=6,
          max_length=30,
          device=device
        )

trainer = Trainer(transformer, train_dataloader, val_dataloader,
          word_to_idx=train_dataset.vocab,
          idx_to_word = train_dataset.reverse_vocab,
          num_epochs=100,
          learning_rate=1e-4,
          device = device
        )

trainer.train()

torch.save(transformer.state_dict(), "/home/yizhi/VQA-Project/multimodal_baseline/model.pth")

# transformer.load_state_dict(torch.load("/home/yizhi/VQA-Project/multimodal_baseline/vision/model_99.pth"))
# transformer.eval()


# Plot the training losses.
plt.plot(trainer.loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
os.makedirs('plots', exist_ok=True)
plt.title('Training loss history')
plt.savefig('plots/' + exp_name + '_loss_out.png')

# Plot the eval losses.
plt.plot(trainer.val_loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
os.makedirs('plots', exist_ok=True)
plt.title('Val loss history')
plt.savefig('plots/' + exp_name + '_loss_out.png')




# vis_imgs('train')
# vis_imgs('val')