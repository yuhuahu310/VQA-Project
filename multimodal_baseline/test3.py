
import torch
from utils import * 
from torch.utils.data import DataLoader
from trainer2 import Trainer
from vit_bert_atten_transformer import TransformerDecoder
from matplotlib import pyplot as plt
import random
import numpy as np
import sys
sys.path.insert(0, '../dataloader')
from dataset import VQA_mm2_Dataset, collate_fn_pad_mm2
from vqa import VQA
from transformers import BertTokenizer, BertModel
import torchvision

set_all_seeds(42)

# config
ds_path = "../../../../net/projects/ranalab/kh310/vqa"
model_path = "./model_hidden128_layer4.pth"
ENC_EMB_DIM = 1000
DEC_EMB_DIM = 256
HID_DIM = 128
N_LAYERS = 4
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
BATCH_SIZE = 16
N_WORKERS = 20


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
new_token_size = len(tokenizer)
pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
train_dataset = VQA_mm2_Dataset(ds_path, "train", img_transforms=pretrained_vit_weights.transforms(), tokenizer=tokenizer)
qa_dataset_val = VQA_mm2_Dataset(ds_path, "val", img_transforms=pretrained_vit_weights.transforms(), tokenizer=tokenizer)

VOCAB_SIZE = len(train_dataset.vocab)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_pad_mm2, num_workers=N_WORKERS)
val_dataloader = DataLoader(qa_dataset_val, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_pad_mm2, num_workers=N_WORKERS)
device = 'cuda'
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

transformer.load_state_dict(torch.load(model_path))


def vis_imgs(split, model_name):
    loader = {'train': train_dataloader, 'val': val_dataloader}[split]
    num_imgs = 0 
    for batch in loader:
        features, questions, answers, image_ids = batch
      
        gt_captions = decode_captions_bert(answers.numpy(), tokenizer)
        sample_captions = transformer.sample(features.float().cuda(), questions.cuda(), max_length=30)
        sample_captions = decode_captions_bert(sample_captions, tokenizer)
      
        for gt_caption, sample_caption, img in zip(gt_captions, sample_captions, features):
            if img is not None: 
                img = img.swapaxes(0, 1)
                img = img.swapaxes(1, 2)
                plt.imshow(img)            
                plt.title('%s\n%s\nGT:%s' % (split, sample_caption, gt_caption))
                plt.axis('off')
                plt.savefig('plots_%s/' % model_name + exp_name + '_%s_%d.png' % (split, num_imgs))
                num_imgs += 1
                if num_imgs >= 16: break
        return


def eval(model, model_name, loader, criterion, split='val', dump=''):
    all_image_ids = []
    all_questions = []
    all_targets = []
    all_predictions = []
    total_loss = 0
    for batch in loader:
        features, questions, answers, questions_org, image_ids = batch

        gt_answers = decode_captions(answers.numpy(), train_dataset.reverse_vocab)
        logits = model(features.float().cuda(), questions.cuda(), answers[:, :-1].cuda())
        output, logits = model.sample(features.float().cuda(), questions.cuda(), max_length=answers.shape[1] - 1)
        sample_answers = decode_captions(output, train_dataset.reverse_vocab)
        gt_questions = decode_captions(questions_org.numpy(), train_dataset.reverse_vocab)

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

# vis_imgs('train', 'clip')
# vis_imgs('val', 'clip')
import torch.nn.functional as F
def criterion(predictions, labels):
    labels = labels.long()
    return F.cross_entropy(predictions.reshape(-1, predictions.size(-1)), labels.reshape(-1), ignore_index=2)


val_loss = eval(transformer, 'vit_bert_attn', val_dataloader, criterion, split='val', dump='vit_bert_attn_outputs_val.txt')
print("val loss:", val_loss)
train_loss = eval(transformer, 'vit_bert_attn', train_dataloader, criterion, split='train', dump='vit_bert_attn_outputs_train.txt')
print("train loss:", train_loss)
