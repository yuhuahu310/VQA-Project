
import torch
from utils import * 
from torch.utils.data import DataLoader
from trainer2 import Trainer
from vit_bert_transformer import TransformerDecoder
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
# print("old_token_size:", len(tokenizer))
# vqa = VQA(annotation_file=os.path.join("../data/Annotations/train.json"))
# added = tokenizer.add_tokens(list(vqa.get_vocab()))
# print(added, "tokens are added to tokenizer")
# vqa = VQA(annotation_file=os.path.join("../data/Annotations/val.json"))
# added = tokenizer.add_tokens(list(vqa.get_vocab()))
# print(added, "tokens are added to tokenizer")
# # add new, random embeddings for the new tokens
# new_token_size = len(tokenizer)
# print("new_token_size:", new_token_size)
pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
train_dataset = VQA_mm2_Dataset("../data", "train", img_transforms=pretrained_vit_weights.transforms(), tokenizer=tokenizer)
qa_dataset_val = VQA_mm2_Dataset("../data", "val", img_transforms=pretrained_vit_weights.transforms(), tokenizer=tokenizer)

VOCAB_SIZE = len(train_dataset.vocab)


train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_pad_mm2, num_workers=64)
val_dataloader = DataLoader(qa_dataset_val, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_pad_mm2, num_workers=64)


def vis_imgs(model, split, exp_name):
    loader = {'train': train_dataloader, 'val': val_dataloader}[split]
    num_imgs = 0 
    for batch in loader:
        features, questions, answers, questions_org, image_ids = batch
      
        gt_captions = decode_captions(answers.numpy(), train_dataset.reverse_vocab)
        output, logits = model.sample(features.float().cuda(), questions.cuda(), max_length=30)
        sample_captions = decode_captions(output, train_dataset.reverse_vocab)
      
        for gt_caption, sample_caption, img in zip(gt_captions, sample_captions, features):
            if img is not None: 
                img = img.swapaxes(0, 1)
                img = img.swapaxes(1, 2)
                plt.imshow(img)            
                plt.title('%s\n%s\nGT:%s' % (split, sample_caption, gt_caption))
                plt.axis('off')
                plt.savefig('plots/'+ '%s_%s_%d.png' % (exp_name, split, num_imgs))
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
        logits = model(features.float().cuda(), questions.cuda(), answers[:,:-1].cuda())
        output, logits = model.sample(features.float().cuda(), questions.cuda(), max_length=answers.shape[1]-1)
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

def train_and_test(param, value):
    exp_name = "vit_bert_mul_%s_%d" % (param, value)
    if param == "HID_DIM":
        transformer = TransformerDecoder(
                  tokenizer,
                  input_dim=ENC_EMB_DIM,
                  embed_dim=value,
                  new_token_size = new_token_size,
                  vocab_size=VOCAB_SIZE,
                  num_heads=4,
                  num_layers=6,
                  max_length=30,
                  device=device
                )
    elif param == "num_layers":
        transformer = TransformerDecoder(
                  tokenizer,
                  input_dim=ENC_EMB_DIM,
                  embed_dim=256,
                  new_token_size = new_token_size,
                  vocab_size=VOCAB_SIZE,
                  num_heads=4,
                  num_layers=value,
                  max_length=30,
                  device=device
                )

    trainer = Trainer(transformer, train_dataloader, val_dataloader,
              word_to_idx=train_dataset.vocab,
              idx_to_word = train_dataset.reverse_vocab,
              num_epochs=20,
              learning_rate=1e-4,
              device = device
            )

    trainer.train()

    torch.save(transformer.state_dict(), "/home/yizhi/VQA-Project/multimodal_baseline/model_%s.pth" % (exp_name))

    print()
    print("finished training, begin evaluation")
    transformer.eval()

    # Plot the training losses.
    print("train loss history:")
    print(trainer.loss_history)
    plt.plot(trainer.loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    os.makedirs('plots', exist_ok=True)
    plt.title('Training loss history')
    plt.savefig('plots/' + exp_name + '_train_loss_out.png')
    plt.close()

    # Plot the eval losses.
    print("eval loss history:")
    print(trainer.val_loss_history)
    plt.plot(trainer.val_loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    os.makedirs('plots', exist_ok=True)
    plt.title('Val loss history')
    plt.savefig('plots/' + exp_name + '_val_loss_out.png')

    import torch.nn.functional as F
    def criterion(predictions, labels):
        labels = labels.long()
        return F.cross_entropy(predictions.reshape(-1, predictions.size(-1)), labels.reshape(-1), ignore_index=2)


    # loss: tensor(6.2143, device='cuda:0')
    # loss: tensor(0.1230, device='cuda:0')
    loss = eval(transformer, exp_name, val_dataloader, criterion, split='val', dump='%s_outputs_val.txt' % exp_name)
    print("%s loss:" % exp_name, loss)
    loss = eval(transformer, exp_name, train_dataloader, criterion, split='train', dump='%s_outputs_train.txt' % exp_name)
    print("%s loss:" % exp_name, loss)
    print()

  


# 20
# HID_DIM 64 128 256
# num_layers 2 4 6
# # vis_imgs('train')
# # vis_imgs('val')

params = {"HID_DIM": [64, 128, 256], "num_layers": [2, 4, 6]}
for param, values in params.items():
  for value in values:
    train_and_test(param, value)