
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

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"
OOV_TOKEN = "<oov>"



set_all_seeds(42)


exp_name = 'ocr_mult_retrain'
BATCH_SIZE = 128
NUM_EPOCH = 10
START_EPOCH = 0
ckpt = ''
eval = False
freeze = True
use_ocr = True


train_ocr_path = '../dataloader/ocr_results/ocr_texts_train_words_only.json' if use_ocr else None
val_ocr_path = '../dataloader/ocr_results/ocr_texts_val_words_only.json' if use_ocr else None
train_dataset = VQA_mm_Dataset("../../../../net/projects/ranalab/kh310/vqa", "train", include_q_vector=True,
                               load_ocr=train_ocr_path, use_all_ans=True, subset=False)
qa_dataset_val = VQA_mm_Dataset("../../../../net/projects/ranalab/kh310/vqa", "val", include_q_vector=True,
                                load_ocr=val_ocr_path, use_all_ans=True, subset=False)


VOCAB_SIZE = len(train_dataset.vocab)
ENC_EMB_DIM = 1000
DEC_EMB_DIM = 256
HID_DIM = 256
N_LAYERS = 6
N_HEADS = 4
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5



train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_pad_mm, num_workers=20)
val_dataloader = DataLoader(qa_dataset_val, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_pad_mm, num_workers=20)

device = 'cuda'
transformer = TransformerDecoder(
          word_to_idx=train_dataset.vocab,
          idx_to_word = train_dataset.reverse_vocab,
          input_dim=ENC_EMB_DIM,
          embed_dim=HID_DIM,
          num_heads=N_HEADS,
          num_layers=N_LAYERS,
          max_length=30,
          device = device,
          freeze_encoder = freeze
        )

if ckpt != '':
    transformer.load_state_dict(torch.load(ckpt))

trainer = Trainer(exp_name, transformer, train_dataloader, val_dataloader,
          word_to_idx=train_dataset.vocab,
          idx_to_word = train_dataset.reverse_vocab,
          num_epochs=NUM_EPOCH,
          learning_rate=1e-4,
          device = device,
          print_every=1,
          start_epoch=START_EPOCH
        )

if not eval:
    trainer.train()

else:
    assert ckpt != ''
    transformer.eval()

    def eval(model, model_name, loader, criterion, split='val', dump=''):

        type2text = ['unanswerable', 'yes/no', 'number', 'other']

        all_image_ids = []
        all_questions = []
        all_targets = []
        all_predictions = []
        all_ans_types = []
        all_ans_types_gt = []
        total_loss = 0
        for batch in loader:
            features, questions, answers, anstypes, image_ids, questions_org = batch
            gt_answers = decode_captions(answers.numpy(), train_dataset.reverse_vocab)
            logits = model(features.float().cuda(), questions.cuda(), answers[:,:-1].cuda())
            scores, answer_type = logits
            ans_type_pred = torch.argmax(F.softmax(answer_type, dim=1), 1) # (B,)
            final_captions, final_logits, predicted_answer_type_logits = model.sample(features.float().cuda(), questions.cuda(), max_length=answers.shape[1]-1)
            sample_answers = decode_captions(final_captions, train_dataset.reverse_vocab)
            gt_questions = decode_captions(questions_org.numpy(), train_dataset.reverse_vocab)
            loss, language_loss, answer_type_loss = criterion((final_logits, predicted_answer_type_logits), answers[:, 1:].long().to("cuda"), anstypes.to("cuda"))
            total_loss += loss

            all_image_ids.extend(image_ids)
            all_questions.extend(gt_questions)
            all_targets.extend(gt_answers)
            all_predictions.extend(sample_answers)
            all_ans_types.extend(ans_type_pred)
            all_ans_types_gt.extend(anstypes)

        if dump:
            out = {
                "model_name": model_name,
                "metadata": {
                    "mode": split,
                    "modality": ["multimodal"]
                },
                "data": []
            }
            assert len(all_image_ids) == len(all_questions) == len(all_targets) == len(all_predictions) == len(all_ans_types) == len(all_ans_types_gt)
            for img, q, tg, pred, anstype_pred, anstype in zip(all_image_ids, all_questions, all_targets, all_predictions, all_ans_types, all_ans_types_gt):
                out["data"].append({
                    "image_id": img.replace(".jpg", ""),
                    "question": q,
                    "predicted_answer": pred,
                    "target_answer": tg,
                    "predicted_answer_type": type2text[anstype_pred],
                    "target_answer_type": type2text[anstype]
                })

            with open(dump, 'w') as f:
                json.dump(out, f, indent=4)
            f.close()

        return total_loss / len(loader)


    import torch.nn.functional as F
    def criterion(predictions, answers, answer_type):

          predicted_answers, predicted_answer_type = predictions
          mask = answer_type != 0
          answers = answers.long()
          masked_logit = predicted_answers * mask.unsqueeze(1).broadcast_to(predicted_answers.shape[:2]).reshape(list(predicted_answers.shape[:2]) + [1])
          language_loss = F.cross_entropy(masked_logit.reshape(-1, masked_logit.size(-1)),
                                          answers.reshape(-1), ignore_index=2)
          answer_type_loss = F.cross_entropy(predicted_answer_type, answer_type)
          total_loss = language_loss + answer_type_loss
          return total_loss, language_loss, answer_type_loss

    loss = eval(transformer, exp_name, val_dataloader, criterion, split='val', dump='%s_outputs_val.txt' % exp_name)
    print("%s loss:" % exp_name, loss)
    loss = eval(transformer, exp_name, train_dataloader, criterion, split='train', dump='%s_outputs_train.txt' % exp_name)
    print("%s loss:" % exp_name, loss)