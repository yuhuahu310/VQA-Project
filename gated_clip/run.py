
import torch
from utils import * 
from torch.utils.data import DataLoader
from trainer import Trainer
from transformer import TransformerDecoder
from matplotlib import pyplot as plt
import random
import numpy as np
import os
from datetime import datetime
import argparse
import sys
sys.path.insert(0, '../dataloader')
sys.path.insert(0, '../quality_detector')
from dataset import VQA_mm_Dataset, collate_fn_pad_mm, SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, OOV_TOKEN, OCR_TOKEN
from quality_detector import QualityDetector, load_new_model
# from qd_dataset import QDDataset

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import torch.nn.functional as F


set_all_seeds(42)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_ocr', action='store_true', default=False, help='use OCR or not')
    parser.add_argument('--freeze_encoders_until_epoch', type=int, default=-1, help='freeze clip encoders until specified epoch, -1 for always freeze, 0 for never freeze')
    parser.add_argument('--resume_ckpt', type=str, default='', help='path to model ckpt to resume training from')
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    parser.add_argument('--exp_name', type=str, default=current_time, help='experiment name')
    parser.add_argument('--eval', action='store_true', default=False, help='eval only')
    parser.add_argument('--epochs', type=int, default=50, help='num epochs to train, not counting resumed epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--data_dir', type=str, default='../data', help='path to vqa data dir')
    parser.add_argument('--subset', action='store_true', default=False, help='use 0.1 subset of full dataset')
    parser.add_argument('--ocr_results_dir', type=str, default='../dataloader/ocr_results', help='path to ocr results json dir')
    parser.add_argument('--fusion', type=str, default='mult', choices=['mult', 'cross_attn'], help='use mult or attn as fusion technique')
    parser.add_argument('--use_quality', action='store_true', default=False, help='use quality detector or not')
    parser.add_argument('--quality_model_path', type=str, default='../quality_detector/QD-resnext50_32x4d-bce-epoch_14-trainloss_0.3725-valloss_0.4276.pth', help='path to quality detector model')

    args = parser.parse_args()

    if args.eval:
        if args.resume_ckpt == '':
            raise ValueError('must provide a model ckpt for eval')

    # "../../../../net/projects/ranalab/kh310/vqa"
    train_ocr_path = f'{args.ocr_results_dir}/ocr_texts_train_words_only.json' if args.use_ocr else None
    train_dataset = VQA_mm_Dataset(args.data_dir, "train", include_q_vector=True, 
        load_ocr=train_ocr_path, use_all_ans=True, subset=args.subset)
    
    val_ocr_path = f'{args.ocr_results_dir}/ocr_texts_val_words_only.json' if args.use_ocr else None
    qa_dataset_val = VQA_mm_Dataset(args.data_dir, "val", include_q_vector=True, 
        load_ocr=val_ocr_path, use_all_ans=True, subset=args.subset)

    VOCAB_SIZE = len(train_dataset.vocab)
    ENC_EMB_DIM = 1000
    DEC_EMB_DIM = 256
    HID_DIM = 256
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    BATCH_SIZE = args.batch_size
    NUM_EPOCH = 100

    NUM_DATALOADER_WORKERS = 20

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_pad_mm, num_workers=NUM_DATALOADER_WORKERS)
    val_dataloader = DataLoader(qa_dataset_val, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_pad_mm, num_workers=NUM_DATALOADER_WORKERS)

    device = 'cuda'
    exp_name = args.exp_name #'gated_clip_1'

    quality_detector = None
    if args.use_quality:
        quality_detector = load_new_model(args.quality_model_path).to(device)
    
    transformer = TransformerDecoder(
            word_to_idx=train_dataset.vocab,
            idx_to_word = train_dataset.reverse_vocab,
            input_dim=ENC_EMB_DIM,
            embed_dim=HID_DIM,
            num_heads=4,
            num_layers=6,
            max_length=30,
            device = device,
            fusion = args.fusion,
            quality_detector=quality_detector
        )

    start_epoch = 0
    if args.resume_ckpt != '':
        transformer.load_state_dict(torch.load(args.resume_ckpt)['model'])
        start_epoch = torch.load(args.resume_ckpt)['epoch'] + 1
        print(f'resumed training from {args.resume_ckpt}')
    
    if not args.eval:
        trainer = Trainer(transformer, train_dataloader, val_dataloader,
                word_to_idx=train_dataset.vocab,
                idx_to_word = train_dataset.reverse_vocab,
                num_epochs=args.epochs,
                learning_rate=1e-3,
                weight_decay=1e-2,
                device = device,
                freeze_encoders_until_epoch=args.freeze_encoders_until_epoch,
                exp_name = exp_name,
                start_epoch=start_epoch,
                print_every=1
                )

        trainer.train()

    # # torch.save(transformer.state_dict(), "/home/yizhi/VQA-Project/gated_clip/model.pth")
    #
    
    # transformer.load_state_dict(torch.load(f"./model_{exp_name}.pth"))
    print("Start Evaluation")
    transformer.eval()
    #
    #
    #
    # # Plot the training losses.
    # # plt.plot(trainer.loss_history)
    # # plt.xlabel('Iteration')
    # # plt.ylabel('Loss')
    # # os.makedirs('plots', exist_ok=True)
    # # plt.title('Training loss history')
    # # plt.savefig('plots/' + exp_name + '_loss_out.png')
    #
    # # plt.close()
    # # plt.plot(trainer.val_loss_history)
    # # plt.xlabel('Iteration')
    # # plt.ylabel('Loss')
    # # os.makedirs('plots', exist_ok=True)
    # # plt.title('Validaion loss history')
    # # plt.savefig('plots/' + exp_name + '_val_loss_out.png')
    #
    #
    # # vis_imgs('train')
    # # vis_imgs('val')
    #
    #

    loss = eval(transformer, exp_name, val_dataloader, criterion, split='val', dump='%s_outputs_val.txt' % exp_name)
    print("%s loss:" % exp_name, loss)
    loss = eval(transformer, exp_name, train_dataloader, criterion, split='train', dump='%s_outputs_train.txt' % exp_name)
    print("%s loss:" % exp_name, loss)