import numpy as np
import torch.nn.functional as F
import os
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from tqdm import tqdm


class Trainer(object):


    def __init__(self, model, train_dataloader, val_dataloader, word_to_idx, idx_to_word, exp_name,
                    learning_rate = 1e-3, weight_decay=1e-2, num_epochs = 10, print_every = 5, verbose = True, device = 'cuda',
                    freeze_encoders_until_epoch=-1, start_epoch=0):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.print_every = print_every
        self.verbose = verbose 
        self.loss_history = []
        self.val_loss_history = []
        self.device = device
        self.optim = AdamW(self.model.parameters(), self.learning_rate, weight_decay=weight_decay)
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.exp_name = exp_name
        self.writer = SummaryWriter(log_dir=f'./runs/{self.exp_name}')
        self.freeze_encoders_until_epoch = freeze_encoders_until_epoch
        self.start_epoch = start_epoch

    # answer_type: (B, 1)
    def loss(self, predictions, answers, answer_type):
        predicted_answers, predicted_answer_type = predictions
        mask = answer_type != 0
        answers = answers.long()
        masked_logit = predicted_answers * mask.unsqueeze(1).broadcast_to(predicted_answers.shape[:2]).reshape(list(predicted_answers.shape[:2]) + [1])
        language_loss = F.cross_entropy(masked_logit.reshape(-1, masked_logit.size(-1)), 
                                        answers.reshape(-1), ignore_index=2)
        answer_type_loss = F.cross_entropy(predicted_answer_type, answer_type)
        total_loss = language_loss + answer_type_loss
        return language_loss, answer_type_loss, total_loss
    
    def val(self):
        """
        Run validation to compute loss and BLEU-4 score.
        """
        self.model.eval()
        val_loss = 0
        val_loss_lang = 0
        val_loss_type = 0
        num_batches = 0
        for batch in self.val_dataloader:
            features, questions, answers, answer_type = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device), batch[3].to(self.device)
            logits = self.model(features, questions, answers[:, :-1])
            language_loss, answer_type_loss, total_loss = self.loss(logits, answers[:, 1:], answer_type)

            val_loss += total_loss.detach().cpu().numpy()
            val_loss_lang += language_loss.detach().cpu().numpy()
            val_loss_type += answer_type_loss.detach().cpu().numpy()
            num_batches += 1

        self.model.train()
        return val_loss/num_batches, val_loss_lang/num_batches, val_loss_type/num_batches
    

    def train(self):
        """
        Run optimization to train the model.
        """
        for i in range(self.start_epoch, self.num_epochs):
            epoch_loss = 0
            epoch_loss_lang = 0
            epoch_loss_type = 0
            num_batches = 0
            b = 0
            print(f'epoch: {i}/{self.num_epochs}')
            for j, batch in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader)):
                # print(f'epoch: {i}, batch: {j}/{len(self.train_dataloader)}')
                features, questions, answers, answer_type = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device), batch[3].to(self.device)
                freeze_encoders = self.freeze_encoders_until_epoch == -1 or i < self.freeze_encoders_until_epoch
                logits = self.model(features, questions, answers[:,:-1], freeze_encoders=freeze_encoders)
                language_loss, answer_type_loss, total_loss = self.loss(logits, answers[:, 1:], answer_type)
                self.optim.zero_grad()
                total_loss.backward()
                self.optim.step()
                
                epoch_loss += total_loss.detach().cpu().numpy()
                epoch_loss_lang += language_loss.detach().cpu().numpy()
                epoch_loss_type += answer_type_loss.detach().cpu().numpy()

                num_batches += 1

            # self.loss_history.append(epoch_loss/num_batches)
            # self.loss_history_lang.append(epoch_loss_lang/num_batches)
            # self.loss_history_type.append(epoch_loss_type / num_batches)

            train_loss_total = epoch_loss/num_batches
            self.writer.add_scalars('train_loss', {
                'total': train_loss_total,
                'lang-loss': epoch_loss_lang/num_batches,
                'type-loss': epoch_loss_type/num_batches
            }, i)

            if self.verbose and ((i +1) % self.print_every == 0 or i == self.num_epochs-1):
                # self.val_loss_history.append(self.val())
                # print( "(epoch %d / %d) loss: %f" % (i+1 , self.num_epochs, self.loss_history[-1]))
                val_loss, val_loss_lang, val_loss_type = self.val()
                self.writer.add_scalars('val_loss', {
                    'total': val_loss,
                    'lang-loss': val_loss_lang,
                    'type-loss': val_loss_type
                }, i)
                ckpt_dir = 'ckpt'
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save({
                    'epoch': i,
                    'model': self.model.state_dict()
                }, f"{ckpt_dir}/model_{self.exp_name}.pth")

            # with open('train_loss.npy', 'wb') as f:
            #     np.save(f, self.loss_history)
            # with open('val_loss.npy', 'wb') as f:
            #     np.save(f, self.val_loss_history)
            # if (i +1) % self.print_every == 0:
            #     torch.save(self.model.state_dict(), "/home/yizhi/VQA-Project/unimodal_baseline/vision/model_%d.pth" % i)