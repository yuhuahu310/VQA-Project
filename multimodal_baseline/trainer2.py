import numpy as np
import torch.nn.functional as F

import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
import torch


class Trainer(object):

    def __init__(self, model, train_dataloader, val_dataloader, word_to_idx, idx_to_word, learning_rate = 0.001, num_epochs = 10, print_every = 5, verbose = True, device = 'cuda'):
      
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
        self.optim = torch.optim.Adam(self.model.parameters(), self.learning_rate)
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word

    def loss(self, predictions, labels):
        labels = labels.long()
        return F.cross_entropy(predictions.reshape(-1, predictions.size(-1)), labels.reshape(-1), ignore_index=2)
    
    def val(self):
        """
        Run validation to compute loss and BLEU-4 score.
        """
        self.model.eval()
        val_loss = 0
        num_batches = 0
        for batch in self.val_dataloader:
            features, questions, answers = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
            logits = self.model(features, questions, answers[:,:-1])
            loss = self.loss(logits, answers[:, 1:])
            val_loss += loss.detach().cpu().numpy()
            num_batches += 1

        self.model.train()
        return val_loss/num_batches
    

    def train(self):
        """
        Run optimization to train the model.
        """
        for i in range(self.num_epochs):
            epoch_loss = 0
            num_batches = 0
            b = 0
            for batch in self.train_dataloader:
                features, questions, answers = batch[0].to(self.device), batch[1].to(self.device), batch[2].to(self.device)
                logits = self.model(features, questions, answers[:,:-1])
                loss = self.loss(logits, answers[:, 1:])
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                
                epoch_loss += loss.detach().cpu().numpy()
                num_batches += 1
                
            self.loss_history.append(epoch_loss/num_batches)
            if self.verbose and (i +1) % self.print_every == 0:
                self.val_loss_history.append(self.val())
                print( "(epoch %d / %d) loss: %f" % (i+1 , self.num_epochs, self.loss_history[-1]))    
                
            
            # if (i +1) % self.print_every == 0:
            #     torch.save(self.model.state_dict(), "/home/yizhi/VQA-Project/unimodal_baseline/vision/model_%d.pth" % i)