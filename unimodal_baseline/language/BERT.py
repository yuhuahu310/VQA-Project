import sys
# sys.path.insert(0, '../../dataloader')
# from dataset import VQADataset
# import torch
# from transformers import T5Tokenizer, T5ForConditionalGeneration

import os
import random
import torch
from torch.utils.data import DataLoader, random_split
from transformers import BertForSequenceClassification, BertTokenizer
from tqdm import tqdm
sys.path.insert(0, '../../dataloader')
from dataset import QADataset, collate_fn_pad

# class Bert_Base:
#     def __init__(self):
#         self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
#         self.model = T5ForConditionalGeneration.from_pretrained("t5-base")
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)
#     def forward(self, sentence, answers):
#         input_ids = tokenizer(sentence, return_tensors="pt").input_ids
#         loss = model(input_ids=input_ids, labels=labels).loss
#         return loss
#     def generate_ans(self, question):
#         input_ids = tokenizer(sentence, return_tensors="pt").input_ids
#         outputs = self.model.generate(input_ids)
#         ans = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return ans
        
# def train(bert_base, dataloader, optimizer, criterion, dataset):
#     total_loss = 0
#     for batch_idx, (questions, answers) in enumerate(dataloader):
#         # Move data to GPU if available
#         questions = questions.to(device)
#         answers = answers.to(device)

#         # Zero the gradients
#         optimizer.zero_grad()

#         # Forward pass

# # the forward function automatically creates the correct decoder_input_ids
# loss = model(input_ids=input_ids, labels=labels).loss
# loss.item()

if __name__ == "__main__":
    
    # Set random seed for reproducibility
    random_seed = 42
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    num_epochs = 10
    batch_size = 64
    # learning_rate = 2e-5

    # Load BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load dataset
    data_dir = "../../data"
    train_dataset = QADataset(data_dir, 'train')
    val_dataset = QADataset(data_dir, 'val')

    # # Split train dataset into train and validation
    # train_size = int(0.9 * len(train_dataset))
    # val_size = len(train_dataset) - train_size
    # train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_pad)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_pad)

    # Load BERT model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased') #, num_labels=len(train_dataset.vocab))

    # Move model to device
    model.to(device)

    # Set optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()

    # Train model
    for epoch in range(num_epochs):
        # Training loop
        model.train()
        train_loss = 0
        train_acc = 0
        for questions, answers in tqdm(train_loader):
            # Move data to device
            questions = questions.T.to(device)
            answers = answers.T.to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            print(questions.shape, answers.shape)
            outputs = model(questions, labels=answers)
            loss, logits = outputs[:2]

            # Backward pass
            loss.backward()
            optimizer.step()

            # Compute accuracy
            preds = torch.argmax(logits, axis=1)
            acc = torch.mean((preds == answers).float())

            # Update metrics
            train_loss += loss.item()
            train_acc += acc.item()

        # Validation loop
        model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for questions, answers in tqdm(val_loader):
                # Move data to device
                questions = questions.to(device)
                answers = answers.to(device)

                # Forward pass
                outputs = model(questions, labels=answers)
                loss, logits = outputs[:2]

                # Compute accuracy
                preds = torch.argmax(logits, axis=1)
                acc = torch.mean((preds == answers).float())

                # Update metrics
                val_loss += loss.item()
                val_acc += acc.item()

        # Compute average loss and accuracy
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        # Print metrics
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')