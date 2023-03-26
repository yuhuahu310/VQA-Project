import sys
sys.path.insert(0, '../../dataloader')
from dataset import QADataset, collate_fn_pad
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random


# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_size, vocab_size, num_layers, dropout):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers)

    def forward(self, sentence):
        embs = self.word_embeddings(sentence)
        _, (hidden, cell) = self.lstm(embs)
        return hidden, cell


class DecoderLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_size, vocab_size, num_layers, dropout):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embs = self.word_embeddings(input)
        output, (hidden, cell) = self.lstm(embs, (hidden, cell))
        output = self.linear(output.squeeze(0))
        return output, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder: EncoderLSTM, decoder: DecoderLSTM, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.hidden_size == decoder.hidden_size
        assert encoder.num_layers == decoder.num_layers

    def forward(self, input, target, teacher_forcing_ratio=0.5):
        '''
        :param target of size [target_len, batch_size]
        '''
        batch_size = target.shape[1]
        target_len = target.shape[0]
        outputs = torch.zeros(target_len, batch_size, self.decoder.vocab_size).to(self.device)
        encoder_hidden, encoder_cell = self.encoder(input)

        decoder_input = target[0]
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell

        loss = 0
        for i in range(1, target_len):
            decoder_output, decoder_hidden, decoder_cell = self.decoder(decoder_input, decoder_hidden, decoder_cell)
            # loss += nn.functional.cross_entropy(decoder_output, target[i])
            outputs[i] = decoder_output
            teacher_forcing = random.random() < teacher_forcing_ratio
            decoder_input = target[i] if teacher_forcing else decoder_output.argmax(dim=1)
            
        return outputs
    
def build_LSTM(encoder_emb_dim, decoder_emb_dim, hidden_dim, vocab_size, n_layers, encoder_dropout, decoder_dropout):
    enc = EncoderLSTM(encoder_emb_dim, hidden_dim, vocab_size, n_layers, encoder_dropout)
    dec = DecoderLSTM(decoder_emb_dim, hidden_dim, vocab_size, n_layers, decoder_dropout)

    model = Seq2Seq(enc, dec, device).to(device)

    def init_weights(m):
        for _, param in m.named_parameters():
            if len(param.data.shape)<2:
                nn.init.uniform_(param.data, -0.08, 0.08)
            else:
                nn.init.xavier_uniform_(param.data)
            
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters())
    TRG_PAD_IDX = 2
    criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

    return model, optimizer, criterion

def print_qa_example(questions, answers, outputs, dataset, batch_idx, len_dataloader, mode):
    curr_bs = questions.shape[1]
    num_samples = 10
    if curr_bs >= num_samples:
        sample_questions = questions[:, :num_samples].T
        sample_targets = answers[:, :num_samples].T
        sample_outputs = outputs[:, :num_samples, :].transpose(0, 1).argmax(dim=-1)
        print(f'{mode} Batch [{batch_idx + 1}/{len_dataloader}]')
        for i in range(num_samples):
            q = dataset.idx_seq_2_sent(sample_questions[i].cpu().numpy())
            a = dataset.idx_seq_2_sent(sample_outputs[i].cpu().numpy())
            tg = dataset.idx_seq_2_sent(sample_targets[i].cpu().numpy())
            print('[Question]: {}\n\t[Answer]: {}\n\t[Target]: {}'.format(q, a, tg))

def train(model, dataloader, optimizer, criterion, dataset):
    total_loss = 0
    model.train()
    for batch_idx, (questions, answers) in enumerate(dataloader):
        # Move data to GPU if available
        questions = questions.to(device)
        answers = answers.to(device)
        # print(questions.T)
        # print()
        # print(answers.T)
        # 1/0

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(questions, answers)

        # Calculate loss
        output_dim = outputs.shape[-1]
        logits = outputs[1:].view(-1, output_dim)
        labels = answers[1:].view(-1).type(torch.LongTensor).to(device)
        # print(logits.shape, labels.shape, type(logits), type(labels))
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Print loss every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f'Batch [{batch_idx + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')
            print_qa_example(questions, answers, outputs, dataset, batch_idx, len(dataloader), 'Train')

        # Accumulate total loss for the epoch
        total_loss += loss.item()
    return total_loss / len(dataloader)


def eval(model, dataloader, criterion, dataset):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (questions, answers) in enumerate(dataloader):
            # Move data to GPU if available
            questions = questions.to(device)
            answers = answers.to(device)

            outputs = model(questions, answers)

            # Calculate loss
            output_dim = outputs.shape[-1]
            logits = outputs[1:].view(-1, output_dim)
            labels = answers[1:].view(-1).type(torch.LongTensor).to(device)
            loss = criterion(logits, labels)
            total_loss += loss

            if (batch_idx+1) % 60 == 0:
                print_qa_example(questions, answers, outputs, dataset, batch_idx, len(dataloader), 'Eval')
    
    return total_loss / len(dataloader)


if __name__ == "__main__":

    qa_dataset_train = QADataset("../../data", "train")
    qa_dataset_val = QADataset("../../data", "val")

    VOCAB_SIZE = len(qa_dataset_train.vocab)
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    BATCH_SIZE = 64
    NUM_EPOCH = 100

    model, optimizer, criterion = build_LSTM(ENC_EMB_DIM, DEC_EMB_DIM, HID_DIM, VOCAB_SIZE, N_LAYERS, 
        ENC_DROPOUT, DEC_DROPOUT)

    train_loader = DataLoader(qa_dataset_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_pad)
    val_loader = DataLoader(qa_dataset_val, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_pad)

    for epoch in range(NUM_EPOCH):
        train_loss = train(model, train_loader, optimizer, criterion, qa_dataset_train)
        val_loss = eval(model, val_loader, criterion, qa_dataset_val)
        print('Epoch [{}/{}], Average Training Loss: {:.4f}'.format(epoch + 1, NUM_EPOCH, train_loss))
        print('Epoch [{}/{}], Average Validation Loss: {:.4f}'.format(epoch + 1, NUM_EPOCH, val_loss))