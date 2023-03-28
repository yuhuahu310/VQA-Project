import sys
sys.path.insert(0, '../../dataloader')
from dataset import VQADataset
from dataset import collate_fn_pad_image
from ResNet import ResNet18
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch


# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, dataloader, optimizer, criterion, dataset):
    total_loss = 0
    model.to(device)
    model.train()
    for batch_idx, (questions, answers, image) in enumerate(dataloader):
        # Move data to GPU if available
        image = image.to(device).float()
        questions = questions.to(device)
        answers = answers.to(device)


        print('image shape:', image.shape) # (batch, 3, 224, 224)

        optimizer.zero_grad()
        img_embedding = model(image)
        print('embedding shape:', img_embedding.shape) # (batch, 256)

        break

        # # Calculate loss
        # output_dim = outputs.shape[-1]
        # logits = outputs[1:].view(-1, output_dim)
        # labels = answers[1:].view(-1).type(torch.LongTensor).to(device)
        # # print(logits.shape, labels.shape, type(logits), type(labels))
        # loss = criterion(logits, labels)
        #
        # # Backward pass
        # loss.backward()
        # optimizer.step()
        #
        # # Print loss every 100 batches
        # if (batch_idx + 1) % 100 == 0:
        #     print(f'Batch [{batch_idx + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')
        #     print_qa_example(questions, answers, outputs, dataset, batch_idx, len(dataloader), 'Train')
        #
        # # Accumulate total loss for the epoch
        # total_loss += loss.item()
    return total_loss / len(dataloader)


if __name__ == "__main__":

    ds_path = "../../../../../net/projects/ranalab/kh310/vqa"
    ds_train = VQADataset(ds_path, "train")
    ds_val = VQADataset(ds_path, "val")

    VOCAB_SIZE = len(ds_train.vocab)
    BATCH_SIZE = 64
    NUM_EPOCH = 100

    # resnet = ResNet18(256)
    resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

    optimizer = optim.Adam(resnet.parameters())
    TRG_PAD_IDX = 2
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_pad_image)
    val_loader = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_pad_image)

    for epoch in range(NUM_EPOCH):
        train_loss = train(resnet, train_loader, optimizer, criterion, ds_train)
        break
        # val_loss = eval(model, val_loader, criterion, qa_dataset_val)
        # print('Epoch [{}/{}], Average Training Loss: {:.4f}'.format(epoch + 1, NUM_EPOCH, train_loss))
        # print('Epoch [{}/{}], Average Validation Loss: {:.4f}'.format(epoch + 1, NUM_EPOCH, val_loss))
