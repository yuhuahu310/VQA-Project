import torch
import torch.nn as nn
from dataset import QDDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import os

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QualityDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnext = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)
        self.fc = nn.Linear(1000, 8) # output dim of resnext = 1000

    def forward(self, x):
        x = self.resnext(x)
        x = self.fc(x)
        return x

def train(model, dataloader, optimizer, criterion):
    total_loss = 0
    model.train()
    for batch_idx, (images, targets) in enumerate(dataloader):
        # Move data to GPU if available
        images = images.to(device)
        targets = targets.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Print loss every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f'Batch [{batch_idx + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')

        # Accumulate total loss for the epoch
        total_loss += loss.item()
    return total_loss / len(dataloader)

def eval(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            # Move data to GPU if available
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def save_model(model, epoch, save_dir, train_loss, val_loss):
    path = os.path.join(save_dir, f'QD-epoch_{epoch}-trainloss_{train_loss:.4f}-valloss_{val_loss:.4f}.pth')
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, file_path):
    model.load_state_dict(torch.load(file_path))
    print(f"Model loaded from {file_path}")

if __name__ == "__main__":
    SAVE_DIR = 'ResNext_ckpt'
    SAVE_FREQ = 5
    BATCH_SIZE = 64
    NUM_EPOCH = 40
    FREEZE_EPOCH = 5
    dataset_train = QDDataset("../data", "train")
    dataset_val = QDDataset("../data", "val")
    os.makedirs(SAVE_DIR, exist_ok=True)

    model = QualityDetector().to(device)
    load_model(model, 'ResNext_ckpt/QD-epoch_19-trainloss_0.5231-valloss_0.8162.pth')
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)

    TRAIN = True
    if TRAIN:
        # for epoch in range(NUM_EPOCH):
        for epoch in range(20, NUM_EPOCH):
            if epoch < FREEZE_EPOCH:
                model.resnext.requires_grad_(False)
            else:
                model.resnext.requires_grad_(True)
            train_loss = train(model, train_loader, optimizer, criterion)
            val_loss = eval(model, val_loader, criterion)
            print('Epoch [{}/{}], Average Training Loss: {:.4f}'.format(epoch + 1, NUM_EPOCH, train_loss))
            print('Epoch [{}/{}], Average Validation Loss: {:.4f}'.format(epoch + 1, NUM_EPOCH, val_loss))
            if (epoch + 1) % SAVE_FREQ == 0:
                save_model(model, epoch, SAVE_DIR, train_loss, val_loss)
    else:
        train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=False)
        # model = load_model(model, f"{SAVE_DIR}/LSTM-epoch_79-trainloss_1.0620-valloss_5.3343.pth")
        train_loss = eval(model, train_loader, criterion, dataset_train)
        val_loss = eval(model, val_loader, criterion, dataset_val)
        print('Training loss: {:.4f}'.format(train_loss))
        print('Validation loss: {:.4f}'.format(val_loss))