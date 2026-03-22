import kagglehub
from torch import optim

path = kagglehub.dataset_download("alessiocorrado99/animals10")

print("Path to dataset files:", path)

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

data_dir = os.path.join(path, 'raw-img')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define image transformations: resize to 128x128 and convert to PyTorch tensors
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

print("Classes found:", full_dataset.classes)
print(f"Total images: {len(full_dataset)}")

# Italian to English mapping for class labels
italian_to_english = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "ragno": "spider",
    "scoiattolo": "squirrel"
}

idx_to_class = {v: k for k, v in full_dataset.class_to_idx.items()}
print("idx_to_class:", idx_to_class)

import json

with open("idx_to_class.json", "w") as f:
    json.dump(idx_to_class, f)

# Split dataset into 80% training and 20% validation
train_size = int(0.8 * len(full_dataset))
valid_size = len(full_dataset) - train_size
train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [train_size, valid_size])

# DataLoaders handle batching and shuffling
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)


import torch
import torch.nn as nn

class AnimalCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        # conv blocks 128x128 -> Output 64x64
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)  # 32x32
        self.bn1   = nn.BatchNorm2d(32) # Standardizes layer inputs for faster training
        self.pool1 = nn.MaxPool2d(2)

        # 64x64 -> 32x32
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False) # 16x16
        self.bn2   = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)

        # 32x32 -> 16x16
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False) # 8x8
        self.bn3   = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2)

        # 16x16 -> 8x8
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False) # 4x4
        self.bn4   = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(2)

        self.relu = nn.ReLU(inplace=True)
        self.drop_conv = nn.Dropout(p=0.25) # Prevents overfitting in conv layers


        self.flatten_dim = 64 * 8 * 8
        self.fc1 = nn.Linear(self.flatten_dim, 64)
        self.bn_fc1 = nn.BatchNorm1d(64)
        self.drop_fc = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(64, num_classes)  # logits output

    def forward(self, x):

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)


        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)

        x = self.drop_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(self.bn_fc1(x))
        x = self.drop_fc(x)
        x = self.fc2(x)
        return x


model = AnimalCNN(num_classes=10).to(device)
print(model)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10

# Scheduler to adjust learning rate dynamically using the OneCycle policy
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.003,
    steps_per_epoch=len(train_loader),
    epochs=epochs,
)


# Training loop - PyTorch requires explicit training code (unlike Keras fit())
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

for epoch in range(epochs):
    # ============ TRAINING PHASE ============
    model.train()  # Set model to training mode (enables dropout, batch norm updates)
    train_loss, train_correct, train_total = 0, 0, 0

    for images, labels in train_loader:
        # Move data to GPU/CPU
        images, labels = images.to(device), labels.to(device)

        # Forward pass: compute predictions
        optimizer.zero_grad()  # Clear gradients from previous step
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass: compute gradients and update weights
        loss.backward()  # Compute gradients
        optimizer.step()  # Update model parameters
        scheduler.step()  # Update learning rate

        # Track statistics
        train_loss += loss.item() * images.size(0)
        train_correct += (outputs.argmax(1) == labels).sum().item()
        train_total += labels.size(0)

    # Validation
    model.eval()  # Set model to evaluation mode (disables dropout, batch norm in eval mode)
    val_loss, val_correct, val_total = 0, 0, 0

    with torch.no_grad():  # Disable gradient computation for validation (saves memory)
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += labels.size(0)

    # Record metrics for plotting
    history['train_loss'].append(train_loss / train_total)
    history['train_acc'].append(train_correct / train_total)
    history['val_loss'].append(val_loss / val_total)
    history['val_acc'].append(val_correct / val_total)

    print(
        f"Epoch {epoch + 1}: "
        f"Train Loss={train_loss / train_total:.3f}, "
        f"Train Acc={train_correct / train_total:.3f}, "
        f"Val Loss={val_loss / val_total:.3f}, "
        f"Val Acc={val_correct / val_total:.3f}"
    )

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in valid_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())


import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history['train_loss'], 'o-', label='Train')
ax1.plot(history['val_loss'], 's-', label='Val')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Loss')
ax1.legend()

ax2.plot(history['train_acc'], 'o-', label='Train')
ax2.plot(history['val_acc'], 's-', label='Val')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Accuracy')
ax2.legend()

plt.tight_layout()
plt.show()

torch.save(model.state_dict(), "animal_cnn.pth")