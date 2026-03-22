import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CNN
class AnimalCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4   = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(2)

        self.relu = nn.ReLU(inplace=True)
        self.drop_conv = nn.Dropout(p=0.25)

        self.flatten_dim = 64 * 8 * 8
        self.fc1 = nn.Linear(self.flatten_dim, 64)
        self.bn_fc1 = nn.BatchNorm1d(64)
        self.drop_fc = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(64, num_classes)

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

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Initialize the model and load the trained weights
model = AnimalCNN(num_classes=10).to(device)
model.load_state_dict(torch.load("animal_cnn.pth", map_location=device))
model.eval()

with open("idx_to_class.json", "r") as f:
    idx_to_class = json.load(f)

idx_to_class = {int(k): v for k, v in idx_to_class.items()}


def predict_animal(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        pred_idx = outputs.argmax(dim=1).item()

    italian_label = idx_to_class[pred_idx]
    english_label = italian_to_english[italian_label]

    return english_label


if __name__ == "__main__":
    test_image = "test.jpg"
    print("Predicted animal:", predict_animal(test_image))