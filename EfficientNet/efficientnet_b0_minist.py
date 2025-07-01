import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import loralib as lora
import time

# 1. デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. データ前処理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# EfficientNetWithLoRAクラスの一部を修正
class EfficientNetWithLoRA(nn.Module):
    def __init__(self, r=16, alpha=32):
        super().__init__()
        self.model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # 一部の層だけ学習対象にする（最後の特徴抽出層）
        for name, param in self.model.named_parameters():
            if "features.6" in name or "features.7" in name or "classifier" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # classifier[1] にLoRAを導入
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = lora.Linear(in_features, 10, r=r, lora_alpha=alpha, fan_in_fan_out=False)

    def forward(self, x):
        return self.model(x)

# モデルインスタンス
model = EfficientNetWithLoRA().to(device)

# 4. 損失関数、最適化、スケジューラ
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=15)

# 5. 学習ループ
epochs = 15
total_start = time.time()

for epoch in range(epochs):
    epoch_start = time.time()
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    scheduler.step()
    epoch_time = time.time() - epoch_start
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Time: {epoch_time:.2f} sec")

total_time = time.time() - total_start
print(f"Total Training Time: {total_time:.2f} sec")

# 6. 評価
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
