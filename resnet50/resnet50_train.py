import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import time

# 1. デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. データセット（MNIST）と前処理
transform = transforms.Compose([
    transforms.Resize((224, 224)),                  # ResNet50に合わせる
    transforms.Grayscale(num_output_channels=3),    # 1ch → 3ch
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 3. ResNet-50の読み込みと修正（出力層：10クラス）
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

# 4. 損失関数・最適化
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. 学習ループ（時間計測付き）
epochs = 5
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

print(f"Test Accuracy: {100 * correct / total:.2f}%")
