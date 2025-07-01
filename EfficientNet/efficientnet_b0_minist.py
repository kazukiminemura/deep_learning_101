import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import loralib as lora
import time
import numpy as np
from torch.utils.data import Subset

# 改良されたEfficientNetWithLoRAクラス
class EfficientNetWithLoRA(nn.Module):
    def __init__(self, r=32, alpha=64, dropout_rate=0.3):
        super().__init__()
        self.model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # より多くの層を微調整対象にする
        for name, param in self.model.named_parameters():
            if any(layer in name for layer in ["features.5", "features.6", "features.7", "classifier"]):
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # LoRAパラメータを増やす
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            lora.Linear(in_features, 512, r=r, lora_alpha=alpha, fan_in_fan_out=False),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    # 1. デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. より強力なデータ拡張
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    # データセット準備
    full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

    # 訓練データの一部を検証用に分割
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(full_train_dataset)))

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_train_dataset, val_indices)

    # バッチサイズを調整（Windowsでは num_workers=0 にする）
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    # モデルインスタンス（パラメータを調整）
    model = EfficientNetWithLoRA(r=32, alpha=64, dropout_rate=0.2).to(device)

    # 4. 改良された最適化設定
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=0.0005,  # 学習率を下げる
        weight_decay=0.01  # 重み減衰を追加
    )

    # より柔軟なスケジューラ（verboseパラメータを削除）
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # Early Stopping用
    best_val_acc = 0
    patience_counter = 0
    early_stop_patience = 7

    # 5. 改良された学習ループ
    epochs = 25  # エポック数を増やす
    total_start = time.time()

    train_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        epoch_start = time.time()
        
        # 訓練フェーズ
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 検証フェーズ
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * val_correct / val_total
        val_accuracies.append(val_accuracy)
        
        # スケジューラステップ
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_accuracy)
        new_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Acc: {val_accuracy:.2f}%, Time: {epoch_time:.2f} sec")
        
        # 学習率が変更された場合に通知
        if old_lr != new_lr:
            print(f"Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")
        
        # Best modelの保存
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
            print(f"New best validation accuracy: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    total_time = time.time() - total_start
    print(f"Total Training Time: {total_time:.2f} sec")

    # 6. 最高性能モデルで最終評価
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    # Test Time Augmentation (TTA)
    def test_time_augmentation(model, test_loader, num_augmentations=5):
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                
                # 複数の予測を平均化
                batch_predictions = torch.zeros(images.size(0), 10).to(device)
                
                for _ in range(num_augmentations):
                    # 軽微な変換を適用
                    augmented_images = transforms.functional.rotate(images, angle=np.random.uniform(-5, 5))
                    outputs = model(augmented_images)
                    batch_predictions += torch.softmax(outputs, dim=1)
                
                batch_predictions /= num_augmentations
                all_predictions.append(batch_predictions)
                all_labels.append(labels)
        
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        _, predicted = torch.max(all_predictions, 1)
        accuracy = 100 * (predicted == all_labels).sum().item() / all_labels.size(0)
        return accuracy

    # 通常のテスト
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    normal_accuracy = 100 * correct / total
    print(f"Normal Test Accuracy: {normal_accuracy:.2f}%")

    # TTAを使用したテスト
    tta_accuracy = test_time_augmentation(model, test_loader, num_augmentations=3)
    print(f"TTA Test Accuracy: {tta_accuracy:.2f}%")

    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
