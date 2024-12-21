import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
from tqdm import tqdm  # 用於進度條顯示
import time  # 用於計算執行時間
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
import json  # 用於保存結果

# 設定基本參數
data_dir = 'archive/eyepac-light-v2-512-jpg'  # 資料根目錄
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'validation')
test_dir = os.path.join(data_dir, 'test')
batch_size = 16
num_classes = 2  # RG, NRG
num_epochs = 10
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定義資料轉換(包含資料增強及正規化)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 建立資料集
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

# 建立 DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# 使用預訓練模型(以 ResNet18 為例)
model = models.resnet18(pretrained=True)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)
model = model.to(device)

# 定義損失函數與優化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 訓練與驗證函數
def train_one_epoch(model, dataloader, optimizer, criterion, epoch, num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs} [Train]", unit="batch") as pbar:
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            pbar.set_postfix({"loss": loss.item()})
            pbar.update(1)
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    return epoch_loss, epoch_acc, all_preds, all_labels

# 訓練主迴圈
best_val_acc = 0.0
best_epoch = 0
total_start_time = time.time()
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, epoch, num_epochs)
    val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
    print(f'\nEpoch [{epoch+1}/{num_epochs}]: '
          f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n')
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        torch.save(model.state_dict(), 'best_model.pth')
        print("Model improved and saved.")

# 測試階段
model.load_state_dict(torch.load('best_model.pth'))
test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion)
precision = precision_score(test_labels, test_preds, average='binary')
recall = recall_score(test_labels, test_preds, average='binary')
conf_matrix = confusion_matrix(test_labels, test_preds)

# 保存結果到 JSON 文件
results = {
    "best_epoch": best_epoch,
    "test_loss": test_loss,
    "test_accuracy": test_acc.item(),
    "precision": precision,
    "recall": recall,
    "confusion_matrix": conf_matrix.tolist()
}
with open('results.json', 'w') as f:
    json.dump(results, f, indent=4)

print(f"Results saved to 'results.json'.")
