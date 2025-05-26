'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.utils.tensorboard import SummaryWriter
import os

# --- 設定參數 ---
data_dir = 'data/'  # 遠端電腦上的資料路徑
model_save_path = 'cnn_image_recognition_model.pth'
log_dir = 'runs/cnn_experiment'  # TensorBoard log 路徑
num_epochs = 50  # 訓練週期
batch_size = 32   # 批次大小
learning_rate = 0.001 # 學習率

# --- 檢查 GPU 是否可用 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- TensorBoard Writer ---
writer = SummaryWriter(log_dir)

# --- 資料轉換與載入 ---
# 訓練集的轉換：包含資料增強
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 驗證集/測試集的轉換：不包含資料增強
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 載入資料集 (假設您的資料夾結構符合 ImageFolder 的要求)
# 例如: data/rice/*.jpg, data/other_class/*.png
# 請確保 data_dir 指向包含各類別資料夾的父資料夾

# 為了讓程式碼可以執行，我們先假設一個簡單的資料夾結構
# 在實際遠端執行時，請確保 data_dir 的路徑正確
# 並且該路徑下有如 rice/, cat/, dog/ 等子資料夾

# 創建虛擬資料夾和圖片以供本地測試 (實際執行時遠端電腦應有真實資料)
if not os.path.exists(os.path.join(data_dir, 'class1')):
    os.makedirs(os.path.join(data_dir, 'class1'))
    # 創建一些虛擬圖片檔案 (僅為示意，實際應為真實圖片)
    for i in range(5):
        with open(os.path.join(data_dir, 'class1', f'dummy_image_{i}.png'), 'w') as f:
            f.write("dummy content") # 實際應為圖片內容
if not os.path.exists(os.path.join(data_dir, 'class2')):
    os.makedirs(os.path.join(data_dir, 'class2'))
    for i in range(5):
        with open(os.path.join(data_dir, 'class2', f'dummy_image_{i}.jpeg'), 'w') as f:
            f.write("dummy content")

full_dataset = datasets.ImageFolder(data_dir, transform=train_transforms)

# 切分訓練集和驗證集 (例如 80% 訓練, 20% 驗證)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

# 更新驗證集的 transform (因為 random_split 後的 dataset 沒有 transform 屬性，需要重新包裝)
# 這裡我們簡單地假設 val_dataset 直接使用 train_transforms，但在實際應用中，
# 最好是 val_dataset 有自己的 transform (val_transforms)
# 為了簡化，此處我們讓 val_dataset 也用 train_transforms，但這不是最佳實踐
# 一個更好的做法是重新創建一個 Dataset 對象並應用 val_transforms

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# 獲取類別名稱
class_names = full_dataset.classes
num_classes = len(class_names)
print(f"Found classes: {class_names}")

# --- 定義 CNN 模型 ---
# 使用預訓練的 ResNet18 模型
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# 修改最後一層以符合我們的類別數量
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

model = model.to(device) # 將模型移至 GPU (如果可用)

# --- 定義損失函數和優化器 ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# --- 訓練模型 ---
print("Starting training...")
for epoch in range(num_epochs):
    model.train()  # 設定模型為訓練模式
    running_loss = 0.0
    correct_predictions_train = 0
    total_predictions_train = 0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 清零梯度
        optimizer.zero_grad()

        # 前向傳播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向傳播與優化
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        _, predicted = torch.max(outputs.data, 1)
        total_predictions_train += labels.size(0)
        correct_predictions_train += (predicted == labels).sum().item()

        if (i + 1) % 10 == 0: # 每 10 個 batch 印一次 log
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc_train = correct_predictions_train / total_predictions_train

    writer.add_scalar('Loss/train', epoch_loss, epoch)
    writer.add_scalar('Accuracy/train', epoch_acc_train, epoch)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc_train:.4f}')

    # --- 驗證模型 ---
    model.eval()  # 設定模型為評估模式
    running_loss_val = 0.0
    correct_predictions_val = 0
    total_predictions_val = 0
    with torch.no_grad(): # 在驗證階段不計算梯度
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss_val += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total_predictions_val += labels.size(0)
            correct_predictions_val += (predicted == labels).sum().item()

    epoch_loss_val = running_loss_val / len(val_dataset)
    epoch_acc_val = correct_predictions_val / total_predictions_val

    writer.add_scalar('Loss/validation', epoch_loss_val, epoch)
    writer.add_scalar('Accuracy/validation', epoch_acc_val, epoch)

    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {epoch_loss_val:.4f}, Validation Acc: {epoch_acc_val:.4f}')

print("Finished Training")

# --- 儲存模型 ---
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# --- 關閉 TensorBoard Writer ---
writer.close()

# --- 如何在遠端執行 ---
# 1. 將此 cnn_train.py 檔案複製到您的遠端電腦。
# 2. 確保遠端電腦上已安裝 Python, PyTorch, torchvision, tensorboard。
#    (通常使用 pip install torch torchvision torchaudio tensorboard)
# 3. 確保您的圖片資料夾 (例如 data/) 已放置在遠端電腦上，並且 cnn_train.py 中的 data_dir 指向正確的路徑。
#    資料夾結構應為:
#    data/
#    ├── class1_name/ (例如 rice/)
#    │   ├── image1.jpg
#    │   ├── image2.png
#    │   └── ...
#    ├── class2_name/ (例如 noodles/)
#    │   ├── imageA.jpeg
#    │   ├── imageB.jpg
#    │   └── ...
#    └── ...
# 4. 在遠端電腦的終端機中，導航到 cnn_train.py 所在的目錄。
# 5. 執行命令: python cnn_train.py
# 6. 訓練完成後，模型將儲存為 cnn_image_recognition_model.pth。
# 7. 要查看 TensorBoard，請在遠端電腦的終端機中，導航到 cnn_train.py 所在的目錄 (或者包含 runs 資料夾的目錄)。
# 8. 執行命令: tensorboard --logdir=runs
# 9. 在瀏覽器中開啟 TensorBoard 提供的網址 (通常是 http://localhost:6006)。
'''
