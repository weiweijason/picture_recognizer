'''
CNN 圖像辨識程式碼 (PyTorch, GPU, TensorBoard)

此程式預計於遠端電腦執行，圖片資料已存放於遠端電腦的 data/ 資料夾中。
資料夾結構應為：
data/
    class1_name/
        image1.jpg
        image2.png
        ...
    class2_name/
        imageA.jpeg
        imageB.jpg
        ...
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.utils.tensorboard import SummaryWriter
from PIL import Image, UnidentifiedImageError # Added imports
import os
import time

# --- 設定參數 ---
data_dir = 'data/'  # 遠端電腦上的資料路徑
model_name = "resnet18"  # 可以選擇 "resnet18", "resnet34", "resnet50", "vgg16", 等
num_epochs = 50
batch_size = 32
learning_rate = 0.001
# image_size = 224 # ResNet 和 VGG 通常需要 224x224

# TensorBoard 設定
log_dir = 'runs/cnn_experiment_{}'.format(time.strftime("%Y%m%d-%H%M%S"))
writer = SummaryWriter(log_dir)

print(f"TensorBoard log directory: {os.path.abspath(log_dir)}")

# --- 檢查 GPU 是否可用 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 資料預處理與載入 ---
# 根據模型調整 image_size
if "resnet" in model_name or "vgg" in model_name or "efficientnet" in model_name:
    image_size = 224
elif "inception" in model_name:
    image_size = 299
else:
    image_size = 128 # 預設或自訂模型的大小，可自行調整

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet 的標準化參數
    ]),
    'val': transforms.Compose([
        transforms.Resize(image_size + 32), # 通常驗證集會比訓練集大一點再裁切
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 重要：為驗證集設定正確的 transform
# 這裡我們需要一個技巧，因為 random_split 後的 dataset subset 沒有 transform 屬性可直接修改
# 一個方法是重新包裝 val_dataset 或在訓練迴圈中手動應用 transform
# 另一個更簡潔的方式是確保 ImageFolder 載入時的 transform 適用於多數情況，或在 DataLoader 中處理
# 為了簡單起見，我們這裡假設 train_transforms 也適用於 val，但最佳實踐是分開處理
# 或者，您可以為 val_dataset 的 subset 重新定義 transform，但这比較複雜
# 此處我們在建立 val_dataloader 時，可以傳遞一個 collate_fn 來應用 val_transform，但更常見的是分開建立 ImageFolder

# 為了正確處理 transform，我們重新定義 val_dataset 的 transform
# 這是一個常見的處理方式：
class DatasetWithTransform(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x) # 注意：ImageFolder 返回的是 PIL Image，ToTensor() 應在 transform 中
        return x, y
    def __len__(self):
        return len(self.subset)

# 由於 ImageFolder 已經做了 ToTensor，我們需要確保 val_transform 不重複做 ToTensor
# 或者，更好的方式是 ImageFolder 不做 ToTensor，讓 train/val transform 自己做
# 讓我們修改一下 ImageFolder 的 transform
base_transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert('RGB')), # 確保圖片是 RGB
    # ToTensor() 和 Normalize() 將由 train/val transform 各自處理
])

def is_valid_image_file(path: str) -> bool:
    """
    檢查給定的路徑是否為有效的圖片檔案。
    如果圖片無法開啟或驗證，則印出警告並回傳 False。
    """
    try:
        img = Image.open(path)
        img.load() # 直接嘗試完整載入圖片資料
        return True
    except (UnidentifiedImageError, IOError, OSError, AttributeError) as e: # 加入 AttributeError
        print(f"Warning: Skipping corrupted, unidentifiable or problematic image file: {path} (Error: {e})")
        return False

full_dataset_pil = datasets.ImageFolder(
    data_dir,
    transform=base_transform,
    is_valid_file=is_valid_image_file  # 加入檔案有效性檢查
)

# 計算類別數量並設定訓練/驗證集大小 (基於過濾後的 full_dataset_pil)
class_names = full_dataset_pil.classes
num_classes = len(class_names)
print(f"Found {num_classes} classes: {class_names}. Total valid images: {len(full_dataset_pil)}")

if num_classes == 0 or len(full_dataset_pil) == 0:
    print(f"Error: No classes found or no valid images in '{data_dir}' after filtering. Please check data and directory structure.")
    exit()

# 確保有足夠的圖片進行分割 (至少需要2張，訓練集和驗證集各一張)
if len(full_dataset_pil) < 2:
    print(f"Error: Not enough valid images ({len(full_dataset_pil)}) to split into training and validation sets. Need at least 2 images.")
    exit()

train_size = int(0.8 * len(full_dataset_pil))
val_size = len(full_dataset_pil) - train_size

# 確保 train_size 和 val_size 至少為 1 (因為 len(full_dataset_pil) >= 2)
if train_size == 0: # 理論上在 len(full_dataset_pil) >= 2 時不會發生
    train_size = 1
    val_size = len(full_dataset_pil) - 1
if val_size == 0: # 理論上在 len(full_dataset_pil) >= 2 時不會發生
    val_size = 1
    train_size = len(full_dataset_pil) - 1

train_dataset_pil, val_dataset_pil = torch.utils.data.random_split(full_dataset_pil, [train_size, val_size])

train_dataset_transformed = DatasetWithTransform(train_dataset_pil, transform=data_transforms['train'])
val_dataset_transformed = DatasetWithTransform(val_dataset_pil, transform=data_transforms['val'])

dataloaders = {
    'train': DataLoader(train_dataset_transformed, batch_size=batch_size, shuffle=True, num_workers=4 if device.type == 'cuda' else 0),
    'val': DataLoader(val_dataset_transformed, batch_size=batch_size, shuffle=False, num_workers=4 if device.type == 'cuda' else 0)
}
dataset_sizes = {'train': len(train_dataset_transformed), 'val': len(val_dataset_transformed)}

print(f"Training set size: {dataset_sizes['train']}")
print(f"Validation set size: {dataset_sizes['val']}")

# --- 模型定義 ---
def get_model(model_name, num_classes, pretrained=True):
    model = None
    if model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
    elif model_name == "resnet34":
        model = models.resnet34(pretrained=pretrained)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=pretrained)
    # 可以加入更多模型選擇
    else:
        raise ValueError(f"Model {model_name} not recognized.")

    if "resnet" in model_name:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif "vgg" in model_name:
        num_ftrs = model.classifier[6].in_features # VGG 的最後一層是 classifier[6]
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    # 其他模型的修改方式可能不同

    return model.to(device)

model = get_model(model_name, num_classes)
print(model)

# --- 損失函數和優化器 ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# (可選) 學習率調整器
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# --- 訓練函數 ---
def train_model(model, criterion, optimizer, num_epochs=25):
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 設定模型為訓練模式
            else:
                model.eval()   # 設定模型為評估模式

            running_loss = 0.0
            running_corrects = 0

            # 迭代資料
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad() # 清零梯度

                # 前向傳播
                # 只在訓練階段追蹤歷史
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 只在訓練階段進行反向傳播和優化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # if phase == 'train' and scheduler:
            #     scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 記錄到 TensorBoard
            if phase == 'train':
                writer.add_scalar('Loss/train', epoch_loss, epoch)
                writer.add_scalar('Accuracy/train', epoch_acc, epoch)
            else:
                writer.add_scalar('Loss/val', epoch_loss, epoch)
                writer.add_scalar('Accuracy/val', epoch_acc, epoch)

            # 深度複製模型權重
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                # 儲存最佳模型
                torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pth'))
                print(f"Best val Acc: {best_acc:.4f}, model saved.")

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # 載入最佳模型權重
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print(f"Error: Data directory '{data_dir}' is empty or does not exist on the remote machine.")
        print("Please ensure your images are in subdirectories named by class within this folder.")
        print("Example structure:")
        print("data/")
        print("├── cat/")
        print("│   ├── cat1.jpg")
        print("│   └── cat2.png")
        print("└── dog/")
        print("    ├── dog1.jpeg")
        print("    └── dog2.jpg")

    elif num_classes > 0:
        print("Starting training...")
        model_trained = train_model(model, criterion, optimizer, num_epochs=num_epochs)
        # 儲存最終訓練好的模型 (可能是最佳的，也可能是最後一個 epoch 的)
        final_model_path = os.path.join(log_dir, 'final_model.pth')
        torch.save(model_trained.state_dict(), final_model_path)
        print(f"Final model saved to {final_model_path}")

        # 關閉 TensorBoard writer
        writer.close()
        print("Training finished. TensorBoard logs are in:", os.path.abspath(log_dir))
        print("To view TensorBoard, run: tensorboard --logdir=\"" + os.path.abspath(log_dir) + "\"")
    else:
        print("No data to train on. Please check data_dir and class structure.")
