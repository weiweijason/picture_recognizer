# filepath: c:\Users\User\OneDrive - National ChengChi University\113-2 Design\project\CNN\efficientnet_dual_gpu_train.py
'''
EfficientNet 圖像辨識程式碼 (PyTorch, 雙GPU, TensorBoard)

此程式預計於遠端電腦執行，使用雙GPU進行平行運算，圖片資料已存放於遠端電腦的 data/ 資料夾中。
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
from PIL import Image, UnidentifiedImageError
import os
import time

# --- 設定參數 ---
data_dir = 'data/'  # 遠端電腦上的資料路徑
model_name = "efficientnet_b7"  # 可以選擇 "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3", "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7"
num_epochs = 50
batch_size = 64  # 雙GPU可以考慮增加batch size
learning_rate = 0.001

# TensorBoard 設定
log_dir = 'runs/efficientnet_dual_gpu_experiment_{}'.format(time.strftime("%Y%m%d-%H%M%S"))
writer = SummaryWriter(log_dir)

print(f"TensorBoard log directory: {os.path.abspath(log_dir)}")

# --- 檢查 GPU 是否可用並設定多GPU ---
if torch.cuda.is_available():
    device = torch.device("cuda")
    gpu_count = torch.cuda.device_count()
    print(f"CUDA is available! Found {gpu_count} GPU(s)")
    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 檢查是否有至少2個GPU
    if gpu_count >= 2:
        print("Using dual GPU setup with DataParallel")
        use_multi_gpu = True
    else:
        print("Only 1 GPU available, using single GPU")
        use_multi_gpu = False
else:
    device = torch.device("cpu")
    use_multi_gpu = False
    print("CUDA is not available, using CPU")

print(f"Primary device: {device}")

# --- 資料預處理與載入 ---
# EfficientNet 的輸入大小設定
efficientnet_input_sizes = {
    'efficientnet_b0': 224,
    'efficientnet_b1': 240,
    'efficientnet_b2': 260,
    'efficientnet_b3': 300,
    'efficientnet_b4': 380,
    'efficientnet_b5': 456,
    'efficientnet_b6': 528,
    'efficientnet_b7': 600
}

image_size = efficientnet_input_sizes.get(model_name, 224)
print(f"Using image size {image_size} for {model_name}")

data_transforms = {
    'train': transforms.Compose([
        transforms.Lambda(lambda x: x.convert('RGB')),  # 確保圖片是 RGB
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),  # 增加資料增強
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet 的標準化參數
    ]),
    'val': transforms.Compose([
        transforms.Lambda(lambda x: x.convert('RGB')),  # 確保圖片是 RGB
        transforms.Resize(image_size + 32),  # 通常驗證集會比訓練集大一點再裁切
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

class DatasetWithTransform(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.subset)

def is_valid_image_file(path: str) -> bool:
    """
    檢查給定的路徑是否為有效的圖片檔案。
    如果圖片無法開啟或驗證，則印出警告並回傳 False。
    """
    try:
        img = Image.open(path)
        img.load()  # 直接嘗試完整載入圖片資料
        return True
    except (UnidentifiedImageError, IOError, OSError, AttributeError) as e:
        print(f"Warning: Skipping corrupted, unidentifiable or problematic image file: {path} (Error: {e})")
        return False

full_dataset_pil = datasets.ImageFolder(
    data_dir,
    transform=None,  # 移除基礎轉換，讓每個階段的轉換自己處理
    is_valid_file=is_valid_image_file  # 加入檔案有效性檢查
)

# 計算類別數量並設定訓練/驗證集大小
class_names = full_dataset_pil.classes
num_classes = len(class_names)
print(f"Found {num_classes} classes: {class_names}. Total valid images: {len(full_dataset_pil)}")

if num_classes == 0 or len(full_dataset_pil) == 0:
    print(f"Error: No classes found or no valid images in '{data_dir}' after filtering. Please check data and directory structure.")
    exit()

# 確保有足夠的圖片進行分割
if len(full_dataset_pil) < 2:
    print(f"Error: Not enough valid images ({len(full_dataset_pil)}) to split into training and validation sets. Need at least 2 images.")
    exit()

train_size = int(0.8 * len(full_dataset_pil))
val_size = len(full_dataset_pil) - train_size

# 確保 train_size 和 val_size 至少為 1
if train_size == 0:
    train_size = 1
    val_size = len(full_dataset_pil) - 1
if val_size == 0:
    val_size = 1
    train_size = len(full_dataset_pil) - 1

train_dataset_pil, val_dataset_pil = torch.utils.data.random_split(full_dataset_pil, [train_size, val_size])

train_dataset_transformed = DatasetWithTransform(train_dataset_pil, transform=data_transforms['train'])
val_dataset_transformed = DatasetWithTransform(val_dataset_pil, transform=data_transforms['val'])

# 增加 worker 數量至64以提高資料載入效率
# 同時根據GPU數量調整batch size
total_batch_size = batch_size * (2 if use_multi_gpu else 1)
print(f"Total effective batch size: {total_batch_size}")

dataloaders = {
    'train': DataLoader(
        train_dataset_transformed, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=64,  # 增加worker數量
        pin_memory=True if torch.cuda.is_available() else False,  # 加速GPU資料傳輸
        persistent_workers=True  # 保持worker持續運行
    ),
    'val': DataLoader(
        val_dataset_transformed, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=64,  # 增加worker數量
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True
    )
}
dataset_sizes = {'train': len(train_dataset_transformed), 'val': len(val_dataset_transformed)}

print(f"Training set size: {dataset_sizes['train']}")
print(f"Validation set size: {dataset_sizes['val']}")
print(f"Using {64} workers for data loading")

# --- EfficientNet 模型定義 ---
def get_efficientnet_model(model_name, num_classes, pretrained=True):
    """
    取得 EfficientNet 模型並修改最後一層以符合類別數量
    """
    model = None
    
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=pretrained)
    elif model_name == "efficientnet_b1":
        model = models.efficientnet_b1(pretrained=pretrained)
    elif model_name == "efficientnet_b2":
        model = models.efficientnet_b2(pretrained=pretrained)
    elif model_name == "efficientnet_b3":
        model = models.efficientnet_b3(pretrained=pretrained)
    elif model_name == "efficientnet_b4":
        model = models.efficientnet_b4(pretrained=pretrained)
    elif model_name == "efficientnet_b5":
        model = models.efficientnet_b5(pretrained=pretrained)
    elif model_name == "efficientnet_b6":
        model = models.efficientnet_b6(pretrained=pretrained)
    elif model_name == "efficientnet_b7":
        model = models.efficientnet_b7(pretrained=pretrained)
    else:
        raise ValueError(f"EfficientNet model {model_name} not recognized. Supported models: efficientnet_b0 to efficientnet_b7")
    
    # 修改分類器層
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    # 移動到主要設備
    model = model.to(device)
    
    # 如果有多個GPU，使用DataParallel
    if use_multi_gpu and torch.cuda.device_count() > 1:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    return model

model = get_efficientnet_model(model_name, num_classes)
print(f"Model: {model_name}")
print(f"Model is on device: {next(model.parameters()).device}")

# 計算模型參數數量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# --- 損失函數和優化器 ---
criterion = nn.CrossEntropyLoss()

# 對於EfficientNet，可以使用不同的優化器策略
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

# 學習率調整器 - 對EfficientNet效果較好
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

# --- 訓練函數 ---
def train_model(model, criterion, optimizer, scheduler=None, num_epochs=25):
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    # 記錄學習率
    writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], 0)

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
            batch_count = 0

            # 迭代資料
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()  # 清零梯度

                # 前向傳播
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
                batch_count += 1
                
                # 每100個batch印一次進度
                if phase == 'train' and batch_count % 100 == 0:
                    print(f'Batch {batch_count}, Loss: {loss.item():.4f}')

            # 學習率調整
            if phase == 'train' and scheduler:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 記錄到 TensorBoard
            if phase == 'train':
                writer.add_scalar('Loss/train', epoch_loss, epoch)
                writer.add_scalar('Accuracy/train', epoch_acc, epoch)
                writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
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
        print("Starting EfficientNet training with enhanced configuration...")
        print(f"GPU configuration: {'Dual GPU' if use_multi_gpu else 'Single GPU/CPU'}")
        print(f"Data loading workers: 64")
        print(f"Model: {model_name}")
        print(f"Image size: {image_size}")
        
        model_trained = train_model(model, criterion, optimizer, scheduler, num_epochs=num_epochs)
        
        # 儲存最終訓練好的模型
        final_model_path = os.path.join(log_dir, 'final_model.pth')
        torch.save(model_trained.state_dict(), final_model_path)
        print(f"Final model saved to {final_model_path}")

        # 關閉 TensorBoard writer
        writer.close()
        print("Training finished. TensorBoard logs are in:", os.path.abspath(log_dir))
        print("To view TensorBoard, run: tensorboard --logdir=\"" + os.path.abspath(log_dir) + "\"")
    else:
        print("No data to train on. Please check data_dir and class structure.")
