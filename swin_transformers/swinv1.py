import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
from PIL import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split # 新增導入
import timm
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.distributed import init_process_group, destroy_process_group

# Updated Label Encoder
class Label_encoder:
    def __init__(self, labels):
        self.labels = {label: idx for idx, label in enumerate(labels)}

    def get_label(self, idx):
        return list(self.labels.keys())[idx]

    def get_idx(self, label):
        return self.labels.get(label)


# Custom Dataset for Image Classification
class Food101Dataset(Dataset):
    def __init__(self, dataframe, encoder, transform=None):
        self.dataframe = dataframe
        self.encoder = encoder
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['path']
        label = self.dataframe.iloc[idx]['label']
        label = self.encoder.get_idx(label)

        try:
            image = Image.open(img_path).convert("RGB")
        except OSError as e:
            print(f"Warning: Failed to load image {img_path}: {e}")
            # 如果圖片載入失敗，可以返回一個標記或者跳過這個樣本
            # 這裡我們返回 None，需要在 DataLoader 中處理這種情況
            # 或者，可以引發一個異常，或者返回一個預設的圖片和標籤
            # 為了簡單起見，這裡返回 None，但更好的做法是讓 Dataset 返回有效資料
            # 並在 prepare_dataframe 中過濾掉損壞的圖片
            # 考慮到現有結構，我們暫時返回 None，但建議後續優化
            # 返回一個 placeholder tensor 可能更適合 DataLoader
            # 例如: return torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE)), -1 # 假設 IMAGE_SIZE 已定義
            # 但為了最小化改動，我們先這樣處理，並在 collate_fn 中過濾
            print(f"Skipping corrupted image: {img_path}")
            return None # 標記為無效樣本

        if self.transform:
            image = self.transform(image)

        return image, label


# Updated prepare_dataframe function
def prepare_dataframe(image_root, encoder): # 修改函數簽名
    data = []
    # 遍歷 image_root (即 'data' 資料夾) 下的所有子資料夾
    for category_folder in os.listdir(image_root):
        category_path = os.path.join(image_root, category_folder)
        # 確保是資料夾且類別在 encoder.labels (即 LABELS) 中定義
        if os.path.isdir(category_path) and category_folder in encoder.labels:
            # 遍歷該類別資料夾下的所有檔案
            for img_name_with_ext in os.listdir(category_path):
                # 檢查副檔名，確保是圖片檔案
                if img_name_with_ext.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    img_path = os.path.join(category_path, img_name_with_ext)
                    data.append({
                        'label': category_folder,
                        'path': img_path
                    })
    df = pd.DataFrame(data)
    if df.empty:
        print(f"警告: 在 {image_root} 中沒有找到對應標籤的資料。")
        return pd.DataFrame(columns=['label', 'path'])
    return shuffle(df)


# Training and Testing Functions
def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, targets in tqdm(dataloader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print(f"Train Loss: {total_loss / len(dataloader):.3f} | Train Accuracy: {accuracy:.2f}%")


def test_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Testing"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print(f"Test Loss: {total_loss / len(dataloader):.3f} | Test Accuracy: {accuracy:.2f}%")
    return accuracy


# Generate CAM for Swin Transformer
def generate_cam_swin(model, input_tensor, target_layer, class_idx=None):
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # 註冊 hooks
    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_backward_hook(backward_hook)

    # 前向傳播
    output = model(input_tensor)
    if class_idx is None:
        class_idx = output.argmax(dim=1).item()

    model.zero_grad()
    output[:, class_idx].backward()

    grad = gradients[0]
    act = activations[0]

    if len(grad.shape) == 4:
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * act).sum(dim=1, keepdim=True)
    elif len(grad.shape) == 3:
        weights = grad.mean(dim=1, keepdim=True)
        cam = (weights * act).sum(dim=2, keepdim=True)
    else:
        raise ValueError("Unexpected gradient shape: {}".format(grad.shape))

    cam = torch.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    handle_fwd.remove()
    handle_bwd.remove()

    return cam[0, 0].detach().cpu().numpy()


def visualize_cam(image, cam):
    img_np = image.cpu().permute(1, 2, 0).numpy()
    img_np = (img_np * 0.5 + 0.5) * 255
    img_np = img_np.astype(np.uint8)

    cam_resized = cv2.resize(cam, (224, 224))
    cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img_np, 0.6, cam_heatmap, 0.4, 0)
    return overlay


# Main Program
if __name__ == "__main__":
    # 設定是否使用分散式訓練
    use_distributed = False
    local_rank = 0
    
    # 檢查環境變數，決定是否使用分散式訓練
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        use_distributed = True
        torch.distributed.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        print(f"分散式訓練已初始化，local_rank: {local_rank}")
    else:
        print("使用單一GPU訓練模式")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"使用設備: {device}")

    BATCH_SIZE = 64
    IMAGE_SIZE = 224
    # IMAGE_ROOT = "food-101/images" # 舊的路徑
    # TRAIN_FILE = "food-101/meta/train.txt" # 舊的檔案
    # TEST_FILE = "food-101/meta/test.txt" # 舊的檔案
    IMAGE_ROOT = "data" # 新的圖片根目錄

    LABELS =  ['Abalone', 'Abalonemushroom', 'Achoy', 'Adzukibean', 'Alfalfasprouts', 'Almond', 'Apple', 'Asparagus', 'Avocado', 'Babycorn', 'Bambooshoot', 'Banana', 'Beeftripe', 'Beetroot', 'Birds-nestfern', 'Birdsnest', 'Bittermelon', 'Blackmoss', 'Blackpepper', 'Blacksoybean', 'Blueberry', 'Bokchoy', 'Brownsugar', 'Buckwheat', 'Cabbage', 'Cardamom', 'Carrot', 'Cashewnut', 'Cauliflower', 'Celery', 'Centuryegg', 'Cheese', 'Cherry', 'Chestnut', 'Chilipepper', 'Chinesebayberry', 'Chinesechiveflowers', 'Chinesechives', 'Chinesekale', 'Cilantro', 'Cinnamon', 'Clove', 'Cocoa', 'Coconut', 'Corn', 'Cowpea', 'Crab', 'Cream', 'Cucumber', 'Daikon', 'Dragonfruit', 'Driedpersimmon', 'Driedscallop', 'Driedshrimp', 'Duckblood', 'Durian', 'Eggplant', 'Enokimushroom', 'Fennel', 'Fig', 'Fishmint', 'Freshwaterclam', 'Garlic', 'Ginger', 'Glutinousrice', 'Gojileaves', 'Grape', 'Grapefruit', 'GreenSoybean', 'Greenbean', 'Greenbellpepper', 'Greenonion', 'Guava', 'Gynuradivaricata', 'Headingmustard', 'Honey', 'Jicama', 'Jobstears', 'Jujube', 'Kale', 'Kelp', 'Kidneybean', 'Kingoystermushroom', 'Kiwifruit', 'Kohlrabi', 'Kumquat', 'Lettuce', 'Limabean', 'Lime', 'Lobster', 'Longan', 'Lotusroot', 'Lotusseed', 'Luffa', 'Lychee', 'Madeira_vine', 'Maitakemushroom', 'Mandarin', 'Mango', 'Mangosteen', 'Milk', 'Millet', 'Minongmelon', 'Mint', 'Mungbean', 'Napacabbage', 'Natto', 'Nori', 'Nutmeg', 'Oat', 'Octopus', 'Okinawaspinach', 'Okra', 'Olive', 'Onion', 'Orange', 'Oystermushroom', 'Papaya', 'Parsley', 'Passionfruit', 'Pea', 'Peach', 'Peanut', 'Pear', 'Pepper', 'Perilla', 'Persimmon', 'Pickledmustardgreens', 'Pineapple', 'Pinenut', 'Plum', 'Pomegranate', 'Pomelo', 'Porktripe', 'Potato', 'Pumpkin', 'Pumpkinseed', 'Quailegg', 'Radishsprouts', 'Rambutan', 'Raspberry', 'Redamaranth', 'Reddate', 'Rice', 'Rosemary', 'Safflower', 'Saltedpotherbmustard', 'Seacucumber', 'Seaurchin', 'Sesameseed', 'Shaggymanemushroom', 'Shiitakemushroom', 'Shrimp', 'Snowfungus', 'Soybean', 'Soybeansprouts', 'Soysauce', 'Staranise', 'Starfruit', 'Strawberry', 'Strawmushroom', 'Sugarapple', 'Sunflowerseed', 'Sweetpotato', 'Sweetpotatoleaves', 'Taro', 'Thyme', 'Tofu', 'Tomato', 'Wasabi', 'Waterbamboo', 'Watercaltrop', 'Watermelon', 'Waterspinach', 'Waxapple', 'Wheatflour', 'Wheatgrass', 'Whitepepper', 'Wintermelon', 'Woodearmushroom', 'Yapear', 'Yauchoy', 'spinach']

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    encoder = Label_encoder(LABELS)

    # train_df = prepare_dataframe(TRAIN_FILE, IMAGE_ROOT, encoder) # 舊的讀取方式
    # test_df = prepare_dataframe(TEST_FILE, IMAGE_ROOT, encoder) # 舊的讀取方式

    # 新的資料讀取和分割方式
    all_data_df = prepare_dataframe(IMAGE_ROOT, encoder)

    if all_data_df.empty or len(all_data_df) < 2:
        print("錯誤: 資料不足以分割成訓練集和測試集。請檢查資料夾和 LABELS。程式即將結束。")
        exit()

    # 分割資料集成訓練集和測試集 (例如 80% 訓練, 20% 測試)
    train_df, test_df = train_test_split(
        all_data_df,
        test_size=0.2,       # 20% 的資料作為測試集
        random_state=42,     # 確保每次分割結果一致
        stratify=all_data_df['label'] if not all_data_df.empty else None # 確保類別分佈在訓練集和測試集中相似
    )

    if train_df.empty or test_df.empty:
        print("錯誤: 資料分割後訓練集或測試集為空。請檢查資料量和分割比例。程式即將結束。")
        exit()

    train_dataset = Food101Dataset(train_df, encoder, transform)
    test_dataset = Food101Dataset(test_df, encoder, transform)

    # 新增 collate_fn 來過濾掉 None 的樣本 (損壞的圖片)
    def collate_fn_skip_corrupted(batch):
        batch = list(filter(lambda x: x is not None, batch))
        if not batch: # 如果整個 batch 都是損壞的圖片
            return torch.Tensor(), torch.Tensor() # 返回空的 tensor
        return torch.utils.data.dataloader.default_collate(batch)

    if use_distributed:
        train_sampler = DistributedSampler(train_dataset)
        test_sampler = DistributedSampler(test_dataset)
        # 在 DataLoader 中使用 collate_fn
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, collate_fn=collate_fn_skip_corrupted, workers=64)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, sampler=test_sampler, collate_fn=collate_fn_skip_corrupted, workers=64)
        device = torch.device(f"cuda:{local_rank}")
    else:
        # 在 DataLoader 中使用 collate_fn
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_skip_corrupted)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_skip_corrupted)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_epochs = 30
    
    model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=len(LABELS))
    model = model.to(device)
    
    if use_distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        target_layer = model.module.layers[-1].blocks[-1].norm2
    else:
        # 若只使用單一GPU，可以選擇使用 DataParallel 加速
        if torch.cuda.device_count() > 1:
            print(f"使用 {torch.cuda.device_count()} 個 GPU 運行 DataParallel")
            model = nn.DataParallel(model)
            target_layer = model.module.layers[-1].blocks[-1].norm2
        else:
            target_layer = model.layers[-1].blocks[-1].norm2

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_acc = 0
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        if use_distributed:
            train_sampler.set_epoch(epoch)
        train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        test_acc = test_epoch(model, test_loader, criterion, device)

        if test_acc > best_acc:
            best_acc = test_acc
            if not use_distributed or local_rank == 0:
                torch.save(model.state_dict(), 'swin_model_test.pth')
                print(f"模型已保存，準確率: {best_acc:.2f}%")

    if use_distributed and torch.distributed.is_initialized():
        destroy_process_group()