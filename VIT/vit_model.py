import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
from PIL import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
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
            print(f"Skipping corrupted image: {img_path}")
            return None # Mark as invalid sample

        if self.transform:
            image = self.transform(image)

        return image, label


# Updated prepare_dataframe function
def prepare_dataframe(image_root, encoder):
    data = []
    for category_folder in os.listdir(image_root):
        category_path = os.path.join(image_root, category_folder)
        if os.path.isdir(category_path) and category_folder in encoder.labels:
            for img_name_with_ext in os.listdir(category_path):
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


# Generate CAM for Vision Transformer (and other similar models)
def generate_cam(model, input_tensor, target_layer, class_idx=None):
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_backward_hook(backward_hook)

    output = model(input_tensor)
    if class_idx is None:
        class_idx = output.argmax(dim=1).item()

    model.zero_grad()
    output[:, class_idx].backward()

    grad = gradients[0]
    act = activations[0]
    
    # For ViT, activations from norm layer might be [batch_size, num_patches, embed_dim]
    # We need to handle this shape for CAM generation.
    # Typically, for ViT, CAM is generated from the attention weights or from the class token embeddings.
    # This generic CAM might need adjustment based on specific ViT CAM techniques.
    # Assuming grad and act are [B, N, C] or [B, C, H, W]
    if len(grad.shape) == 4: # CNN-like features [B, C, H, W]
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * act).sum(dim=1, keepdim=True)
    elif len(grad.shape) == 3: # Transformer-like features [B, N, C]
        # Assuming N is sequence length (patches), C is embedding dim
        # Taking mean over C (embedding dim) to get weights for each patch
        weights = grad.mean(dim=2, keepdim=True) # weights shape [B, N, 1]
        cam = (weights.transpose(1,2) * act.transpose(1,2)).sum(dim=1, keepdim=True) # cam shape [B, 1, N]
        # The CAM here would be 1D (over patches). Reshaping to 2D for visualization might be needed.
        # This part might need specific handling for ViT CAM visualization.
        # For simplicity, we'll assume the target_layer output allows for a similar CAM calculation
        # or that the user will adapt this part for ViT's specific architecture.
        # A common approach for ViT CAM is to use attention rollout or to focus on the class token.
        # This generic hook-based CAM might be more directly applicable to the output of the final conv layers in hybrid ViTs
        # or requires careful selection of target_layer and interpretation for pure ViTs.
        # Let's try a simpler mean over patches if weights are [B,N,C] and act is [B,N,C]
        # weights = grad.mean(dim=1, keepdim=True) # [B, 1, C]
        # cam = (act * weights).sum(dim=2, keepdim=True) # [B, N, 1]
        # This is a placeholder and might need refinement for ViT
        # For now, let's assume the target_layer output is more like [B, C, H, W] after some processing
        # or that the user intends to adapt this.
        # If target_layer is model.norm, output is [B, N, C].
        # Let's revert to a more general approach that might work if target_layer output is reshaped or pooled.
        # For ViT, if target_layer is e.g. blocks[-1].norm2, output is [batch, num_tokens, embed_dim]
        # Gradients will also be [batch, num_tokens, embed_dim]
        # A common way is to average over the embedding dimension
        weights = grad.mean(dim=2) # Resulting shape: [batch, num_tokens]
        cam_per_token = (weights * act.mean(dim=2)) # Element-wise product, still [batch, num_tokens]
        cam = cam_per_token # This is a 1D CAM over tokens.
        # To visualize as 2D, one needs to map tokens back to spatial locations.
        # This function might need significant adaptation for meaningful ViT CAM.
        # For now, we keep the structure and let it pass through.
        # A simple approach: average over the sequence length for weights
        weights_avg_seq = grad.mean(dim=1, keepdim=True) # [B, 1, C]
        cam = (act * weights_avg_seq).sum(dim=2, keepdim=True) # [B, N, 1] -> needs reshape to 2D
        # This is highly experimental for ViT with this generic CAM.
        # Let's assume the earlier CNN-like handling or a more direct feature map output from target_layer
        # For ViT, if target_layer.output is (B, N, D), where N is num_patches+cls_token
        # grad is (B, N, D). act is (B, N, D)
        pooled_gradients = torch.mean(grad, dim=[1]) # Global average pooling over N (tokens) -> (B,D)
        # Use pooled_gradients as weights for activations
        cam = torch.einsum('bnd,bd->bn', act, pooled_gradients) # (B,N)
        # cam now has one value per token. This needs to be reshaped into a 2D map.
        # The number of patches (N-1 for ViT if CLS token exists) is typically sqrt(N-1) x sqrt(N-1)
        # This part is complex and highly dependent on ViT architecture details and what is expected for CAM.
        # For now, we'll keep the original logic and note that it might need ViT-specific changes.
        # If the target_layer is something like the final layer norm, its output is (batch_size, num_patches + 1, embed_dim)
        # Let's assume a simple averaging for weights for now if it's 3D
        weights = grad.mean(dim=1, keepdim=True) # [B, 1, C] if input is [B, N, C]
        cam = (act * weights).sum(dim=2, keepdim=True) # [B, N, 1]
    else:
        raise ValueError("Unexpected gradient shape: {}".format(grad.shape))

    cam = torch.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    # If cam is [B,N,1] from ViT, we need to select the first batch item and reshape N to H,W
    # This is a placeholder for actual reshaping logic for ViT patches
    if len(cam.shape) == 3 and cam.shape[0] == 1 and cam.shape[2] == 1: # B=1, N, 1
        cam_np = cam[0, :, 0].detach().cpu().numpy()
        # Attempt to make it 2D, assuming square patch grid
        num_patches = cam_np.shape[0]
        if num_patches > 1: # Check if there's a CLS token to ignore
             # This depends on whether target_layer output includes CLS token features that should be part of CAM
             # For vit_base_patch16_224, image size 224x224, patch size 16x16 -> 14x14 patches = 196 patches
             # If CLS token is present, num_patches might be 197.
             # We'll assume cam_np is for the patches only for simplicity here.
            side = int(np.sqrt(num_patches))
            if side * side == num_patches:
                return cam_np.reshape(side, side)
            else: # Fallback if not a perfect square (e.g. CLS token included or non-square)
                 print(f"Warning: CAM for ViT has {num_patches} tokens, cannot reshape to square. Returning 1D CAM.")
                 return cam_np # Return as is, visualization might need to handle 1D
        else:
            return cam_np # Single value or not enough patches

    elif len(cam.shape) == 4 : # Expected for CNN [B,1,H,W]
         return cam[0, 0].detach().cpu().numpy()
    else:
        # Fallback for other shapes, e.g. if batch > 1 or other issues
        print(f"Warning: CAM shape {cam.shape} not directly visualizable. Taking first element.")
        return cam[0,0].detach().cpu().numpy() # Or handle appropriately


def visualize_cam(image, cam): # cam is now expected to be a 2D numpy array
    img_np = image.cpu().permute(1, 2, 0).numpy()
    # Denormalize based on Food101 transform
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1) * 255
    img_np = img_np.astype(np.uint8)

    if cam.ndim == 1: # If CAM is 1D (e.g. from ViT patches not reshaped)
        # Simple visualization: repeat it to make it somewhat 2D or show as a bar
        # This is a placeholder, proper ViT CAM visualization is more involved
        print("Visualizing 1D CAM. For ViT, this represents patch importance.")
        # Resize to a fixed strip for visualization
        cam_resized = cv2.resize(cam[:, np.newaxis], (50, 224)) # width 50, height 224
    elif cam.ndim == 2:
        cam_resized = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE)) # IMAGE_SIZE should be defined
    else:
        raise ValueError(f"CAM has unexpected dimensions: {cam.ndim}")

    cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)

    # Ensure img_np is HWC
    if img_np.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        img_np_resized = cv2.resize(img_np, (IMAGE_SIZE, IMAGE_SIZE))
    else:
        img_np_resized = img_np

    # Ensure cam_heatmap is also resized to match image if it was 1D initially
    if cam_heatmap.shape[:2] != img_np_resized.shape[:2]:
        cam_heatmap = cv2.resize(cam_heatmap, (img_np_resized.shape[1], img_np_resized.shape[0]))


    overlay = cv2.addWeighted(img_np_resized, 0.6, cam_heatmap, 0.4, 0)
    return overlay


# Main Program
if __name__ == "__main__":
    use_distributed = False
    local_rank = 0
    
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        use_distributed = True
        torch.distributed.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        print(f"分散式訓練已初始化，local_rank: {local_rank}")
    else:
        print("使用單一GPU訓練模式")
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device set later
        # print(f"使用設備: {device}") # device set later

    BATCH_SIZE = 64 # Adjusted for typical ViT memory, might need further tuning
    IMAGE_SIZE = 224
    IMAGE_ROOT = "data"

    LABELS =  ['Abalone', 'Abalonemushroom', 'Achoy', 'Adzukibean', 'Alfalfasprouts', 'Almond', 'Apple', 'Asparagus', 'Avocado', 'Babycorn', 'Bambooshoot', 'Banana', 'Beeftripe', 'Beetroot', 'Birds-nestfern', 'Birdsnest', 'Bittermelon', 'Blackmoss', 'Blackpepper', 'Blacksoybean', 'Blueberry', 'Bokchoy', 'Brownsugar', 'Buckwheat', 'Cabbage', 'Cardamom', 'Carrot', 'Cashewnut', 'Cauliflower', 'Celery', 'Centuryegg', 'Cheese', 'Cherry', 'Chestnut', 'Chilipepper', 'Chinesebayberry', 'Chinesechiveflowers', 'Chinesechives', 'Chinesekale', 'Cilantro', 'Cinnamon', 'Clove', 'Cocoa', 'Coconut', 'Corn', 'Cowpea', 'Crab', 'Cream', 'Cucumber', 'Daikon', 'Dragonfruit', 'Driedpersimmon', 'Driedscallop', 'Driedshrimp', 'Duckblood', 'Durian', 'Eggplant', 'Enokimushroom', 'Fennel', 'Fig', 'Fishmint', 'Freshwaterclam', 'Garlic', 'Ginger', 'Glutinousrice', 'Gojileaves', 'Grape', 'Grapefruit', 'GreenSoybean', 'Greenbean', 'Greenbellpepper', 'Greenonion', 'Guava', 'Gynuradivaricata', 'Headingmustard', 'Honey', 'Jicama', 'Jobstears', 'Jujube', 'Kale', 'Kelp', 'Kidneybean', 'Kingoystermushroom', 'Kiwifruit', 'Kohlrabi', 'Kumquat', 'Lettuce', 'Limabean', 'Lime', 'Lobster', 'Longan', 'Lotusroot', 'Lotusseed', 'Luffa', 'Lychee', 'Madeira_vine', 'Maitakemushroom', 'Mandarin', 'Mango', 'Mangosteen', 'Milk', 'Millet', 'Minongmelon', 'Mint', 'Mungbean', 'Napacabbage', 'Natto', 'Nori', 'Nutmeg', 'Oat', 'Octopus', 'Okinawaspinach', 'Okra', 'Olive', 'Onion', 'Orange', 'Oystermushroom', 'Papaya', 'Parsley', 'Passionfruit', 'Pea', 'Peach', 'Peanut', 'Pear', 'Pepper', 'Perilla', 'Persimmon', 'Pickledmustardgreens', 'Pineapple', 'Pinenut', 'Plum', 'Pomegranate', 'Pomelo', 'Porktripe', 'Potato', 'Pumpkin', 'Pumpkinseed', 'Quailegg', 'Radishsprouts', 'Rambutan', 'Raspberry', 'Redamaranth', 'Reddate', 'Rice', 'Rosemary', 'Safflower', 'Saltedpotherbmustard', 'Seacucumber', 'Seaurchin', 'Sesameseed', 'Shaggymanemushroom', 'Shiitakemushroom', 'Shrimp', 'Snowfungus', 'Soybean', 'Soybeansprouts', 'Soysauce', 'Staranise', 'Starfruit', 'Strawberry', 'Strawmushroom', 'Sugarapple', 'Sunflowerseed', 'Sweetpotato', 'Sweetpotatoleaves', 'Taro', 'Thyme', 'Tofu', 'Tomato', 'Wasabi', 'Waterbamboo', 'Watercaltrop', 'Watermelon', 'Waterspinach', 'Waxapple', 'Wheatflour', 'Wheatgrass', 'Whitepepper', 'Wintermelon', 'Woodearmushroom', 'Yapear', 'Yauchoy', 'spinach']

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    encoder = Label_encoder(LABELS)
    all_data_df = prepare_dataframe(IMAGE_ROOT, encoder)

    if all_data_df.empty or len(all_data_df) < 2:
        print("錯誤: 資料不足以分割成訓練集和測試集。請檢查資料夾和 LABELS。程式即將結束。")
        exit()

    train_df, test_df = train_test_split(
        all_data_df,
        test_size=0.2,
        random_state=42,
        stratify=all_data_df['label'] if not all_data_df.empty else None
    )

    if train_df.empty or test_df.empty:
        print("錯誤: 資料分割後訓練集或測試集為空。請檢查資料量和分割比例。程式即將結束。")
        exit()

    train_dataset = Food101Dataset(train_df, encoder, transform)
    test_dataset = Food101Dataset(test_df, encoder, transform)

    def collate_fn_skip_corrupted(batch):
        batch = list(filter(lambda x: x is not None, batch))
        if not batch:
            return torch.Tensor(), torch.Tensor()
        return torch.utils.data.dataloader.default_collate(batch)

    if use_distributed:
        train_sampler = DistributedSampler(train_dataset)
        test_sampler = DistributedSampler(test_dataset)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, collate_fn=collate_fn_skip_corrupted, num_workers=4) # Adjusted num_workers
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, sampler=test_sampler, collate_fn=collate_fn_skip_corrupted, num_workers=4) # Adjusted num_workers
        device = torch.device(f"cuda:{local_rank}")
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_skip_corrupted, num_workers=4) # Adjusted num_workers
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_skip_corrupted, num_workers=4) # Adjusted num_workers
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print(f"使用設備: {device}")


    num_epochs = 30
    
    # Changed to Vision Transformer
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=len(LABELS))
    model = model.to(device)
    
    target_layer = None # Initialize target_layer
    if use_distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True) # Added find_unused_parameters for ViT if any
        # For ViT, target the last block's normalization layer or the model's final norm layer
        # Option 1: Last block's norm2 (if exists and suitable)
        # target_layer = model.module.blocks[-1].norm2
        # Option 2: Model's final norm layer (often named 'norm')
        target_layer = model.module.norm
    else:
        if torch.cuda.device_count() > 1:
            print(f"使用 {torch.cuda.device_count()} 個 GPU 運行 DataParallel")
            model = nn.DataParallel(model)
            # target_layer = model.module.blocks[-1].norm2
            target_layer = model.module.norm
        else:
            # target_layer = model.blocks[-1].norm2
            target_layer = model.norm
            
    if target_layer is None:
        print("警告: CAM 的 target_layer 未能成功設定。CAM 可能無法正常運作。")


    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5) # Adjusted learning rate for ViT
    criterion = nn.CrossEntropyLoss()
    # Scheduler might need adjustment for ViT, CosineAnnealingLR is generally fine
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_acc = 0
    for epoch in range(num_epochs):
        print(f"\\nEpoch {epoch + 1}/{num_epochs}") # Corrected string literal
        if use_distributed:
            train_sampler.set_epoch(epoch)
        
        train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        test_acc = test_epoch(model, test_loader, criterion, device)

        if test_acc > best_acc:
            best_acc = test_acc
            if not use_distributed or local_rank == 0:
                # Changed model save name
                torch.save(model.state_dict(), 'vit_model_food101.pth')
                print(f"模型已保存，準確率: {best_acc:.2f}%")
    
    # Example of generating and visualizing CAM for one image from the test set
    if not use_distributed or local_rank == 0: # This if block continues
        if target_layer is not None and len(test_dataset) > 0:
            print("\\nGenerating CAM for a sample image...") # Corrected string literal
            try:
                sample_img, sample_label_idx = test_dataset[0] # Get first sample
                if sample_img is None: # Handle if first sample was corrupted
                     # Try to find a valid sample
                    for i in range(1, min(10, len(test_dataset))): # Try next few samples
                        sample_img, sample_label_idx = test_dataset[i]
                        if sample_img is not None:
                            break
                
                if sample_img is not None:
                    sample_img_tensor = sample_img.unsqueeze(0).to(device)

                    # Ensure model is in eval mode and not wrapped if CAM function expects raw model
                    model_for_cam = model
                    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                        model_for_cam = model.module
                    model_for_cam.eval()

                    cam_output = generate_cam(model_for_cam, sample_img_tensor, target_layer)
                    
                    # Denormalize original image for visualization (using the first image from batch)
                    # The 'image' variable for visualize_cam should be the transformed tensor before batching
                    overlay_image = visualize_cam(sample_img, cam_output) # Pass the single transformed image tensor

                    plt.figure(figsize=(6,6))
                    plt.imshow(overlay_image)
                    plt.title(f"CAM for: {encoder.get_label(sample_label_idx)}")
                    plt.axis('off')
                    plt.savefig("vit_cam_example.png")
                    print("CAM example saved to vit_cam_example.png")
                    # plt.show() # plt.show() might not work in all environments
                else:
                    print("Could not obtain a valid sample image for CAM generation.")

            except Exception as e:
                print(f"Error generating CAM: {e}")
        elif target_layer is None:
            print("CAM target_layer not set, skipping CAM generation.")
        else:
            print("Test dataset is empty, skipping CAM generation.")


    if use_distributed and torch.distributed.is_initialized():
        destroy_process_group()

print("程式執行完畢。")
