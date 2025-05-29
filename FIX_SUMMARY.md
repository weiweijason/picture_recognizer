# 深度學習訓練腳本錯誤修復總結

## 🔧 問題描述

執行 ConvNeXt 訓練時遇到 `AttributeError` 錯誤：

```
AttributeError: Caught AttributeError in DataLoader worker process 0.
File "convnext_dual_gpu_train.py", line 94, in __getitem__
    x = self.transform(x)
...
File "/usr/local/lib/python3.8/dist-packages/torchvision/transforms/transforms.py", line 1731, in forward
    if value is not None and not (len(value) in (1, img.shape[-3]))
```

## 🎯 根本原因

錯誤發生在資料轉換管道中的正規化步驟，系統期望3通道 (RGB) 圖像，但可能接收到了不同通道數的圖像。

**原始問題架構：**
1. `ImageFolder` 使用 `base_transform` 進行 RGB 轉換
2. `DatasetWithTransform` 再次應用訓練/驗證轉換
3. 雙重轉換管道可能導致通道數不匹配

## ✅ 修復方案

### 修復內容

1. **簡化轉換管道**：
   - 移除 `ImageFolder` 中的 `base_transform`
   - 在每個訓練/驗證轉換的開頭加入 RGB 轉換

2. **修改的檔案**：
   - `ConvNeXt/convnext_dual_gpu_train.py`
   - `efficientnet/efficientnet_dual_gpu_train.py`
   - `DenseNet/densenet_dual_gpu_train.py`

### 具體修改

**原始程式碼：**
```python
# 基礎轉換
base_transform = transforms.Compose([
    transforms.Lambda(lambda x: x.convert('RGB'))
])

# 資料集建立
full_dataset_pil = datasets.ImageFolder(
    data_dir,
    transform=base_transform,
    is_valid_file=is_valid_image_file
)

# 訓練轉換（缺少RGB轉換）
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        # ... 其他轉換
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
```

**修復後程式碼：**
```python
# 移除基礎轉換，改為None
full_dataset_pil = datasets.ImageFolder(
    data_dir,
    transform=None,
    is_valid_file=is_valid_image_file
)

# 在每個轉換開頭加入RGB轉換
data_transforms = {
    'train': transforms.Compose([
        transforms.Lambda(lambda x: x.convert('RGB')),  # 確保RGB格式
        transforms.RandomResizedCrop(image_size),
        # ... 其他轉換
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
```

## 🚀 新增工具

### 1. 測試腳本

**Linux:**
```bash
chmod +x test_fix.sh
./test_fix.sh
```

**Windows:**
```powershell
.\test_fix.ps1
```

### 2. 增強的快速執行腳本

**Linux:**
```bash
./quick_run.sh test        # 測試修復
./quick_run.sh check       # 檢查環境
./quick_run.sh efficientnet # 執行單一模型
./quick_run.sh all         # 執行所有模型
```

**Windows:**
```powershell
.\quick_run.ps1 test       # 測試修復
.\quick_run.ps1 check      # 檢查環境
.\quick_run.ps1 efficientnet # 執行單一模型
.\quick_run.ps1 all        # 執行所有模型
```

## 📋 驗證步驟

### 1. 環境檢查
```bash
# Linux
./quick_run.sh check

# Windows
.\quick_run.ps1 check
```

### 2. 測試修復
```bash
# Linux
./quick_run.sh test

# Windows
.\quick_run.ps1 test
```

### 3. 執行訓練
如果測試通過，可以開始正式訓練：

```bash
# Linux - 執行所有模型
./run_models_sequential.sh

# Linux - 執行單一模型
./quick_run.sh convnext

# Windows - 執行所有模型
.\run_models_sequential.ps1

# Windows - 執行單一模型
.\quick_run.ps1 convnext
```

## 🔍 背景執行建議

### Linux 遠端主機
```bash
# 使用 nohup
nohup ./run_models_sequential.sh > training.log 2>&1 &

# 使用 screen
screen -S training
./run_models_sequential.sh

# 使用 tmux
tmux new-session -s training
./run_models_sequential.sh
```

### Windows
```powershell
# 使用 Start-Job
$job = Start-Job -ScriptBlock { 
    Set-Location "C:\Users\User\OneDrive - National ChengChi University\113-2 Design\project"
    .\run_models_sequential.ps1 
}

# 檢查狀態
Get-Job
Receive-Job $job
```

## 📊 預期結果

修復後，所有深度學習模型應該能夠：

1. ✅ 正確載入和處理圖像資料
2. ✅ 正常執行雙 GPU 訓練
3. ✅ 使用 64 個 worker 進行高效資料載入
4. ✅ 生成 TensorBoard 日誌
5. ✅ 儲存訓練好的模型

## 🛠 故障排除

如果仍有問題：

1. **檢查 CUDA 版本匹配**
2. **確認資料目錄結構正確**
3. **驗證 PIL/Pillow 安裝**
4. **檢查記憶體是否充足**
5. **確認檔案權限設定**

## 📝 變更日誌

- **2025-05-29**: 修復 AttributeError 錯誤
- **2025-05-29**: 簡化資料轉換管道
- **2025-05-29**: 新增測試和驗證腳本
- **2025-05-29**: 增強快速執行功能

---

**修復完成！** 🎉 現在可以正常執行深度學習訓練了。
