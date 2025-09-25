# Deep Learning Image Classification Project

這是一個使用多種深度學習架構進行圖像分類的專案，支援分散式訓練和GPU加速。

## 📁 專案結構

```
project/
├── CNN/                    # CNN 模型訓練
├── VIT/                    # Vision Transformer 模型
├── ConvNeXt/              # ConvNeXt 模型訓練
├── DenseNet/              # DenseNet 模型訓練
├── efficientnet/          # EfficientNet 模型訓練
├── VGG/                   # VGG 模型訓練
├── swin_transformers/     # Swin Transformer v1
├── swin_transformer_2/    # Swin Transformer v2
├── model/                 # 訓練好的模型檔案
├── run_models_sequential.* # 批次執行腳本
├── quick_run.*            # 快速執行腳本
└── install_dependencies.* # 環境安裝腳本
```

## 🚀 快速開始

### 環境安裝

**Linux/Mac:**
```bash
chmod +x install_dependencies.sh
./install_dependencies.sh
```

**Windows:**
```powershell
.\quick_run.ps1
```

### 資料準備

將圖像資料放置在 `data/` 資料夾中，結構如下：
```
data/
├── class1/
│   ├── image1.jpg
│   └── image2.png
├── class2/
│   ├── imageA.jpg
│   └── imageB.png
└── ...
```

### 訓練模型

**單一模型訓練:**
```bash
cd VIT/
python vit_model.py
```

**批次訓練多個模型:**
```bash
chmod +x run_models_sequential.sh
./run_models_sequential.sh
```

## 📊 支援的模型架構

| 模型 | 檔案位置 | 特色 |
|------|----------|------|
| Vision Transformer | `VIT/vit_model.py` | Transformer架構，支援CAM視覺化 |
| ConvNeXt | `ConvNeXt/convnext_dual_gpu_train.py` | 現代CNN架構 |
| DenseNet | `DenseNet/densenet_dual_gpu_train.py` | 密集連接網路 |
| EfficientNet | `efficientnet/efficientnet_dual_gpu_train.py` | 高效率CNN |
| Swin Transformer | `swin_transformers/swinv1.py` | 階層式Transformer |
| VGG | `VGG/vgg.py` | 經典CNN架構 |
| ResNet/CNN | `CNN/cnn_gpu_tensorboard_train.py` | 支援TensorBoard |

## 🔧 功能特色

- ✅ **多GPU支援**: 支援單GPU、DataParallel和DistributedDataParallel訓練
- ✅ **視覺化**: CAM (Class Activation Mapping) 熱力圖生成
- ✅ **TensorBoard**: 訓練過程視覺化
- ✅ **自動化腳本**: 批次訓練和環境檢查
- ✅ **錯誤處理**: 自動跳過損壞的圖像檔案
- ✅ **模型保存**: 自動保存最佳模型

## 🖥️ 系統需求

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.3+ (GPU訓練)
- 8GB+ GPU記憶體 (建議)

### 主要依賴套件

```
torch
torchvision
timm
transformers
opencv-python
matplotlib
scikit-learn
tensorboard
```

## 📈 訓練結果

已訓練的模型檔案位於 `model/` 資料夾：

- `swinv2_food101_best_94.pth` - Swin Transformer V2 (94% 準確率)
- `convnext_best_model_90.pth` - ConvNeXt (90% 準確率)
- `densenet_best_86.pth` - DenseNet (86% 準確率)
- 其他模型檔案...

## 🐛 疑難排解

### 常見問題

1. **記憶體不足**: 調整 `BATCH_SIZE` 參數
2. **CUDA錯誤**: 檢查CUDA版本相容性
3. **圖像載入錯誤**: 檢查圖像檔案格式和完整性

### 修復腳本

**測試修復:**
```bash
# Linux
./test_fix.sh

# Windows
.\test_fix.ps1
```

## 📝 更新日誌

詳細的錯誤修復和更新記錄請參考 `FIX_SUMMARY.md`。

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！

## 📄 授權

MIT License