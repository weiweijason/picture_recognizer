#!/bin/bash

# 遠端Linux主機依賴套件安裝腳本

echo "======================================"
echo "深度學習依賴套件安裝腳本"
echo "======================================"

# 檢查是否為 root 或有 sudo 權限
if [[ $EUID -eq 0 ]]; then
    SUDO=""
else
    SUDO="sudo"
    echo "將使用 sudo 安裝系統套件..."
fi

# 更新套件列表
echo "1. 更新套件列表..."
$SUDO apt update

# 安裝系統依賴
echo ""
echo "2. 安裝系統依賴套件..."
$SUDO apt install -y python3 python3-pip python3-dev build-essential

# 升級 pip
echo ""
echo "3. 升級 pip..."
python3 -m pip install --upgrade pip

# 安裝 PyTorch (CUDA 版本)
echo ""
echo "4. 安裝 PyTorch (CUDA 版本)..."
echo "請選擇 CUDA 版本:"
echo "1) CUDA 11.8"
echo "2) CUDA 12.1"
echo "3) CPU 版本"
read -p "請輸入選項 (1-3): " cuda_choice

case $cuda_choice in
    1)
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ;;
    2)
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ;;
    3)
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        ;;
    *)
        echo "無效選項，安裝 CUDA 11.8 版本..."
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ;;
esac

# 安裝其他必要套件
echo ""
echo "5. 安裝其他必要套件..."
pip3 install tensorboard pillow numpy matplotlib seaborn

# 安裝額外的有用套件
echo ""
echo "6. 安裝額外套件..."
pip3 install tqdm scikit-learn pandas

# 驗證安裝
echo ""
echo "7. 驗證安裝..."
python3 -c "
import torch
import torchvision
import tensorboard
from PIL import Image
import numpy as np

print('✓ 所有套件安裝成功！')
print(f'PyTorch 版本: {torch.__version__}')
print(f'Torchvision 版本: {torchvision.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU 數量: {torch.cuda.device_count()}')
"

echo ""
echo "======================================"
echo "依賴套件安裝完成！"
echo "======================================"
echo ""
echo "接下來您可以："
echo "1. 執行環境檢查: ./check_environment.sh"
echo "2. 執行單一模型: ./quick_run.sh efficientnet"
echo "3. 執行所有模型: ./quick_run.sh all"
