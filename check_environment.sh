#!/bin/bash

# 遠端Linux主機環境檢查腳本
# 在執行訓練腳本之前使用此腳本檢查環境

echo "======================================"
echo "深度學習環境檢查腳本"
echo "======================================"

# 檢查 Python 版本
echo "1. 檢查 Python 版本:"
python --version
python3 --version 2>/dev/null || echo "Python3 未安裝"

# 檢查 GPU 資訊
echo ""
echo "2. 檢查 GPU 資訊:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    echo "GPU 數量: $(nvidia-smi --list-gpus | wc -l)"
else
    echo "未檢測到 NVIDIA GPU 或 nvidia-smi 未安裝"
fi

# 檢查 PyTorch 安裝
echo ""
echo "3. 檢查 PyTorch 安裝:"
python -c "
import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA 版本: {torch.version.cuda}')
    print(f'可用 GPU 數量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
" 2>/dev/null || echo "PyTorch 未正確安裝"

# 檢查其他必要套件
echo ""
echo "4. 檢查必要套件:"
python -c "
packages = ['torchvision', 'tensorboard', 'PIL', 'numpy']
for pkg in packages:
    try:
        if pkg == 'PIL':
            import PIL
            print(f'✓ {pkg} 已安裝 (版本: {PIL.__version__})')
        else:
            module = __import__(pkg)
            version = getattr(module, '__version__', '未知版本')
            print(f'✓ {pkg} 已安裝 (版本: {version})')
    except ImportError:
        print(f'✗ {pkg} 未安裝')
" 2>/dev/null || echo "套件檢查失敗"

# 檢查記憶體
echo ""
echo "5. 檢查系統記憶體:"
free -h

# 檢查 CPU 資訊
echo ""
echo "6. 檢查 CPU 資訊:"
echo "CPU 核心數: $(nproc)"
echo "CPU 資訊: $(lscpu | grep 'Model name' | cut -d':' -f2 | xargs)"

# 檢查磁碟空間
echo ""
echo "7. 檢查磁碟空間:"
df -h | head -1
df -h | grep -E "/$|/home|/data" | head -5

# 檢查目錄結構
echo ""
echo "8. 檢查專案目錄結構:"
PROJECT_ROOT="/home/user/project"  # 與主腳本一致
if [ -d "$PROJECT_ROOT" ]; then
    echo "專案根目錄: $PROJECT_ROOT ✓"
    echo "目錄結構:"
    find "$PROJECT_ROOT" -name "*.py" | head -10
else
    echo "專案根目錄不存在: $PROJECT_ROOT ✗"
    echo "請檢查路徑設定"
fi

echo ""
echo "======================================"
echo "環境檢查完成"
echo "======================================"
