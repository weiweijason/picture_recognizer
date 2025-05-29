#!/bin/bash

# 測試修復後的深度學習訓練腳本
# 此腳本會執行簡短的測試以驗證修復是否成功

PROJECT_ROOT="/home/user/project"  # 請根據實際路徑修改

echo "======================================"
echo "深度學習腳本修復驗證測試"
echo "======================================"

# 檢查環境
echo "1. 檢查 Python 和 PyTorch..."
python3 -c "
import torch
import torchvision
print(f'✓ PyTorch {torch.__version__} 已安裝')
print(f'✓ Torchvision {torchvision.__version__} 已安裝')
print(f'✓ CUDA 可用: {torch.cuda.is_available()}')
"

# 創建測試資料目錄（如果不存在）
echo ""
echo "2. 檢查測試資料..."
cd "$PROJECT_ROOT"

# 檢查是否有 data 目錄
if [ ! -d "data" ]; then
    echo "⚠️  未找到 data/ 目錄，將創建示例測試資料..."
    mkdir -p data/test_class1 data/test_class2
    
    # 創建一些測試圖片（使用 Python 生成）
    python3 -c "
import os
from PIL import Image
import numpy as np

# 創建測試圖片
for class_name in ['test_class1', 'test_class2']:
    class_dir = f'data/{class_name}'
    os.makedirs(class_dir, exist_ok=True)
    
    for i in range(5):  # 每個類別5張測試圖片
        # 創建隨機彩色圖片
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(f'{class_dir}/test_image_{i}.jpg')
    
    print(f'✓ 已為 {class_name} 創建 5 張測試圖片')
"
else
    echo "✓ 找到 data/ 目錄"
fi

# 測試每個模型的資料載入部分
echo ""
echo "3. 測試 ConvNeXt 資料載入..."
cd "$PROJECT_ROOT/ConvNeXt"
timeout 30 python3 -c "
import sys
sys.path.append('..')
exec(open('convnext_dual_gpu_train.py').read().split('if __name__')[0])
print('✓ ConvNeXt 資料載入測試成功')
" 2>/dev/null && echo "✓ ConvNeXt 資料載入正常" || echo "✗ ConvNeXt 資料載入失敗"

echo ""
echo "4. 測試 EfficientNet 資料載入..."
cd "$PROJECT_ROOT/efficientnet"
timeout 30 python3 -c "
import sys
sys.path.append('..')
exec(open('efficientnet_dual_gpu_train.py').read().split('if __name__')[0])
print('✓ EfficientNet 資料載入測試成功')
" 2>/dev/null && echo "✓ EfficientNet 資料載入正常" || echo "✗ EfficientNet 資料載入失敗"

echo ""
echo "5. 測試 DenseNet 資料載入..."
cd "$PROJECT_ROOT/DenseNet"
timeout 30 python3 -c "
import sys
sys.path.append('..')
exec(open('densenet_dual_gpu_train.py').read().split('if __name__')[0])
print('✓ DenseNet 資料載入測試成功')
" 2>/dev/null && echo "✓ DenseNet 資料載入正常" || echo "✗ DenseNet 資料載入失敗"

echo ""
echo "======================================"
echo "測試完成"
echo "======================================"
echo ""
echo "如果所有測試都顯示 ✓，表示修復成功！"
echo "您現在可以運行完整的訓練："
echo "  ./quick_run.sh convnext"
echo "  ./quick_run.sh efficientnet"
echo "  ./quick_run.sh densenet"
echo "  ./run_models_sequential.sh"
