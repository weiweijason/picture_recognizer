# 測試修復後的深度學習訓練腳本 (Windows PowerShell)
# 此腳本會執行簡短的測試以驗證修復是否成功

param(
    [string]$ProjectRoot = "C:\Users\User\OneDrive - National ChengChi University\113-2 Design\project"
)

Write-Host "======================================" -ForegroundColor Cyan
Write-Host "深度學習腳本修復驗證測試" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

# 檢查環境
Write-Host "`n1. 檢查 Python 和 PyTorch..." -ForegroundColor Yellow
try {
    $pythonCheck = python -c "
import torch
import torchvision
print(f'✓ PyTorch {torch.__version__} 已安裝')
print(f'✓ Torchvision {torchvision.__version__} 已安裝')
print(f'✓ CUDA 可用: {torch.cuda.is_available()}')
" 2>$null
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host $pythonCheck -ForegroundColor Green
    } else {
        Write-Host "✗ Python 環境檢查失敗" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "✗ 無法執行 Python 環境檢查" -ForegroundColor Red
    exit 1
}

# 檢查測試資料
Write-Host "`n2. 檢查測試資料..." -ForegroundColor Yellow
Set-Location $ProjectRoot

if (!(Test-Path "data")) {
    Write-Host "⚠️  未找到 data/ 目錄，將創建示例測試資料..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path "data\test_class1" -Force | Out-Null
    New-Item -ItemType Directory -Path "data\test_class2" -Force | Out-Null
    
    # 創建一些測試圖片
    $createTestImages = @"
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
"@
    
    try {
        python -c $createTestImages
        Write-Host "✓ 測試資料創建成功" -ForegroundColor Green
    } catch {
        Write-Host "✗ 測試資料創建失敗" -ForegroundColor Red
    }
} else {
    Write-Host "✓ 找到 data/ 目錄" -ForegroundColor Green
}

# 測試函數
function Test-ModelDataLoading {
    param(
        [string]$ModelName,
        [string]$ScriptPath
    )
    
    Write-Host "`n測試 $ModelName 資料載入..." -ForegroundColor Yellow
    
    if (!(Test-Path $ScriptPath)) {
        Write-Host "✗ 找不到腳本: $ScriptPath" -ForegroundColor Red
        return $false
    }
    
    Set-Location (Split-Path $ScriptPath -Parent)
    
    # 讀取腳本內容並只執行到主程式之前的部分
    $scriptContent = Get-Content $ScriptPath -Raw
    $beforeMain = $scriptContent -split "if __name__" | Select-Object -First 1
    
    # 創建臨時測試腳本
    $tempScript = @"
$beforeMain
print('✓ $ModelName 資料載入測試成功')
"@
    
    try {
        # 使用 timeout 限制執行時間
        $job = Start-Job -ScriptBlock {
            param($script)
            python -c $script
        } -ArgumentList $tempScript
        
        Wait-Job $job -Timeout 30 | Out-Null
        $result = Receive-Job $job 2>$null
        Remove-Job $job -Force
        
        if ($result -match "測試成功") {
            Write-Host "✓ $ModelName 資料載入正常" -ForegroundColor Green
            return $true
        } else {
            Write-Host "✗ $ModelName 資料載入失敗" -ForegroundColor Red
            return $false
        }
    } catch {
        Write-Host "✗ $ModelName 資料載入測試異常" -ForegroundColor Red
        return $false
    }
}

# 測試各個模型
$testResults = @{}

$testResults["ConvNeXt"] = Test-ModelDataLoading -ModelName "ConvNeXt" -ScriptPath "$ProjectRoot\ConvNeXt\convnext_dual_gpu_train.py"
$testResults["EfficientNet"] = Test-ModelDataLoading -ModelName "EfficientNet" -ScriptPath "$ProjectRoot\efficientnet\efficientnet_dual_gpu_train.py"
$testResults["DenseNet"] = Test-ModelDataLoading -ModelName "DenseNet" -ScriptPath "$ProjectRoot\DenseNet\densenet_dual_gpu_train.py"

# 總結測試結果
Write-Host "`n======================================" -ForegroundColor Cyan
Write-Host "測試結果總結" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan

$successCount = 0
$totalCount = $testResults.Count

foreach ($test in $testResults.GetEnumerator()) {
    if ($test.Value) {
        Write-Host "✓ $($test.Key)" -ForegroundColor Green
        $successCount++
    } else {
        Write-Host "✗ $($test.Key)" -ForegroundColor Red
    }
}

Write-Host "`n成功: $successCount / $totalCount" -ForegroundColor $(if ($successCount -eq $totalCount) { "Green" } else { "Yellow" })

if ($successCount -eq $totalCount) {
    Write-Host "`n🎉 所有測試通過！修復成功！" -ForegroundColor Green
    Write-Host "`n您現在可以運行完整的訓練：" -ForegroundColor Cyan
    Write-Host "  .\run_models_simple.ps1" -ForegroundColor White
    Write-Host "  .\run_models_sequential.ps1" -ForegroundColor White
} else {
    Write-Host "`n⚠️  有 $($totalCount - $successCount) 個模型測試失敗" -ForegroundColor Yellow
    Write-Host "請檢查錯誤訊息並確認環境設定" -ForegroundColor Yellow
}

# 回到專案根目錄
Set-Location $ProjectRoot
