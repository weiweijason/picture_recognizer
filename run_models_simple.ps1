# 簡化版 PowerShell 腳本：依序執行深度學習模型訓練

# 設定工作目錄
$ProjectRoot = "C:\Users\User\OneDrive - National ChengChi University\113-2 Design\project"

Write-Host "開始依序執行深度學習模型訓練..." -ForegroundColor Green

# 1. 執行 EfficientNet
Write-Host "`n[1/3] 執行 EfficientNet..." -ForegroundColor Yellow
Set-Location "$ProjectRoot\efficientnet"
python "efficientnet_dual_gpu_train.py"

# 2. 執行 DenseNet  
Write-Host "`n[2/3] 執行 DenseNet..." -ForegroundColor Yellow
Set-Location "$ProjectRoot\DenseNet"
python "densenet_dual_gpu_train.py"

# 3. 執行 ConvNeXt
Write-Host "`n[3/3] 執行 ConvNeXt..." -ForegroundColor Yellow
Set-Location "$ProjectRoot\ConvNeXt"
python "convnext_dual_gpu_train.py"

# 回到原始目錄
Set-Location $ProjectRoot

Write-Host "`n所有模型訓練完成！" -ForegroundColor Green
Write-Host "按任意鍵退出..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
