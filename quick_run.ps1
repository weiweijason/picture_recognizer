# Windows PowerShell 快速執行腳本
# 用法: .\quick_run.ps1 [選項]

param(
    [string]$Action = "all"
)

# 設定工作目錄
$ProjectRoot = "C:\Users\User\OneDrive - National ChengChi University\113-2 Design\project"

# 顏色設定
function Write-ColorText {
    param(
        [string]$Text,
        [string]$Color = "White"
    )
    Write-Host $Text -ForegroundColor $Color
}

# 顯示使用方法
function Show-Usage {
    Write-ColorText "`n使用方法:" "Blue"
    Write-Host "  .\quick_run.ps1 [選項]"
    Write-ColorText "`n可用選項:" "Blue"
    Write-Host "  efficientnet  - 只執行 EfficientNet"
    Write-Host "  densenet      - 只執行 DenseNet"
    Write-Host "  convnext      - 只執行 ConvNeXt"
    Write-Host "  all           - 執行所有模型 (預設)"
    Write-Host "  check         - 檢查環境"
    Write-Host "  test          - 測試修復是否成功"
    Write-ColorText "`n範例:" "Blue"
    Write-Host "  .\quick_run.ps1 efficientnet"
    Write-Host "  .\quick_run.ps1 all"
    Write-Host "  .\quick_run.ps1 test"
}

# 檢查環境
function Test-Environment {
    Write-ColorText "正在檢查環境..." "Yellow"
    
    try {
        $pythonVersion = python --version 2>$null
        Write-ColorText "✓ Python: $pythonVersion" "Green"
        
        $torchCheck = python -c "import torch; print(f'PyTorch {torch.__version__}')" 2>$null
        Write-ColorText "✓ $torchCheck" "Green"
        
        $cudaCheck = python -c "import torch; print(f'CUDA 可用: {torch.cuda.is_available()}')" 2>$null
        Write-ColorText "✓ $cudaCheck" "Green"
        
    } catch {
        Write-ColorText "✗ 環境檢查失敗" "Red"
    }
}

# 執行單一模型
function Start-SingleModel {
    param([string]$ModelName)
    
    $scriptPaths = @{
        "efficientnet" = "$ProjectRoot\efficientnet\efficientnet_dual_gpu_train.py"
        "densenet"     = "$ProjectRoot\DenseNet\densenet_dual_gpu_train.py"
        "convnext"     = "$ProjectRoot\ConvNeXt\convnext_dual_gpu_train.py"
    }
    
    if (-not $scriptPaths.ContainsKey($ModelName)) {
        Write-ColorText "✗ 未知模型: $ModelName" "Red"
        return $false
    }
    
    $scriptPath = $scriptPaths[$ModelName]
    
    if (-not (Test-Path $scriptPath)) {
        Write-ColorText "✗ 找不到腳本: $scriptPath" "Red"
        return $false
    }
    
    Write-ColorText "正在執行 $ModelName..." "Green"
    $dirPath = Split-Path $scriptPath -Parent
    Set-Location $dirPath
    
    try {
        python (Split-Path $scriptPath -Leaf)
        $exitCode = $LASTEXITCODE
        
        if ($exitCode -eq 0) {
            Write-ColorText "✓ $ModelName 執行成功！" "Green"
            return $true
        } else {
            Write-ColorText "✗ $ModelName 執行失敗！" "Red"
            return $false
        }
    } catch {
        Write-ColorText "✗ $ModelName 執行異常！" "Red"
        return $false
    }
}

# 執行所有模型
function Start-AllModels {
    Write-ColorText "執行所有模型訓練..." "Green"
    
    $models = @("efficientnet", "densenet", "convnext")
    $results = @{}
    
    foreach ($model in $models) {
        $results[$model] = Start-SingleModel -ModelName $model
        Write-Host ""
    }
    
    # 顯示總結
    Write-ColorText "=====================================" "Cyan"
    Write-ColorText "執行結果總結" "Cyan"
    Write-ColorText "=====================================" "Cyan"
    
    $successCount = 0
    foreach ($result in $results.GetEnumerator()) {
        if ($result.Value) {
            Write-ColorText "✓ $($result.Key)" "Green"
            $successCount++
        } else {
            Write-ColorText "✗ $($result.Key)" "Red"
        }
    }
    
    Write-ColorText "`n成功: $successCount / $($models.Count)" $(if ($successCount -eq $models.Count) { "Green" } else { "Yellow" })
}

# 測試修復
function Test-Fix {
    Write-ColorText "測試深度學習腳本修復..." "Yellow"
    
    $testScript = "$ProjectRoot\test_fix.ps1"
    if (Test-Path $testScript) {
        & $testScript
    } else {
        Write-ColorText "✗ 找不到測試腳本: $testScript" "Red"
    }
}

# 主程式
switch ($Action.ToLower()) {
    "check" {
        Test-Environment
    }
    "test" {
        Test-Fix
    }
    "all" {
        Start-AllModels
    }
    "efficientnet" {
        Start-SingleModel -ModelName "efficientnet"
    }
    "densenet" {
        Start-SingleModel -ModelName "densenet"
    }
    "convnext" {
        Start-SingleModel -ModelName "convnext"
    }
    "help" {
        Show-Usage
    }
    "-h" {
        Show-Usage
    }
    "--help" {
        Show-Usage
    }
    default {
        Write-ColorText "✗ 未知選項: $Action" "Red"
        Show-Usage
        exit 1
    }
}

# 回到專案根目錄
Set-Location $ProjectRoot
