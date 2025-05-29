# PowerShell 腳本：依序執行深度學習模型訓練
# 執行順序：EfficientNet -> DenseNet -> ConvNeXt

# 設定工作目錄
$ProjectRoot = "C:\Users\User\OneDrive - National ChengChi University\113-2 Design\project"

# 定義要執行的Python檔案清單
$PythonFiles = @(
    @{
        Name = "EfficientNet"
        Path = "$ProjectRoot\efficientnet\efficientnet_dual_gpu_train.py"
        WorkDir = "$ProjectRoot\efficientnet"
    },
    @{
        Name = "DenseNet"
        Path = "$ProjectRoot\DenseNet\densenet_dual_gpu_train.py"
        WorkDir = "$ProjectRoot\DenseNet"
    },
    @{
        Name = "ConvNeXt"
        Path = "$ProjectRoot\ConvNeXt\convnext_dual_gpu_train.py"
        WorkDir = "$ProjectRoot\ConvNeXt"
    }
)

# 記錄開始時間
$StartTime = Get-Date
Write-Host "=====================================" -ForegroundColor Green
Write-Host "開始執行深度學習模型訓練腳本" -ForegroundColor Green
Write-Host "開始時間: $StartTime" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green

# 初始化結果記錄
$Results = @()

# 依序執行每個Python檔案
for ($i = 0; $i -lt $PythonFiles.Count; $i++) {
    $File = $PythonFiles[$i]
    $CurrentTime = Get-Date
    
    Write-Host ""
    Write-Host "[$($i+1)/$($PythonFiles.Count)] 執行 $($File.Name)" -ForegroundColor Yellow
    Write-Host "檔案路徑: $($File.Path)" -ForegroundColor Cyan
    Write-Host "開始時間: $CurrentTime" -ForegroundColor Cyan
    Write-Host "=====================================" -ForegroundColor Yellow
    
    # 檢查檔案是否存在
    if (-not (Test-Path $File.Path)) {
        Write-Host "錯誤: 找不到檔案 $($File.Path)" -ForegroundColor Red
        $Results += @{
            Name = $File.Name
            Status = "失敗"
            Error = "檔案不存在"
            StartTime = $CurrentTime
            EndTime = Get-Date
            Duration = "N/A"
        }
        continue
    }
    
    # 切換到對應的工作目錄
    Set-Location $File.WorkDir
    Write-Host "切換到工作目錄: $($File.WorkDir)" -ForegroundColor Cyan
    
    try {
        # 執行Python檔案
        $ProcessStartTime = Get-Date
        Write-Host "正在執行: python $($File.Path)" -ForegroundColor Green
        
        # 使用 Start-Process 來執行Python，並等待完成
        $Process = Start-Process -FilePath "python" -ArgumentList $File.Path -NoNewWindow -Wait -PassThru
        
        $ProcessEndTime = Get-Date
        $Duration = $ProcessEndTime - $ProcessStartTime
        
        if ($Process.ExitCode -eq 0) {
            Write-Host "$($File.Name) 執行完成！" -ForegroundColor Green
            Write-Host "耗時: $($Duration.ToString('hh\:mm\:ss'))" -ForegroundColor Green
            
            $Results += @{
                Name = $File.Name
                Status = "成功"
                Error = $null
                StartTime = $ProcessStartTime
                EndTime = $ProcessEndTime
                Duration = $Duration.ToString('hh\:mm\:ss')
            }
        } else {
            Write-Host "$($File.Name) 執行失敗！退出代碼: $($Process.ExitCode)" -ForegroundColor Red
            
            $Results += @{
                Name = $File.Name
                Status = "失敗"
                Error = "退出代碼: $($Process.ExitCode)"
                StartTime = $ProcessStartTime
                EndTime = $ProcessEndTime
                Duration = $Duration.ToString('hh\:mm\:ss')
            }
        }
    }
    catch {
        $ProcessEndTime = Get-Date
        $Duration = $ProcessEndTime - $ProcessStartTime
        
        Write-Host "執行 $($File.Name) 時發生錯誤: $($_.Exception.Message)" -ForegroundColor Red
        
        $Results += @{
            Name = $File.Name
            Status = "失敗"
            Error = $_.Exception.Message
            StartTime = $ProcessStartTime
            EndTime = $ProcessEndTime
            Duration = $Duration.ToString('hh\:mm\:ss')
        }
    }
    
    Write-Host "=====================================" -ForegroundColor Yellow
}

# 回到原始目錄
Set-Location $ProjectRoot

# 總結報告
$EndTime = Get-Date
$TotalDuration = $EndTime - $StartTime

Write-Host ""
Write-Host "=====================================" -ForegroundColor Green
Write-Host "執行完成！總結報告" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green
Write-Host "總開始時間: $StartTime" -ForegroundColor Cyan
Write-Host "總結束時間: $EndTime" -ForegroundColor Cyan
Write-Host "總耗時: $($TotalDuration.ToString('hh\:mm\:ss'))" -ForegroundColor Cyan
Write-Host ""

# 顯示詳細結果
Write-Host "詳細執行結果:" -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Yellow

foreach ($Result in $Results) {
    $StatusColor = if ($Result.Status -eq "成功") { "Green" } else { "Red" }
    
    Write-Host "模型: $($Result.Name)" -ForegroundColor White
    Write-Host "狀態: $($Result.Status)" -ForegroundColor $StatusColor
    Write-Host "耗時: $($Result.Duration)" -ForegroundColor Cyan
    
    if ($Result.Error) {
        Write-Host "錯誤: $($Result.Error)" -ForegroundColor Red
    }
    
    Write-Host "----------------------------------------" -ForegroundColor Yellow
}

# 統計
$SuccessCount = ($Results | Where-Object { $_.Status -eq "成功" }).Count
$FailCount = ($Results | Where-Object { $_.Status -eq "失敗" }).Count

Write-Host ""
Write-Host "執行統計:" -ForegroundColor Green
Write-Host "成功: $SuccessCount 個" -ForegroundColor Green
Write-Host "失敗: $FailCount 個" -ForegroundColor Red
Write-Host "總計: $($Results.Count) 個" -ForegroundColor Cyan

# 暫停以便查看結果
Write-Host ""
Write-Host "按任意鍵退出..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
