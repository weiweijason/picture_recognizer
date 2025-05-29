#!/bin/bash

# Linux 腳本：依序執行深度學習模型訓練
# 執行順序：EfficientNet -> DenseNet -> ConvNeXt

# 設定工作目錄
PROJECT_ROOT="/home/user/project"  # 請根據實際遠端主機路徑修改
# 常見路徑範例：
# PROJECT_ROOT="/home/username/deep_learning_project"
# PROJECT_ROOT="/data/username/CNN_training"
# PROJECT_ROOT="$HOME/project"

# 定義要執行的Python檔案
declare -a models=(
    "efficientnet:efficientnet/efficientnet_dual_gpu_train.py"
    "DenseNet:DenseNet/densenet_dual_gpu_train.py"
    "ConvNeXt:ConvNeXt/convnext_dual_gpu_train.py"
)

# 記錄開始時間
start_time=$(date)
echo "======================================"
echo "開始執行深度學習模型訓練腳本"
echo "開始時間: $start_time"
echo "======================================"

# 初始化計數器
success_count=0
fail_count=0
total_count=${#models[@]}

# 切換到專案根目錄
cd "$PROJECT_ROOT" || {
    echo "錯誤: 無法切換到專案目錄 $PROJECT_ROOT"
    exit 1
}

# 依序執行每個模型
for i in "${!models[@]}"; do
    # 解析模型名稱和路徑
    IFS=':' read -r model_name model_path <<< "${models[$i]}"
    
    current_time=$(date)
    echo ""
    echo "[$((i+1))/$total_count] 執行 $model_name"
    echo "檔案路徑: $PROJECT_ROOT/$model_path"
    echo "開始時間: $current_time"
    echo "======================================"
    
    # 檢查檔案是否存在
    if [ ! -f "$model_path" ]; then
        echo "錯誤: 找不到檔案 $model_path"
        ((fail_count++))
        continue
    fi
    
    # 切換到對應的工作目錄
    model_dir=$(dirname "$model_path")
    cd "$PROJECT_ROOT/$model_dir" || {
        echo "錯誤: 無法切換到目錄 $PROJECT_ROOT/$model_dir"
        ((fail_count++))
        continue
    }
    
    echo "切換到工作目錄: $PROJECT_ROOT/$model_dir"
    
    # 記錄模型開始時間
    model_start_time=$(date +%s)
    
    # 執行Python檔案
    echo "正在執行: python $(basename "$model_path")"
    python "$(basename "$model_path")"
    
    # 檢查執行結果
    exit_code=$?
    model_end_time=$(date +%s)
    duration=$((model_end_time - model_start_time))
    
    if [ $exit_code -eq 0 ]; then
        echo "$model_name 執行完成！"
        echo "耗時: $(printf '%02d:%02d:%02d\n' $((duration/3600)) $((duration%3600/60)) $((duration%60)))"
        ((success_count++))
    else
        echo "$model_name 執行失敗！退出代碼: $exit_code"
        ((fail_count++))
    fi
    
    echo "======================================"
    
    # 回到專案根目錄準備下一個模型
    cd "$PROJECT_ROOT"
done

# 計算總耗時
end_time=$(date)
echo ""
echo "======================================"
echo "執行完成！總結報告"
echo "======================================"
echo "總開始時間: $start_time"
echo "總結束時間: $end_time"
echo ""
echo "執行統計:"
echo "成功: $success_count 個"
echo "失敗: $fail_count 個"
echo "總計: $total_count 個"

if [ $success_count -eq $total_count ]; then
    echo ""
    echo "🎉 所有模型都執行成功！"
    exit 0
else
    echo ""
    echo "⚠️  有 $fail_count 個模型執行失敗"
    exit 1
fi
