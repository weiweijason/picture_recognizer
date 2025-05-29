#!/bin/bash

# 遠端主機快速執行腳本
# 使用方法: ./quick_run.sh [model_name]
# 可用選項: efficientnet, densenet, convnext, all

PROJECT_ROOT="/home/user/project"

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 顯示使用方法
show_usage() {
    echo -e "${BLUE}使用方法:${NC}"
    echo "  $0 [model_name]"
    echo ""
    echo -e "${BLUE}可用選項:${NC}"
    echo "  efficientnet  - 只執行 EfficientNet"
    echo "  densenet      - 只執行 DenseNet"
    echo "  convnext      - 只執行 ConvNeXt"
    echo "  all           - 執行所有模型 (預設)"
    echo "  check         - 檢查環境"
    echo ""
    echo -e "${BLUE}範例:${NC}"
    echo "  $0 efficientnet"
    echo "  $0 all"
    echo "  $0 check"
}

# 檢查環境
check_env() {
    echo -e "${YELLOW}正在檢查環境...${NC}"
    if [ -f "$PROJECT_ROOT/check_environment.sh" ]; then
        bash "$PROJECT_ROOT/check_environment.sh"
    else
        echo -e "${RED}找不到環境檢查腳本${NC}"
    fi
}

# 執行單一模型
run_single_model() {
    local model=$1
    local script_path=""
    
    case $model in
        "efficientnet")
            script_path="$PROJECT_ROOT/efficientnet/efficientnet_dual_gpu_train.py"
            ;;
        "densenet")
            script_path="$PROJECT_ROOT/DenseNet/densenet_dual_gpu_train.py"
            ;;
        "convnext")
            script_path="$PROJECT_ROOT/ConvNeXt/convnext_dual_gpu_train.py"
            ;;
        *)
            echo -e "${RED}未知模型: $model${NC}"
            return 1
            ;;
    esac
    
    if [ ! -f "$script_path" ]; then
        echo -e "${RED}找不到腳本: $script_path${NC}"
        return 1
    fi
    
    echo -e "${GREEN}正在執行 $model...${NC}"
    local dir_path=$(dirname "$script_path")
    cd "$dir_path"
    python $(basename "$script_path")
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}$model 執行成功！${NC}"
    else
        echo -e "${RED}$model 執行失敗！${NC}"
    fi
    
    return $exit_code
}

# 執行所有模型
run_all_models() {
    echo -e "${GREEN}執行所有模型訓練...${NC}"
    bash "$PROJECT_ROOT/run_models_sequential.sh"
}

# 主程式
main() {
    local option=${1:-"all"}
    
    case $option in
        "check")
            check_env
            ;;
        "all")
            run_all_models
            ;;
        "efficientnet"|"densenet"|"convnext")
            run_single_model "$option"
            ;;
        "-h"|"--help"|"help")
            show_usage
            ;;
        *)
            echo -e "${RED}未知選項: $option${NC}"
            show_usage
            exit 1
            ;;
    esac
}

# 執行主程式
main "$@"
