# 遠端 Linux 主機深度學習訓練部署指南

## 檔案清單

```
project/
├── efficientnet/
│   └── efficientnet_dual_gpu_train.py
├── DenseNet/
│   └── densenet_dual_gpu_train.py
├── ConvNeXt/
│   └── convnext_dual_gpu_train.py
├── run_models_sequential.sh      # 主要執行腳本
├── quick_run.sh                  # 快速執行腳本
├── check_environment.sh          # 環境檢查腳本
├── install_dependencies.sh       # 依賴安裝腳本
└── README_REMOTE.md              # 本說明文件
```

## 部署步驟

### 1. 上傳檔案到遠端主機

```bash
# 方法一: 使用 scp
scp -r "project/" username@remote-host:/home/username/

# 方法二: 使用 rsync
rsync -avz "project/" username@remote-host:/home/username/project/

# 方法三: 使用 git (如果有版本控制)
git clone <repository-url> /home/username/project/
```

### 2. 登入遠端主機並設定權限

```bash
# SSH 登入
ssh username@remote-host

# 設定腳本執行權限
cd /home/username/project/
chmod +x *.sh
```

### 3. 修改路徑設定

編輯所有腳本中的 `PROJECT_ROOT` 變數，改為您的實際路徑：

```bash
# 編輯主要執行腳本
nano run_models_sequential.sh
# 修改: PROJECT_ROOT="/home/username/project"

# 編輯其他腳本
nano quick_run.sh
nano check_environment.sh
```

### 4. 安裝依賴套件

```bash
# 執行依賴安裝腳本
./install_dependencies.sh

# 或手動安裝
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install tensorboard pillow numpy
```

### 5. 檢查環境

```bash
# 執行環境檢查
./check_environment.sh
```

## 使用方法

### 基本執行

```bash
# 執行所有模型 (推薦)
./run_models_sequential.sh

# 或使用快速腳本
./quick_run.sh all
```

### 單獨執行特定模型

```bash
# 只執行 EfficientNet
./quick_run.sh efficientnet

# 只執行 DenseNet
./quick_run.sh densenet

# 只執行 ConvNeXt
./quick_run.sh convnext
```

### 背景執行

```bash
# 使用 nohup 在背景執行
nohup ./run_models_sequential.sh > training.log 2>&1 &

# 查看執行狀態
tail -f training.log

# 查看執行程序
ps aux | grep python
```

### 使用 screen 或 tmux

```bash
# 使用 screen
screen -S training
./run_models_sequential.sh
# Ctrl+A, D 離開 screen

# 重新連接
screen -r training

# 使用 tmux
tmux new-session -s training
./run_models_sequential.sh
# Ctrl+B, D 離開 tmux

# 重新連接
tmux attach -t training
```

## 監控與除錯

### 查看 TensorBoard

```bash
# 啟動 TensorBoard
tensorboard --logdir=./runs --bind_all

# 在瀏覽器中查看
# http://remote-host-ip:6006
```

### 查看 GPU 使用情況

```bash
# 即時監控
nvidia-smi

# 持續監控
watch -n 1 nvidia-smi

# 查看特定程序的 GPU 使用
nvidia-smi pmon
```

### 查看系統資源

```bash
# CPU 和記憶體使用
htop

# 磁碟使用
df -h

# 即時系統狀態
top
```

## 故障排除

### 常見問題

1. **CUDA 版本不匹配**
   ```bash
   # 檢查 CUDA 版本
   nvcc --version
   nvidia-smi
   
   # 重新安裝對應版本的 PyTorch
   pip3 uninstall torch torchvision torchaudio
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **記憶體不足**
   ```bash
   # 減少 batch size 或 workers 數量
   # 編輯 Python 檔案中的參數
   nano efficientnet/efficientnet_dual_gpu_train.py
   ```

3. **權限問題**
   ```bash
   # 確保有執行權限
   chmod +x *.sh
   
   # 確保有寫入權限
   chmod 755 /path/to/project/
   ```

4. **路徑問題**
   ```bash
   # 檢查工作目錄
   pwd
   
   # 檢查檔案是否存在
   ls -la *.py
   ```

## 效能優化建議

1. **使用 SSD 儲存資料**
2. **確保足夠的 RAM (建議 32GB+)**
3. **使用多個 GPU 時確保 PCIe 頻寬充足**
4. **監控溫度避免過熱**
5. **定期清理暫存檔案**

## 備份與恢復

```bash
# 備份訓練結果
tar -czf training_results_$(date +%Y%m%d).tar.gz runs/ checkpoints/

# 同步到本地
rsync -avz username@remote-host:/home/username/project/runs/ ./remote_results/
```

## 聯絡資訊

如有問題，請檢查：
1. 環境檢查腳本輸出
2. 訓練日誌檔案
3. GPU 狀態
4. 系統資源使用情況
