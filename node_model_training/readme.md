# node_model_training

資料生成（`node_dataset/node_traindata_generate.py`）採 VBD MaskBlended：方塊/grid 類 trigger 在有毒圖上為**絕對 3×3**（不隨 32/64 資產畫布等比放大）；hello_kitty 仍整圖 resize。

## 主要執行

- **`node_training.py`**：訓練各節點的入口腳本。  
  執行後會自動產生訓練報告（過程紀錄、結果、訓練資料樣本圖、模型權重等）。  
  訓練參數、訓練方案（`node_specs` 等）都在此檔最上方的 `DEFAULTS` 調整，或用命令列覆寫。

```bash
cd E2EDT/node_model_training
source ../.venv/bin/activate   # 整包 E2EDT 共用環境
python node_training.py
```

## 紀錄模組

- **`node_training_report.py`**：報表工具函式，由 `node_training.py` 在訓練過程中自動呼叫。  

## 報告輸出位置

預設寫入本目錄下的 `training_report/<run_name>/`（可用 `--training_note_root`、`--run_name` 調整）：

```text
training_report/<run_name>/
  config.json
  data_samples/
  epoch_logs/
  results/
  checkpoints/          # 各節點 .pth + index.json
```

模型權重固定放在該次報告的 `checkpoints/`（檔名 `{node}_{classes}.pth`），與 `diffvector_results/<同 run_name>/` 方案名對齊。
