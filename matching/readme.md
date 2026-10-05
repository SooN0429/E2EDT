# matching（差分向量）

依最新 `node_model_training/training_report` 的訓練方案，為各節點的**攻擊類別**計算差分向量，供後續匹配使用。

## 主要執行

一般只需跑 **`diffvector_cal.py`**（會自動呼叫資料生成、算向量、存結果、清理暫存）：

```bash
cd E2EDT/matching
source ../.venv/bin/activate   # 整包 E2EDT 共用環境
python diffvector_cal.py
```

常用選項：

| 參數 | 說明 |
|------|------|
| `--run_dir` | 指定某次 training_report；預設用最新 |
| `--extracted_layer` | 特徵層；預設讀該次報告的 `config.json`（常見為 `7_point`） |
| `--keep_generated` | 保留暫存 paired 圖（預設會刪） |
| `--no_confirm_samples` | 不複製確認用範例圖 |
| `--gpu_id` | 如 `cuda:0`；無 GPU 時會自動用 CPU |

單獨只生成／清理 paired 資料可用 `diffvector_data_generate.py`；日常流程以 `diffvector_cal.py` 為準。

## 流程說明

1. **讀取最新訓練方案**  
   從 `training_report/*/config.json` 取得節點、攻擊類別、`seed`、`blended_alpha`、`extracted_layer` 等。

2. **產生 paired 資料（暫存）**  
   以與訓練相同的選圖條件，為每個攻擊類生成：
   - `attack/`：植入 trigger 的圖  
   - `clean/`：**同一張**圖植入前的乾淨版本（檔名一一對應）  
   寫入 `diffvector_generated/<run_name>/`（計算結束後預設刪除）。

3. **目視確認樣本（持久）**  
   每個攻擊類複製 1 組到  
   `diffvector_results/<run_name>/confirm_paired_samples/`。

4. **方法一：平均特徵差分**  
   凍結 ImageNet ResNet-18，在指定層取出特徵後：
   ```text
   差分向量 = mean_over_pairs( flatten(f(attack)) − flatten(f(clean)) )
   ```
   （方法二尚在設計，目前略過。）

5. **儲存結果並清理暫存**  
   向量與 `meta.json` 寫入 `diffvector_results/<run_name>/`；刪除 `diffvector_generated/`。

`<run_name>` 與對應的 `training_report` 資料夾名相同。

## 腳本分工

| 檔案 | 用途 |
|------|------|
| `diffvector_data_generate.py` | 讀報告、生成 / 清理 paired 資料、複製確認範例 |
| `diffvector_cal.py` | 編排整條管線 + 方法一差分向量計算 |
| `diffvector_matching.py` | （尚未實作） |

## 結果怎麼看

輸出根目錄：`matching/diffvector_results/<run_name>/`

```text
diffvector_results/<run_name>/
  meta.json
  confirm_paired_samples/
    node_2/00_white_square/
      attack.jpg      # 有 trigger
      clean.jpg       # 同圖、無 trigger
      pair_info.json  # 原始檔名等資訊
    node_3/...
  method1_mean_diff/
    node_2/00_white_square.npy
    node_3/00_small_hello_kitty.npy
    node_3/01_color_grid.npy
```

### 確認圖對不對

打開 `confirm_paired_samples/`：同資料夾內 `attack.jpg` 與 `clean.jpg` 應是同一張臉；clean 無 trigger、attack 有對應樣式（如白方塊、hello kitty、color grid）。

### 差分向量檔

- 每個 **節點 × 攻擊類** 一個 `.npy`（`float32` 一維向量）
- `meta.json` 的 `method1.nodes` 裡可看：路徑、`n_pairs`（參與平均的配對數）、`vector_shape`、`feature_shape`、`extracted_layer`
- `diff_order` 為 `attack_minus_clean`（攻擊特徵減乾淨特徵）

### meta.json 快速對照

- `source_report`：對應哪次訓練報告  
- `run_name`：結果資料夾名  
- `confirm_paired_samples`：確認圖路徑  
- `method1`：各攻擊差分向量路徑與統計；`method2` 目前為 `null`
