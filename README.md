# VeriPromiseESG4K Promise Verification

本專案用於 AI CUP ESG 承諾驗證任務。

## 1. 任務與輸出格式

競賽要求輸出 5 個欄位：
```
id,promise_status,verification_timeline,evidence_status,evidence_quality
```

欄位定義：

| 欄位 | 說明 | 合法值 |
|---|---|---|
| `id` | 測試資料段落 ID | 必須與測試集一致 |
| `promise_status` | 是否為 ESG 承諾 | `Yes`, `No` |
| `verification_timeline` | 承諾可驗證時間 | `already`, `within_2_years`, `between_2_and_5_years`, `more_than_5_years`, `N/A` |
| `evidence_status` | 是否提供支持證據 | `Yes`, `No`, `N/A` |
| `evidence_quality` | 證據品質 | `Clear`, `Not Clear`, `Misleading`, `N/A` |

後處理規則由 `esg.data.probabilities_to_prediction()` 與 `esg.data.enforce_logic()` 處理：

- `promise_status = No` 時，其他三欄輸出 `N/A`
- `evidence_status = No` 或 `N/A` 時，`evidence_quality = N/A`
- 其他情況依模型機率取最大類別

## 2. 專案結構

```text
project_root/
├─ config/                         YAML 設定檔
├─ Data/                           競賽資料，需由使用者自行放入
├─ src/
│  ├─ common/                      設定讀取、logging、路徑工具
│  ├─ preprocessing/               資料讀取與格式驗證
│  ├─ training/                    訓練與 artifact 產生
│  ├─ inference/                   推論、cache 驗證、submission 輸出
│  └─ models/                      模型相關包裝
├─ esg/                            baseline 與共用資料、模型、訓練 helper
├─ experiments/                    訓練與推論產生的 artifacts/cache
├─ outputs/                        submission 輸出
├─ weights/                        保留給 .pt/.pth/.ckpt checkpoint
├─ scripts/                        Windows 批次指令
├─ requirements.txt
├─ .env.example
└─ README.md
```

## 3. 環境需求

建議環境：

| 項目 | 建議 |
| OS | Windows 10/11|
| Python | 3.11 到 3.13 |
| GPU | 完整重訓 Transformer 建議使用 NVIDIA GPU |
| CPU | 可執行資料檢查、TF-IDF 重算、已存在 cache 的推論 |
| CUDA | 依安裝的 PyTorch 版本決定 |

主要套件列於 `requirements.txt`：

numpy
pandas
scikit-learn
scipy
torch
transformers
safetensors
sentencepiece
accelerate
pyyaml

## 4. Hugging Face 模型

完整訓練會用到下列 pretrained models：

microsoft/mdeberta-v3-base
hfl/chinese-roberta-wwm-ext
hfl/chinese-roberta-wwm-ext-large
hfl/chinese-macbert-large
BAAI/bge-m3

本機 cache 方式載入 `BAAI/bge-m3`。若本機沒有模型，請先在可連網環境下載到 Hugging Face cache。

可選環境變數可參考 `.env.example`：

## 5. 資料放置方式

請在專案根目錄建立或保留 `Data/`，並放入以下檔案：

Data/vpesg4k_train_1000 V1.csv
Data/vpesg4k_val_1000.csv
Data/vpesg4k_test_2000.csv
Data/sample_submission_format.csv

預設路徑定義於 `config/default.yaml`

若資料檔名不同，可以使用 `--set` 覆蓋

## 6. 設定檔

主要設定：
config/default.yaml
config/train.yaml
config/inference.yaml


設定分工：

| 設定 | 用途 |
| `data.*` | train、validation、test、sample submission 路徑 |
| `paths.baseline_predictions` | dual-source baseline artifacts |
| `paths.source_predictions` | baseline 訓練原始輸出 |
| `paths.semantic_multitask` | semantic multitask artifacts |
| `paths.semantic_bge` | BGE embedding/logistic artifacts |
| `paths.lexical_tfidf` | TF-IDF/SVC artifacts |
| `paths.output` | 預設 submission 輸出位置 |
| `runtime.seed` | random seed |
| `runtime.cache_policy` | 推論 cache 策略 |
| `training.*` | batch size、learning rate、max length 等訓練參數 |
| `blend.*` | 最終機率融合權重 |
| `lexical_tfidf.*` | TF-IDF ngram 設定 |
| `semantic_bge.*` | BGE 模型與 logistic regression 設定 |


## 7. 重要模組輸入與輸出

### 7.1 前處理

位置：

src/preprocessing/

主要功能：
| 檔案 | 功能 | 輸入 | 輸出 |
| `loaders.py` | 讀取 labeled/unlabeled data | CSV 路徑 | `pandas.DataFrame` |
| `transforms.py` | 共同文字格式轉換 | DataFrame row | 模型輸入文字 |
| `validators.py` | submission 格式驗證 | submission DataFrame、expected ids | 無錯誤表示通過 |

### 7.2 Dual-source Baseline

入口：
dual_source_baseline_pipeline.py

用途：

- 訓練 full-data 分支
- 訓練 train-only 分支
- 依 `blend.baseline_full_weight = 0.8` 做 80/20 融合
- 產生 baseline submission 與 prediction artifacts

輸入：

Data/vpesg4k_train_1000 V1.csv
Data/vpesg4k_val_1000.csv
Data/vpesg4k_test_2000.csv

重要輸出：

experiments/predictions/full/<model_key>/<task>.npz
experiments/predictions/train_only/<model_key>/<task>.npz
experiments/dual_source_baseline/predictions/full/<model_key>/<task>.npz
experiments/dual_source_baseline/predictions/train_only/<model_key>/<task>.npz
outputs/dual_source_baseline_retrained.csv

`.npz` 內容：
key: target
shape: (測試資料筆數, 該任務類別數)

### 7.3 Semantic Multitask

位置：
src/training/semantic_multitask.py

用途：
- 使用同一個 Transformer encoder
- 接四個 task heads
- 對四個任務同時輸出機率

輸入：
train + validation labeled data
test unlabeled data

重要輸出：
experiments/semantic_multitask/submission/target.npz

`.npz` 內容：
keys:
  promise_status
  evidence_status
  evidence_quality
  verification_timeline

### 7.4 Semantic BGE

位置：
src/training/pipeline.py::train_semantic_bge

用途：
- 使用 `BAAI/bge-m3` 產生文本 embedding
- 以 logistic regression 補強 `promise_status`

輸入：
train + validation labeled data
test unlabeled data

重要輸出：
experiments/semantic_bge/embeddings.npy
experiments/semantic_bge/logreg_c10/promise_status.npz


`.npz` 內容：
key: target
shape: (測試資料筆數, 2)

### 7.5 Lexical TF-IDF

位置：
src/training/pipeline.py::train_lexical_tfidf

用途：
- 使用 character TF-IDF + LinearSVC
- 補強 `evidence_status`
- 補強 `verification_timeline`

`.npz` 內容：
key: target
shape: (測試資料筆數, 該任務類別數)

### 7.6 預測

入口：
src/inference/predict.py

輸入：
Data/vpesg4k_test_2000.csv
experiments/ 內的 artifacts/cache
config/inference.yaml

輸出：
outputs/promise_verification_submission.csv

## 8. 最終融合規則

第一層 baseline：
dual_source_baseline = 0.8 * full_data_branch + 0.2 * train_only_branch

第二層 final blend：
| Task | Dual-source baseline | Semantic multitask | Extra |
| `promise_status` | 0.60 | 0.25 | 0.15 BGE |
| `evidence_status` | 0.60 | 0.05 | 0.35 TF-IDF |
| `evidence_quality` | 1.00 | 0.00 | 0.00 |
| `verification_timeline` | 0.70 | 0.10 | 0.20 TF-IDF |

融合後再經過競賽邏輯規則產生最終類別。

## 9. 完整訓練
完整訓練並產生 submission：
python -m src.training.train --config config/train.yaml --stage all --force

若只想補缺失 artifact，不強制重訓已存在結果：
python -m src.training.train --config config/train.yaml --stage all

分段重訓：
python -m src.training.train --stage prepare-baseline --force
python -m src.training.train --stage semantic-multitask --force
python -m src.training.train --stage semantic-bge --force
python -m src.training.train --stage lexical-tfidf --force

## 10. 重現既有結果
若 `experiments/` 內 artifacts 已存在，可直接執行：
python -m src.inference.predict --output outputs/reproduce_submission.csv

## 11. Cache 策略

預測入口會把 `experiments/` 內 `.npz` 視為可選 cache，而不是不可缺少的固定輸入。

策略：
| 策略 | 行為 |
| `auto` | 預設。cache 合法就使用；缺失或損壞就重新產生 |
| `refresh` | 重新產生 prediction artifacts |
| `off` | 與 `refresh` 相同，保證不依賴既有 cache |

執行：
python -m src.inference.predict --cache-policy auto
python -m src.inference.predict --cache-policy refresh
python -m src.inference.predict --cache-policy off

cache 驗證項目：

- 檔案存在
- `.npz` 欄位存在
- shape 與測試資料筆數、類別數一致
- 數值不是 NaN 或 infinity
