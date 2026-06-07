# 多任務學習下的空氣品質跨域預警

**Cross-Domain Air Quality Early Warning via Multi-Task Learning: A Systematic Comparison of LSTM, Transformer, MoE, and S4**

東吳大學資料科學研究所 · 指導教授：曾俊然 博士  
研究生：蔡宇風 · 吳歷其 · 陳宥宸 · 中華民國 115 年 6 月

---

## 研究概要

本研究將空氣品質預測定位為**多任務學習**問題，單一模型於同一前向傳遞中同時輸出：

| 任務 | 說明 | 指標 |
|------|------|------|
| 污染物回歸 | 預測 PM2.5、PM10、O₃、NO₂ 未來濃度（1h / 3h / 6h） | RMSE |
| 二元超標分類 | PM2.5 是否超過 35.4 μg/m³ | F1、ROC-AUC |
| 六等級 AQI 分類 | PM2.5 落於哪一等級（Lv0–Lv5） | 宏觀 F1 |

核心研究問題：**純污染物驅動的模型是否具備跨地理的零樣本泛化能力，以及時間粒度落差如何限制此能力。**

---

## 資料集

### 訓練語料（逐小時）

| 地區 | 來源 | 筆數 | 站點 | 時間範圍 |
|------|------|------|------|----------|
| 台灣 | [Taiwan Air Quality Data 2016–2024](https://www.kaggle.com/datasets/taweilo/taiwan-air-quality-data-20162024) | 5,882,208 | 123 站 | 2016/11–2024/08 |
| 印度 | [Air Quality Dataset: Indian Cities 2022–2025](https://www.kaggle.com/datasets/bhautikvekariya21/air-quality-dataset-indian-cities-2022-2025) | 842,160 | 29 城市 | 2022/08–2025/11 |
| 首爾 | [Air Pollution in Seoul](https://www.kaggle.com/datasets/bappekim/air-pollution-in-seoul) | 647,511 | 25 站 | 2017/01–2019/12 |
| 北京 | [Beijing Multi-Site Air Quality](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data) | 420,768 | 12 站 | 2013/03–2017/02 |

清整後合併：**6,731,463 筆**（移除缺失率 >25% 之台灣 16 站）

### 跨域評估集（不參與訓練）

| 地區 | 來源 | 時間粒度 | 角色 |
|------|------|----------|------|
| 菲律賓 | [Philippine Cities Air Quality Index 2026](https://www.kaggle.com/datasets/bwandowando/philippine-cities-air-quality-index-data-2026) | 逐小時（與訓練一致） | 跨地理零樣本泛化主要場景 |
| 東京 | [Tokyo JP Air Quality](https://www.kaggle.com/datasets/nitirajkulkarni/tokyo-jp-1850147/suggestions) | **逐日**（相差 24 倍） | 跨時間粒度泛化限制場景 |

---

## 模型架構

四種序列建模思路，在完全相同的訓練設定下比較：

| 模型 | 架構特點 |
|------|----------|
| **LSTM** | 雙向，隱藏維度 128，2 層；閘門記憶機制 |
| **Transformer** | 標準編碼器，dim=128，4 頭，2 層；前置層正規化 + 可學習位置編碼 |
| **MoE** | Transformer 骨幹；前饋層替換為 8 專家混合，每次激活 2 個 |
| **S4** | 結構化狀態空間；以矩陣運算 + FFT 加速替代注意力機制 |

**共用訓練設定：** AdamW (lr=1e-3)、batch size 2048、最大 35 epoch、早停監控六等級宏觀 F1、bfloat16 混合精度、梯度裁剪 max_norm=1.0

---

## 特徵工程

輸入為 **24 時間步 × 25 維**特徵矩陣，僅依賴污染物歷史觀測（不含氣象變數）：

| 類別 | 維度 | 說明 |
|------|------|------|
| 污染物移動平均 | 12 | 各污染物 3 / 6 / 12 步滑動均值 |
| 原始污染物濃度 | 4 | PM2.5、PM10、O₃、NO₂（μg/m³，原始量綱） |
| 時間週期編碼 | 4 | hour_sin/cos、month_sin/cos |
| 其他時間與來源 | 3 | day、season、source_enc |
| 空間座標 | 2 | latitude、longitude |

> 不進行跨資料集的全域標準化，以保留 AQI 等級邊界的物理意義（35.4 μg/m³ 為真實警戒線）。

---

## 評估場景

| 場景 | 資料 | 評估面向 |
|------|------|----------|
| 混合全域測試集 | 四訓練來源各站較晚時段 | 域內短期外推 |
| 台灣域內驗證集 | 台灣 107 站最後 10% | 域內終極驗證 |
| 菲律賓跨域測試集 | 菲律賓全段（逐小時） | **跨地理零樣本泛化** |
| 東京跨域測試集 | 東京全段（逐日） | 跨時間粒度泛化（限制） |

---

## 主要結果

### 各場景最佳模型

| 場景 | 六等級分類 (1h) | 二元超標 (1h) | 回歸 RMSE |
|------|----------------|---------------|-----------|
| 混合全域 | LSTM **0.803** | Transformer F1 0.871 | Transformer **18.06** |
| 台灣域內 | MoE **0.642** | Transformer F1 0.742 | Transformer **9.52** |
| 菲律賓跨域 | LSTM **0.619** | LSTM F1 0.764 / AUC 0.9978 | Transformer **5.29** |
| 東京跨域 | MoE **0.420** | Transformer F1 0.517 | LSTM **18.02** |

### 核心發現

> **純污染物驅動的多任務時序模型，其泛化能力呈現「跨地理可行、跨時間粒度受限」的特性。**

- **跨地理泛化（菲律賓）**：LSTM 六等級宏觀 F1 達 0.619，二元 ROC-AUC 達 0.9978，表現甚至優於台灣域內驗證集，證明模型習得可跨地理遷移的污染物時序規律。
- **時間粒度落差（東京）**：時間粒度相差 24 倍時，六等級 F1 由 0.62 跌至 0.42，ROC-AUC 由 0.99 跌至 0.79，效能顯著受限，但仍明顯優於隨機。
- **架構分工**：LSTM 長於分類（捕捉等級轉換樣式）；Transformer 長於回歸（連續數值精準外推）。

---

## 專案結構

```
.
├── workspace/
│   ├── final.ipynb                    # 主要訓練與評估 notebook
│   ├── predict_philippines.ipynb      # 菲律賓跨域預測
│   ├── model(.pt)/                    # 訓練好的模型權重
│   │   ├── LSTM_best_baseline-2.pt
│   │   ├── Transformer_best_baseline-2.pt
│   │   ├── MoE_best_model.pt
│   │   └── S4_best_baseline.pt
│   └── windows data(.npz)/            # 滑動視窗打包後的資料
│       ├── Cross-Domain Air Pollution Prediction Wimdows.npz
│       ├── windows_philippines.npz
│       └── windows_japan.npz
├── 資料前處理/
│   └── 資料處理.ipynb                 # 資料清整、單位轉換、特徵工程
└── 物聯網期末書面報告.pdf             # 完整研究報告
```

---

## 環境需求

```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn
```

---

## 引用

若參考本研究，請引用：

> 蔡宇風、吳歷其、陳宥宸（2026）。多任務學習下的空氣品質跨域預警：LSTM、Transformer、MoE 與 S4 之系統性比較。東吳大學資料科學研究所。
