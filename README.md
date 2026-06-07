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
