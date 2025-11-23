# 警示戶預測模型README
## 模型原理
基於隨機森林的警示戶預測模型，使用下採樣與K折驗證，增進模型預測效能與穩定性。

## 目錄&檔案說明
```
├─Model
|    └─trainer.py   隨機森林訓練與預測程式，包含訓練前的下採樣程序
└─Preprocess
     └─features.py  特徵工程產製程式
     └─loader.py    載入原始資料集util
.gitignore          忽略檔
.python-version     鎖定Python版本為3.12
main.py             主程式、進入點，主要調用Model與Preprocess目錄程式
pyproject.toml      專案設定檔
README.md           專案設定文件
requirements.txt    專案所使用套件清單
uv.lock             套件版本鎖定（本專案可使用uv）
```

---
## 超參數設定
可復現模型結果的超參數設定及資源配置，皆已記載於`trainer.py`當中