"""
main.py

本專案的主執行入口 (Entry Point)。

此模組負責串接前處理 (Preprocess) 與模型 (Model) 兩大套件，
按順序執行完整的機器學習工作流程：
1. 資料載入與清洗
2. 特徵工程
3. 模型交叉驗證評估
4. 最終模型訓練
5. 測試集預測與結果輸出

Usage:
    請確保 'Preprocess' 與 'Model' 資料夾位於同級目錄，
    並且資料檔案 (csv) 位於 TRAIN_DATA_PATH 設定的路徑下。
    直接執行此檔案即可： python main.py
"""

import warnings
from Preprocess.loader import load_and_process_data
from Model.trainer import DownsampleEnsemble

# 忽略警告以保持輸出介面整潔
warnings.filterwarnings('ignore')

# 全域設定：資料路徑
TRAIN_DATA_PATH = 'data/'   # 原始訓練資料所在的資料夾
OUTPUT_DATA_PATH = './'     # 預測結果輸出的資料夾

def main():
    """
    主程式函式。

    協調並執行以下步驟：
    1. **資料前處理**: 呼叫 `load_and_process_data` 讀取資料並生成特徵。
    2. **模型初始化**: 建立 `DownsampleEnsemble` 實例，設定下採樣參數。
    3. **交叉驗證**: 執行 `train_cv` 評估模型在驗證集上的 F1 Score。
    4. **全量訓練**: 執行 `train_final` 使用所有可用資料訓練最終模型群。
    5. **預測輸出**: 執行 `predict_and_save` 產出提交檔案 (Submission CSV)。

    此函式不接受參數，路徑設定依賴模組頂部的全域變數。
    """
    print("="*80)
    print("🚀 模組化訓練啟動 - 下採樣優化版")
    print("="*80)

    # 1. Preprocess: 載入與處理資料
    # 回傳訓練特徵, 測試特徵, 預測名單(用於生成csv)
    train_df, test_df, predict_info = load_and_process_data(TRAIN_DATA_PATH, min_txn_count=14)
    
    print(f"\n[Main] 準備訓練資料: {train_df.shape}")

    # 2. Model: 初始化模型管理器
    # n_splits: CV 的折數
    # n_vote_splits: 正常帳戶切分為幾份來進行下採樣投票
    model_trainer = DownsampleEnsemble(n_splits=5, n_vote_splits=5)

    # 3. 執行交叉驗證 (查看模型體質)
    model_trainer.train_cv(train_df)

    # 4. 訓練最終模型
    model_trainer.train_final(train_df)

    # 5. 預測並輸出 CSV
    model_trainer.predict_and_save(test_df, predict_info, output_path=OUTPUT_DATA_PATH)
    
    print("\nAll Done!")
    print("閾值為0.7是最好的")

if __name__ == "__main__":
    main()