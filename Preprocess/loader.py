"""
Preprocess/loader.py

此模組負責資料載入、篩選與流程控制。
主要功能是讀取原始 CSV 檔案，針對正常帳戶進行基於交易次數的下採樣 (Undersampling) 篩選，
並呼叫特徵工程模組來準備最終的訓練與測試資料集。
"""

import pandas as pd
import time
from .features import create_features_fast

def load_and_process_data(data_path='./', min_txn_count=14):
    """
    載入原始資料，執行下採樣篩選，並協調特徵工程產出。

    此函式執行以下步驟：
    1. 讀取交易明細、警示帳戶名單與預測帳戶名單。
    2. 計算所有帳戶的交易總次數 (轉出 + 轉入)。
    3. 針對「非警示且非待預測」的正常帳戶進行篩選，僅保留交易次數大於等於
       `min_txn_count` 的活躍帳戶 (作為下採樣策略)。
    4. 合併警示帳戶與篩選後的正常帳戶作為訓練集名單。
    5. 呼叫 `create_features_fast` 為訓練集與測試集生成特徵。

    Args:
        data_path (str, optional): 資料檔案所在的資料夾路徑。預設為 './'。
            需包含 'acct_transaction.csv', 'acct_alert.csv', 'acct_predict.csv'。
        min_txn_count (int, optional): 正常帳戶納入訓練集的最小交易次數門檻。
            用於過濾掉大量不活躍的正常帳戶，平衡正負樣本比例。預設為 14。

    Returns:
        tuple: 包含三個 pandas DataFrame 的 Tuple:
            1. train_df: 訓練資料集 (包含特徵與 'label' 欄位)。
            2. test_df: 測試資料集 (包含特徵，無 'label')。
            3. predict: 原始的 'acct_predict.csv' DataFrame (用於後續生成提交檔格式)。
    """
    print("\n[Preprocess] 正在載入資料...")
    transaction = pd.read_csv(data_path + 'acct_transaction.csv')
    alert = pd.read_csv(data_path + 'acct_alert.csv')
    predict = pd.read_csv(data_path + 'acct_predict.csv')
    
    print(f"   ✓ 交易明細: {len(transaction):,} 筆")
    print(f"   ✓ 警示帳戶: {len(alert):,} 個")

    # === 計算帳戶交易數 (用於篩選) ===
    print("\n[Preprocess] 計算帳戶活躍度...")
    from_counts = transaction['from_acct'].value_counts()
    to_counts = transaction['to_acct'].value_counts()
    total_counts = from_counts.add(to_counts, fill_value=0).astype(int)

    acct_txn_df = pd.DataFrame({
        'acct': total_counts.index,
        'txn_count': total_counts.values
    })

    # === 標記帳戶 ===
    alert_accounts = set(alert['acct'].unique())
    predict_accounts = set(predict['acct'].unique())

    acct_txn_df['is_alert'] = acct_txn_df['acct'].isin(alert_accounts)
    acct_txn_df['is_predict'] = acct_txn_df['acct'].isin(predict_accounts)

    # === 篩選訓練集 (正常帳戶下採樣策略) ===
    normal_acct_df = acct_txn_df[
        (~acct_txn_df['is_alert']) &
        (~acct_txn_df['is_predict'])
    ].copy()

    # 只選擇活躍的正常帳戶
    active_normal_df = normal_acct_df[normal_acct_df['txn_count'] >= min_txn_count]
    print(f"   篩選交易數 >= {min_txn_count} 的正常帳戶: {len(active_normal_df):,} 個")

    # 訓練集帳戶名單 = 所有警示帳戶 + 活躍正常帳戶
    training_accounts = list(alert_accounts) + active_normal_df['acct'].tolist()
    
    # === 產生特徵 ===
    print("\n[Preprocess] 產生特徵中 (這可能需要一點時間)...")
    start_time = time.time()
    
    # 訓練特徵
    train_df = create_features_fast(transaction, training_accounts)
    train_df['label'] = train_df['acct'].apply(lambda x: 1 if x in alert_accounts else 0)
    
    # 測試特徵
    test_df = create_features_fast(transaction, predict['acct'].tolist())
    
    print(f"   ✓ 特徵工程完成！耗時: {time.time()-start_time:.1f} 秒")
    
    return train_df, test_df, predict