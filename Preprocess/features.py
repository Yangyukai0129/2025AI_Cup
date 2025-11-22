"""
Preprocess/features.py

此模組負責特徵工程 (Feature Engineering) 的核心邏輯。
主要功能是將原始的交易明細資料 (Transaction Data) 轉換為以帳戶為單位的數值特徵向量，
供後續機器學習模型訓練使用。
"""

import pandas as pd
import numpy as np

def create_features_fast(df_txn, account_list):
    """
    利用向量化操作快速生成帳戶交易特徵。

    針對指定的帳戶列表，從原始交易紀錄中匯總匯出 (Out) 與匯入 (In) 的統計數據
    (如金額總和、平均、標準差、次數、對手帳戶數等)，並進一步計算衍生特徵
    (如資金進出比例、金額波動率、網路度數比、每日平均金額等)。

    此函式使用 Pandas GroupBy 進行向量化運算，相較於迴圈處理有顯著的效能優勢。

    Args:
        df_txn (pd.DataFrame): 原始交易明細資料表。需包含以下欄位:
            - 'from_acct': 轉出帳號
            - 'to_acct': 轉入帳號
            - 'txn_amt': 交易金額
            - 'txn_date': 交易日期
            - 'channel_type': 交易管道
            - 'is_self_txn': 是否為本人交易 ('Y'/'N')
        account_list (list or np.array or pd.Series): 需要產生特徵的目標帳戶清單。

    Returns:
        pd.DataFrame: 包含目標帳戶各項統計特徵的資料表。
            - 包含 'acct' 欄位做為識別。
            - 其餘欄位為計算出的數值特徵 (如 'out_amt', 'in_ratio', 'net_outflow' 等)。
            - 缺失值 (NaN) 已填補為 0。
    """
    account_set = set(account_list)

    # 1. 篩選相關交易：只保留與目標帳戶有關的轉出或轉入記錄
    df_out = df_txn[df_txn['from_acct'].isin(account_set)].copy()
    df_in = df_txn[df_txn['to_acct'].isin(account_set)].copy()

    # 2. 匯出特徵聚合 (Aggregation)
    # 針對每個轉出帳戶計算統計量
    out_features = df_out.groupby('from_acct').agg({
        'txn_amt': ['count', 'sum', 'mean', 'std', 'min', 'max'],  # 金額統計
        'to_acct': 'nunique',                                      # 轉給多少個不同帳戶 (Out-Degree)
        'txn_date': 'nunique',                                     # 活躍天數
        'channel_type': lambda x: (x == 'UNK').sum(),              # 未知管道次數
        'is_self_txn': lambda x: (x == 'Y').sum()                  # 給自己的交易次數
    })
    # 重新命名欄位以利識別
    out_features.columns = ['out_txn', 'out_amt', 'out_avg', 'out_std', 'out_min', 'out_max',
                            'out_degree', 'out_days', 'out_unk', 'out_self']

    # 3. 匯入特徵聚合
    # 針對每個轉入帳戶計算統計量
    in_features = df_in.groupby('to_acct').agg({
        'txn_amt': ['count', 'sum', 'mean', 'std', 'min', 'max'],
        'from_acct': 'nunique',                                    # 收到多少個不同帳戶的錢 (In-Degree)
        'txn_date': 'nunique',
        'channel_type': lambda x: (x == 'UNK').sum()
    })
    in_features.columns = ['in_txn', 'in_amt', 'in_avg', 'in_std', 'in_min', 'in_max',
                           'in_degree', 'in_days', 'in_unk']

    # 4. 合併特徵
    # 使用 account_list 作為基底，確保所有目標帳戶都在結果中 (即使沒有交易紀錄也會保留)
    all_accts_df = pd.DataFrame({'acct': list(account_list)})
    features = all_accts_df.merge(out_features, left_on='acct', right_index=True, how='left')
    features = features.merge(in_features, left_on='acct', right_index=True, how='left')
    features = features.fillna(0) # 無交易紀錄者補 0

    # 5. 衍生特徵計算 (Derived Features)
    # 基礎加總與比例
    features['total_txn_count'] = features['out_txn'] + features['in_txn']
    features['total_amt'] = features['out_amt'] + features['in_amt']
    features['avg_amt'] = features['total_amt'] / features['total_txn_count'].replace(0, 1)
    features['in_ratio'] = features['in_txn'] / features['total_txn_count'].replace(0, 1)
    features['in_amt_ratio'] = features['in_amt'] / features['total_amt'].replace(0, 1)
    
    # 活躍度與網路特徵
    features['active_days'] = features[['out_days', 'in_days']].max(axis=1)
    features['degree_diff'] = features['in_degree'] - features['out_degree']
    
    # 波動性 (取匯出與匯入標準差的平均)
    features['amt_volatility'] = (features['out_std'].fillna(0) + features['in_std'].fillna(0)) / 2

    # 6. 進階特徵 (Advanced Features)
    # 網路度數比 (衡量是匯集點還是發散點)
    features['degree_ratio'] = features['in_degree'] / (features['out_degree'] + 1)
    
    # 每日平均交易金額 (衡量資金密集度)
    features['daily_avg_amt'] = features['total_amt'] / features['active_days'].replace(0, 1)
    
    # 淨流出 (正值代表流出大於流入)
    features['net_outflow'] = features['out_amt'] - features['in_amt']
    
    # 可疑管道比例
    features['unk_ratio'] = (features['in_unk'] + features['out_unk']) / features['total_txn_count'].replace(0, 1)
    
    # 變異係數概念 (標準差/平均值)，衡量金額不穩定程度
    features['volatility_ratio'] = features['amt_volatility'] / features['avg_amt'].replace(0, 1)

    return features