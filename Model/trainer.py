"""
Model/trainer.py

此模組負責機器學習模型的訓練、評估與預測流程。
主要包含 DownsampleEnsemble 類別，該類別實作了針對不平衡資料的下採樣集結策略。
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report

class DownsampleEnsemble:
    """
    實作下採樣集結 (Downsample Ensemble) 策略的分類器管理類別。

    透過將多數類別 (Majority Class, 這裡是正常帳戶) 分割成多個子集，
    分別與少數類別 (Minority Class, 這裡是警示帳戶) 結合訓練多個隨機森林模型。
    最終預測結果由這些子模型的預測機率平均得出，旨在解決資料高度不平衡的問題。

    Attributes:
        n_splits (int): 交叉驗證 (Cross-Validation) 的 Fold 數量。
        n_vote_splits (int): 下採樣投票的分割數量 (即將正常帳戶分成幾等份)。
        models_final (list): 儲存最終訓練完成的所有子模型列表。
        feature_cols (list): 訓練時使用的特徵欄位名稱列表。
    """

    def __init__(self, n_splits=5, n_vote_splits=5):
        """
        初始化 DownsampleEnsemble 類別。

        Args:
            n_splits (int, optional): StratifiedKFold 的切分數量. Defaults to 5.
            n_vote_splits (int, optional): 將多數類別切分為幾個子集進行 Ensemble. Defaults to 5.
        """
        self.n_splits = n_splits
        self.n_vote_splits = n_vote_splits
        self.models_final = []
        self.feature_cols = []

    def train_cv(self, train_df):
        """
        執行層狀 K-Fold 交叉驗證 (Stratified K-Fold Cross-Validation) 以評估模型效能。

        此方法不會儲存最終模型，僅用於輸出每個 Fold 的 F1 Score 以及平均效能，
        幫助判斷當前參數與特徵工程的有效性。

        Args:
            train_df (pd.DataFrame): 包含特徵、'acct' (帳號) 與 'label' (標籤) 的完整訓練資料集。
        """
        print(f"\n[Model] 開始交叉驗證 (K={self.n_splits}, VoteSplits={self.n_vote_splits})...")
        
        # 排除非特徵欄位
        self.feature_cols = [col for col in train_df.columns if col not in ['acct', 'label']]
        X = train_df[self.feature_cols + ['acct']].fillna(0)
        y = train_df['label']
        
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train_fold = X.iloc[train_idx].copy()
            X_val_fold = X.iloc[val_idx].copy()
            y_train_fold = y.iloc[train_idx].copy()
            y_val_fold = y.iloc[val_idx].copy()

            # 呼叫內部方法進行下採樣訓練與驗證集預測
            fold_preds = self._train_vote_predict(
                X_train_fold, y_train_fold, 
                X_val_fold.drop(columns=['acct'])
            )
            
            # 以 0.5 為閾值計算 F1 Score
            fold_pred_labels = (fold_preds >= 0.5).astype(int)
            f1 = f1_score(y_val_fold, fold_pred_labels)
            fold_scores.append(f1)
            print(f"   Fold {fold+1} F1: {f1:.4f}")

        print(f"   平均 F1: {np.mean(fold_scores):.4f} (±{np.std(fold_scores):.4f})")

    def train_final(self, train_df):
        """
        使用完整的訓練資料集進行最終模型的訓練。

        此方法會呼叫 _train_vote_models 來訓練多個子模型，
        並將結果儲存在 self.models_final 中以供後續預測使用。

        Args:
            train_df (pd.DataFrame): 包含特徵與標籤的完整訓練資料集。
        """
        print("\n[Model] 訓練最終全量模型...")
        self.feature_cols = [col for col in train_df.columns if col not in ['acct', 'label']]
        X = train_df[self.feature_cols].fillna(0)
        y = train_df['label']
        
        # 儲存最終模型列表
        self.models_final = self._train_vote_models(X, y)
        print(f"   ✓ 已訓練 {len(self.models_final)} 個子模型")

    def predict_and_save(self, test_df, predict_info, output_path='./'):
        """
        對測試集進行預測，計算不同閾值的統計分佈，並儲存預測結果為 CSV 檔案。

        Args:
            test_df (pd.DataFrame): 測試集的特徵資料。
            predict_info (pd.DataFrame): 包含測試集帳號資訊 ('acct') 的 DataFrame，用於生成提交檔。
            output_path (str, optional): 輸出 CSV 檔案的路徑. Defaults to './'.
        """
        print("\n[Model] 正在預測測試集...")
        X_test = test_df[self.feature_cols].fillna(0)
        
        # 加總所有子模型的預測機率
        test_preds = np.zeros(len(X_test))
        for model in self.models_final:
            test_preds += model.predict_proba(X_test)[:, 1]
        
        # 取平均
        test_preds /= self.n_vote_splits

        # 顯示不同閾值的統計資訊
        print("-" * 35)
        print(f"{'閾值':<10} {'預測警示':<12} {'警示比例':<12}")
        for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
            pred_count = (test_preds >= thresh).sum()
            print(f"{thresh:<10.1f} {pred_count:<12} {pred_count / len(test_preds):<12.2%}")
        print("-" * 35)

        # 產生並儲存不同閾值的提交檔案
        for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
            pred = (test_preds >= thresh).astype(int)
            sub = pd.DataFrame({'acct': predict_info['acct'], 'label': pred})
            filename = f"{output_path}submission_thresh_{thresh:.1f}.csv"
            sub.to_csv(filename, index=False)
            print(f"   ✓ 已儲存: {filename}")

    def _train_vote_predict(self, X_train_df, y_train, X_eval_features):
        """
        (內部方法) 執行下採樣投票訓練並回傳驗證集預測機率。

        此方法專為 Cross-Validation 設計，它會訓練模型並直接對 X_eval_features 進行預測，
        不儲存模型實體以節省記憶體。

        Args:
            X_train_df (pd.DataFrame): 當前 Fold 的訓練特徵。
            y_train (pd.Series): 當前 Fold 的訓練標籤。
            X_eval_features (pd.DataFrame): 當前 Fold 的驗證集特徵 (不含 'acct')。

        Returns:
            np.ndarray: 驗證集的預測機率 (平均值)。
        """
        class_0_idx = np.where(y_train == 0)[0] # 正常帳戶索引
        class_1_idx = np.where(y_train == 1)[0] # 警示帳戶索引
        
        split_size = len(class_0_idx) // self.n_vote_splits
        combined_preds = np.zeros(len(X_eval_features))

        for i in range(self.n_vote_splits):
            # 分割正常帳戶
            start = i * split_size
            end = (i + 1) * split_size if i != self.n_vote_splits - 1 else len(class_0_idx)
            subset_idx = class_0_idx[start:end]
            
            # 結合全部警示帳戶
            combined_idx = np.concatenate([subset_idx, class_1_idx])

            X_subset = X_train_df.iloc[combined_idx]
            # 確保移除 acct 欄位
            if 'acct' in X_subset.columns:
                X_subset = X_subset.drop(columns=['acct'])
                
            y_subset = y_train.iloc[combined_idx]

            # 訓練隨機森林
            model = RandomForestClassifier(
                n_estimators=150, max_depth=10, min_samples_split=15,
                min_samples_leaf=8, random_state=42 + i,
                class_weight='balanced_subsample', n_jobs=-1
            )
            model.fit(X_subset, y_subset)
            
            # 累加預測機率
            combined_preds += model.predict_proba(X_eval_features)[:, 1]

        return combined_preds / self.n_vote_splits

    def _train_vote_models(self, X_train, y_train):
        """
        (內部方法) 執行下採樣投票訓練並回傳訓練好的模型列表。

        此方法用於最終模型訓練階段 (train_final)，它會回傳實體模型列表。

        Args:
            X_train (pd.DataFrame): 完整訓練特徵 (不含 'acct')。
            y_train (pd.Series): 完整訓練標籤。

        Returns:
            list: 包含多個訓練好的 RandomForestClassifier 物件的列表。
        """
        models = []
        class_0_idx = np.where(y_train == 0)[0]
        class_1_idx = np.where(y_train == 1)[0]
        split_size = len(class_0_idx) // self.n_vote_splits

        for i in range(self.n_vote_splits):
            start = i * split_size
            end = (i + 1) * split_size if i != self.n_vote_splits - 1 else len(class_0_idx)
            subset_idx = class_0_idx[start:end]
            combined_idx = np.concatenate([subset_idx, class_1_idx])

            X_subset = X_train.iloc[combined_idx]
            y_subset = y_train.iloc[combined_idx]

            model = RandomForestClassifier(
                n_estimators=150, max_depth=10, min_samples_split=15,
                min_samples_leaf=8, random_state=100 + i,
                class_weight='balanced_subsample', n_jobs=-1
            )
            model.fit(X_subset, y_subset)
            models.append(model)
        return models