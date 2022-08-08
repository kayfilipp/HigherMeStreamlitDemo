from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle
X = pd.read_csv('https://raw.githubusercontent.com/kayfilipp/HigherME/main/census_data/SMOTE_df.csv')
y = pd.read_csv('https://raw.githubusercontent.com/kayfilipp/HigherME/main/census_data/SMOTE_df_target.csv')
pickl_path = 'rf.pkl'


def get_random_forest():
    best_rf = RandomForestClassifier(
        criterion='log_loss'
        , max_depth=7
        , min_samples_leaf=150
        , min_samples_split=24
        , n_estimators=200
        , random_state=1
    )
    best_rf.fit()
    return best_rf
