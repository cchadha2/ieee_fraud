
# coding: utf-8

# # IEEE Fraud Detection Kaggle Competition

import random, os

import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.preprocessing import LabelEncoder

from country_match import country_match

SEED = 5000
train = pd.read_csv('../output/train_reduced_mem.csv')
test = pd.read_csv('../output/test_reduced_mem.csv')
sub_df = pd.read_csv('../data/sample_submission.csv.zip')
print('Loaded in data')
print(train.shape)
print(test.shape)
print(sub_df.shape)

def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything()

# https://www.kaggle.com/roydatascience/light-gbm-with-complete-eda/notebook
params = {
            'objective':'binary',
            'boosting_type':'gbdt',
            'metric':'auc',
            'n_jobs':-1,
            'num_leaves': 2**8,
            'max_depth':-1,
            'tree_learner':'serial',
            'colsample_bytree': 0.85,
            'subsample_freq':1,
            'subsample':0.85,
            'max_bin':255,
            'verbose':-1,
            'seed': SEED,
            'reg_alpha':0.3,
            'reg_lamdba':0.243,
            'learning_rate': 0.005}

# alpha = 0.7855459544383346
# alpha = 0.715
# alpha = 1
# best_iter = 2507
# num_iterations = int(best_iter*alpha)
params['n_estimators'] = 2000

# one_value_cols = [col for col in train.columns if train[col].nunique() <= 1]
# one_value_cols_test = [col for col in test.columns if test[col].nunique() <= 1]
#
# many_null_cols = [col for col in train.columns if train[col].isnull().sum() / train.shape[0] > 0.9]
# many_null_cols_test = [col for col in test.columns if test[col].isnull().sum() / test.shape[0] > 0.9]
#
# big_top_value_cols = [col for col in train.columns if train[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
# big_top_value_cols_test = [col for col in test.columns if test[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
#
# cols_to_drop = list(set(many_null_cols + many_null_cols_test + big_top_value_cols + big_top_value_cols_test + one_value_cols + one_value_cols_test))
# cols_to_drop.remove('isFraud')
# print('{} features are going to be dropped for being useless'.format(len(cols_to_drop)))
#
# train = train.drop(cols_to_drop, axis=1)
# test = test.drop(cols_to_drop, axis=1)
#
# important_features = [features.rstrip('\n') for features in open('../output/important_features_from_RFE.txt')]
# features_to_drop = [feature for feature in list(train) if feature not in important_features + ['isFraud']]

valid_card = train['card1'].value_counts()
valid_card = valid_card[valid_card > 10]
valid_card = list(valid_card.index)
train['card1'] = np.where(train['card1'].isin(valid_card), train['card1'], np.nan)
test['card1'] = np.where(test['card1'].isin(valid_card), test['card1'], np.nan)

i_cols = ['card1','card2','card3','card5',
          'C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14',
          'D1','D2','D3','D4','D5','D6','D7','D8','D9',
          'addr1','addr2',
          'dist1','dist2',
          'P_emaildomain', 'R_emaildomain'
         ]

for col in i_cols:
    temp_df = pd.concat([train[[col]], test[[col]]])
    fq_encode = temp_df[col].value_counts().to_dict()

    train[col + '_fq_enc'] = train[col].map(fq_encode)
    test[col + '_fq_enc'] = test[col].map(fq_encode)

for col in ['ProductCD', 'M4']:
    temp_dict = train.groupby([col])['isFraud'].agg(['mean']).reset_index().rename(
        columns={'mean': col + '_target_mean'})
    temp_dict.index = temp_dict[col].values
    temp_dict = temp_dict[col + '_target_mean'].to_dict()

    train[col + '_target_mean'] = train[col].map(temp_dict)
    test[col + '_target_mean'] = test[col].map(temp_dict)


for col in list(train):
    if train[col].dtype == 'O':
        print(col)
        train[col] = train[col].fillna('unseen_before_label')
        test[col] = test[col].fillna('unseen_before_label')

        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)

        le = LabelEncoder()
        le.fit(list(train[col]) + list(test[col]))
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])



train['uid'] = train['card1'].astype(str) + '_' + train['card2'].astype(str) + '_' + train['card3'].astype(str) + '_' + \
               train['card4'].astype(str)
test['uid'] = test['card1'].astype(str) + '_' + test['card2'].astype(str) + '_' + test['card3'].astype(str) + '_' + \
              test['card4'].astype(str)

train['uid2'] = train['uid'].astype(str) + '_' + train['addr1'].astype(str) + '_' + train['addr2'].astype(str)
test['uid2'] = test['uid'].astype(str) + '_' + test['addr1'].astype(str) + '_' + test['addr2'].astype(str)

valid_card = train['TransactionAmt'].value_counts()
valid_card = valid_card[valid_card > 10]
valid_card = list(valid_card.index)

train['TransactionAmt_check'] = np.where(train['TransactionAmt'].isin(test['TransactionAmt']), 1, 0)
test['TransactionAmt_check'] = np.where(test['TransactionAmt'].isin(train['TransactionAmt']), 1, 0)

i_cols = ['card1', 'card2', 'card3', 'card5', 'uid', 'uid2']

for col in i_cols:
    for agg_type in ['mean', 'std']:
        new_col_name = col + '_TransactionAmt_' + agg_type
        temp_df = pd.concat([train[[col, 'TransactionAmt']], test[[col, 'TransactionAmt']]])
        temp_df = temp_df.groupby([col])['TransactionAmt'].agg([agg_type]).reset_index().rename(
            columns={agg_type: new_col_name})

        temp_df.index = list(temp_df[col])
        temp_df = temp_df[new_col_name].to_dict()

        train[new_col_name] = train[col].map(temp_df)
        test[new_col_name] = test[col].map(temp_df)

train['card_TransactionAmt_ave']=train[['card1_TransactionAmt_mean', 'card2_TransactionAmt_mean',
                                             'card3_TransactionAmt_mean', 'card5_TransactionAmt_mean']].mean(axis=1)
test['card_TransactionAmt_ave']=test[['card1_TransactionAmt_mean', 'card2_TransactionAmt_mean',
                                             'card3_TransactionAmt_mean', 'card5_TransactionAmt_mean']].mean(axis=1)

train['bank_type'] = train['card3'].astype(str) + '_' + train['card5'].astype(str)
test['bank_type'] = test['card3'].astype(str) + '_' + test['card5'].astype(str)

train['address_match'] = train['bank_type'].astype(str) + '_' + train['addr2'].astype(str)
test['address_match'] = test['bank_type'].astype(str) + '_' + test['addr2'].astype(str)

for col in ['address_match', 'bank_type']:
    temp_df = pd.concat([train[[col]], test[[col]]])
    temp_df[col] = np.where(temp_df[col].str.contains('nan'), np.nan, temp_df[col])
    temp_df = temp_df.dropna()
    fq_encode = temp_df[col].value_counts().to_dict()
    train[col] = train[col].map(fq_encode)
    test[col] = test[col].map(fq_encode)

train['address_match'] = train['address_match'] / train['bank_type']
test['address_match'] = test['address_match'] / test['bank_type']

# Split train into features and target
X = train.drop(['isFraud', 'TransactionID', 'TransactionDT', 'bank_type', 'uid', 'uid2'], axis=1)
y = train['isFraud']

# val_indices = np.load('../output/val_indices.npy')
# X_val = X.iloc[val_indices, :]
# y_val = y.iloc[val_indices]
# X_train = X.drop(val_indices)
# y_train = y.drop(val_indices)
X_test = test.drop(['TransactionID', 'TransactionDT', 'bank_type', 'uid', 'uid2'], axis=1)

print(X.shape)
#
# print(X_train.shape)
# print(X_val.shape)
print(X_test.shape)

sub_preds = np.zeros(len(sub_df))
trn_data = lgb.Dataset(X, label=y)

# trn_data = lgb.Dataset(X_train, label=y_train)
# val_data = lgb.Dataset(X_val, label=y_val)

print('Beginning training on entire training set')
print('')
print('_________________________________________')
clf = lgb.train(params, trn_data, valid_sets=[trn_data], verbose_eval=50)

# clf = lgb.train(params, trn_data, valid_sets=[trn_data, val_data], verbose_eval=50)

sub_preds = clf.predict(X_test)
print(len(sub_preds))
sub_df['isFraud'] = sub_preds
sub_df.to_csv('../submissions/submission_cv_0.9306291154305194.csv', index=False)
