# # IEEE Fraud Detection Kaggle Competition
# ## LightGBM Validation
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
import pymongo
from sklearn.metrics import roc_auc_score

# Enter notes on validation here:
notes = ''
# Set to true if looking to create a submission file
submission = False

# MongoDB parameters
client = pymongo.MongoClient('mongodb://localhost:27017')
db = client.validation
collection = db.feature_engineering
# Create dictionary to send to MongoDB alongside validation score
mongo_dict = {'notes':notes}

# # Load data
train = pd.read_csv('../output/train_card1_count.csv')
val_indices = np.load('../output/val_indices.npy')
if submission:
    test = pd.read_csv('../output/test_card1_count.csv')
    sub_df = pd.read_csv('../data/sample_submission.csv.zip')

# Lightgbm parameters from https://www.kaggle.com/vincentlugat/ieee-lgb-bayesian-opt
SEED = 5000
early_rounds = 50
num_iterations = 200
params = {
        'bagging_fraction': 0.8999999999997461,
        'feature_fraction': 0.8999999999999121,
        'max_depth': int(50.0),
        'min_child_weight': 0.0029805017044362268,
        'min_data_in_leaf': int(20.0),
        'num_leaves': int(381.85354295079446),
        'reg_alpha': 1.0,
        'reg_lambda': 2.0,
        'objective': 'binary',
        'save_binary': True,
        'seed': SEED,
        'feature_fraction_seed': SEED,
        'bagging_seed': SEED,
        'drop_seed': SEED,
        'data_random_seed': SEED,
        'boosting_type': 'gbdt',
        'verbose': 1,
        'is_unbalance': False,
        'boost_from_average': True,
        'metric':'auc'
    }

mongo_dict['metric'] = params['metric']


categorical_features=['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
                      'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 'M1',
                      'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9',
                      'DeviceType', 'DeviceInfo', 'id_12', 'id_13', 'id_14', 'id_15',
                      'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22',
                      'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29',
                      'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36',
                      'id_37', 'id_38']
for feature in categorical_features:
    train[feature] = train[feature].astype('category')
if submission:
    for feature in categorical_features:
        test[feature] = test[feature].astype('category')

# Split train into features and target
X = train.drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1)
y = train['isFraud']
X_val = X.iloc[val_indices, :]
y_val = y.iloc[val_indices]
X_train = X.drop(val_indices)
y_train = y.drop(val_indices)
if submission:
    X_test = test.drop(['TransactionDT', 'TransactionID'], axis=1)


# # Train LightGBM
# Create arrays for oof predictions and sub predictions
if submission:
    sub_preds = np.zeros(len(sub_df))
else:
    val_preds = np.zeros(len(X_val))

trn_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val)

clf = lgb.train(params, trn_data, num_iterations, valid_sets=[trn_data, val_data], verbose_eval=1,
                early_stopping_rounds=early_rounds)

# # Predict on test sets
if submission:
    preds = clf.predict(X_test, num_iteration=clf.best_iteration)
    # # Save submission predictions
    sub_df['isFraud'] = sub_preds
    sub_df.to_csv('../submissions/submission.csv', index=False)
else:
    preds = clf.predict(X_val, num_iteration=clf.best_iteration)
    score = roc_auc_score(y_val.values, preds)
    # Bootstrapping AUROC
    n_bootstraps = 10000
    bootstrapped_scores = []

    np.random.seed(SEED)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = np.random.randint(0, len(val_preds) - 1, len(val_preds))
        if len(np.unique(y_val.values[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = roc_auc_score(y_val.values[indices], val_preds[indices])
        bootstrapped_scores.append(score)

    plt.hist(bootstrapped_scores, bins=50)
    plt.title('Histogram of the bootstrapped ROC AUC scores')
    plt.show()


    mongo_dict['score'] = score
    collection.insert_one(mongo_dict)


if not submission:
    # # Feature importances
    # Concatenate fold importances into feature importance dataframe
    feature_importance_df = pd.DataFrame()
    feature_importance_df["feature"] = X_train.columns.tolist()
    feature_importance_df["importance"] = clf.feature_importance()
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

    plt.figure(figsize=(15,10))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(20))
    plt.savefig('../output/lgb_importance.jpg')
    plt.show()


    # Save average feature importances
    feature_importance_df.to_csv('../output/lightgbm_importance.csv', index=False)

