from catboost import Pool, CatBoostClassifier
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
np.random.seed(0)
from sklearn.metrics import f1_score
import scipy
import os
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib import pyplot as plt

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import argparse
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
np.random.seed(0)
from sklearn.metrics import f1_score
import scipy
import os
from pathlib import Path
from matplotlib import pyplot as plt
import argparse
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, accuracy_score
import shap



def mkdir(path):
    # 引入模块
    import os

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        # print(path + ' 目录已存在')
        return False



parser = argparse.ArgumentParser()
parser.add_argument('--fold', default="1",type=str)
parser.add_argument('--dataset', default="nhanes-mets-strict",type=str, help="choose from  [ nhanes-homa  nhanes-tyg  nhanes-mets-loose  nhanes-mets-strict]")
args = parser.parse_args()
fold=args.fold
print(fold)

savename='catboost_'+args.dataset

path='/data/gaowh/work/24process/tab-transformer/tabkanet_github_version_IR/'+savename

mkdir(path)




if args.dataset=="nhanes-homa":

    target_name = 'target1_bin'
    task = 'classification'
    continuous_features = ['RIDAGEYR', 'BMXWT', 'BMXHT', 'BMXBMI', 'BMXWAIST', 'BPXPLS', 'average_sy', 'average_di', 'LBXGLU']
    categorical_features = ['RIAGENDR','RIDRETH1']
    key="nhanes11-20"


elif args.dataset=="nhanes-tyg":

    target_name = 'target2_bin'
    task = 'classification'
    continuous_features = ['RIDAGEYR', 'BMXWT', 'BMXHT', 'BMXBMI', 'BMXWAIST', 'BPXPLS', 'average_sy', 'average_di', 'LBXGLU']
    categorical_features = ['RIAGENDR','RIDRETH1']
    key="nhanes11-20"

elif args.dataset=="nhanes-mets-strict":


    target_name = 'target3_bin'
    task = 'classification'
    continuous_features = ['RIDAGEYR', 'BMXWT', 'BMXHT', 'BMXBMI', 'BMXWAIST', 'BPXPLS', 'average_sy', 'average_di', 'LBXGLU']
    categorical_features = ['RIAGENDR','RIDRETH1']
    key="nhanes11-20"


elif args.dataset=="nhanes-mets-loose":


    target_name = 'target3_bin2'
    task = 'classification'
    continuous_features = ['RIDAGEYR', 'BMXWT', 'BMXHT', 'BMXBMI', 'BMXWAIST', 'BPXPLS', 'average_sy', 'average_di', 'LBXGLU']
    categorical_features = ['RIAGENDR','RIDRETH1']
    key="nhanes11-20"



train_data = pd.read_csv('/data/gaowh/work/24process/tab-transformer/use_tabtransformers/templates/'+key+'/Fold'+fold+'/train.csv').fillna('0')


val_data = pd.read_csv('/data/gaowh/work/24process/tab-transformer/use_tabtransformers/templates/'+key+'/Fold'+fold+'/val.csv').fillna('0')

test_data = pd.read_csv('/data/gaowh/work/24process/tab-transformer/use_tabtransformers/templates/'+key+'/Fold'+fold+'/test.csv').fillna('0')



if args.dataset!="nhanes-homa" :

    test_data_external = pd.read_csv('/data/gaowh/work/24process/tab-transformer/use_tabtransformers/templates/nhanes11-20/Fold1/CHARLS2015.csv').fillna('0')
else:
    test_data_external = pd.read_csv('/data/gaowh/work/24process/tab-transformer/use_tabtransformers/templates/'+key+'/Fold'+fold+'/test.csv').fillna('0')




label_encoders = {}

all_data = pd.concat([train_data, test_data, val_data])

for feature in categorical_features:
    le = LabelEncoder()
    all_data[feature] = le.fit_transform(all_data[feature])
    label_encoders[feature] = le  # 保存每个特征的编码器以便将来使用


train_data[categorical_features] = all_data[categorical_features].iloc[:len(train_data)].astype(int)
test_data[categorical_features] = all_data[categorical_features].iloc[len(train_data):len(train_data) + len(test_data)].astype(int)
val_data[categorical_features] = all_data[categorical_features].iloc[-len(val_data):].astype(int)

train_data[categorical_features] = train_data[categorical_features].astype(int)
test_data[categorical_features] = test_data[categorical_features].astype(int)
val_data[categorical_features] = val_data[categorical_features].astype(int)


y_train = train_data[target_name]
y_test = test_data[target_name]
y_valid = val_data[target_name]


X_train = train_data[categorical_features + continuous_features]
X_test = test_data[categorical_features + continuous_features]
X_valid = val_data[categorical_features + continuous_features]



X_test_external = test_data_external[categorical_features+continuous_features ]
y_test_external = test_data_external[target_name]

for feature in categorical_features:
    X_test_external[feature] = X_test_external[feature].astype(str)





external_ids = test_data_external["SEQN"]

test_ids = test_data["SEQN"]



features = categorical_features + continuous_features
categorical_features_indices = [i for i, f in enumerate(features) if f in categorical_features]




model = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6, loss_function='Logloss', eval_metric='AUC', random_seed=99, od_type='Iter', od_wait=100) 
model.fit(X_train, y_train,cat_features=categorical_features_indices,eval_set=(X_valid, y_valid),plot=True)


model.save_model(os.path.join(path,'catboost_model.cbm'))


preds_valid = np.array(model.predict_proba(X_valid))
valid_auc = roc_auc_score(y_score=preds_valid[:,1], y_true=y_valid)

preds = np.array(model.predict_proba(X_test))
test_auc = roc_auc_score(y_score=preds[:,1], y_true=y_test)



test_data_external=np.array(model.predict_proba(X_test_external))
test_auc_external = roc_auc_score(y_score=test_data_external[:,1], y_true=y_test_external)


print(f"VALID AUC SCORE FOR {key} : {valid_auc}")
print(f"TEST AUC SCORE FOR {key} : {test_auc}")
print(f"TEST External AUC SCORE FOR {key} : {test_auc_external}")



print("Test report")


print(classification_report(y_test, (preds[:,1] > 0.5).astype(int), digits=4))

print("External  report")

print(classification_report(y_test_external,  (test_data_external[:,1] > 0.5).astype(int), digits=4))




valid_f1 = f1_score(y_valid, (preds_valid[:,1] > 0.5).astype(int), average='macro')
test_f1 = f1_score(y_test, (preds[:,1] > 0.5).astype(int), average='macro')
test_external_f1 = f1_score(y_test_external, (test_data_external[:,1] > 0.5).astype(int), average='macro')

print(f"VALID MACRO F1 SCORE FOR {key} : {valid_f1}")
print(f"TEST MACRO F1 SCORE FOR {key} : {test_f1}")
print(f"TEST External F1 SCORE FOR {key} : {test_external_f1}")







valid_preds_class = (preds_valid[:,1] > 0.5).astype(int)
test_preds_class = (preds[:,1] > 0.5).astype(int)
test_external_preds_class = (test_data_external[:,1] > 0.5).astype(int)

valid_conf_matrix = confusion_matrix(y_valid, valid_preds_class)
test_conf_matrix = confusion_matrix(y_test, test_preds_class)
test_external_conf_matrix = confusion_matrix(y_test_external, test_external_preds_class)

valid_acc = accuracy_score(y_valid, valid_preds_class)
test_acc = accuracy_score(y_test, test_preds_class)
test_external_acc = accuracy_score(y_test_external, test_external_preds_class)



print("TEST CONFUSION MATRIX:")
print(test_conf_matrix)
print(f"TEST ACCURACY : {test_acc}")

print("TEST External CONFUSION MATRIX:")
print(test_external_conf_matrix)
print(f"TEST External ACCURACY : {test_external_acc}")






test_df = pd.DataFrame({
    'SEQN': test_ids,
    'y_true': y_test,
    'y_score': preds[:,1]
})

# 保存 DataFrame 到 CSV 文件
test_df.to_csv(os.path.join(path,'test_results.csv'), index=False)





results_df = pd.DataFrame({
    'SEQN': external_ids,
    'y_true': y_test_external,
    'y_score': test_data_external[:,1]
})


results_df.to_csv(os.path.join(path,'external_data_results.csv'), index=False)







# 预测概率
preds_external = model.predict_proba(X_test_external)[:, 1]

# 尝试不同的阈值
thresholds = np.arange(0.1, 0.9, 0.05)
best_threshold = 0.5
best_f1 = 0
best_acc = 0
best_preds_class = None

best_precision = 0
best_recall = 0
best_threshold = 0
best_preds_class = []

for threshold in thresholds:
    preds_class = (preds_external > threshold).astype(int)
    f1 = f1_score(y_test_external, preds_class, average='macro')
    acc = accuracy_score(y_test_external, preds_class)
    precision = precision_score(y_test_external, preds_class, average='macro')
    recall = recall_score(y_test_external, preds_class, average='macro')
    
    if f1 > best_f1:
        best_f1 = f1
        best_acc = acc
        best_precision = precision
        best_recall = recall
        best_threshold = threshold
        best_preds_class = preds_class

print(f"Best Threshold: {best_threshold}")
print(f"Best F1 Score: {best_f1}")
print(f"Best Accuracy: {best_acc}")
print(f"Best Precision: {best_precision}")
print(f"Best Recall: {best_recall}")



test_external_conf_matrix_best = confusion_matrix(y_test_external, best_preds_class)

print("TEST External CONFUSION MATRIX with Best Threshold:")
print(test_external_conf_matrix_best)


explainer = shap.TreeExplainer(model)


shap.summary_plot(explainer.shap_values(X_test_external), X_test_external, show=False)
plt.savefig("/data/gaowh/work/24process/tab-transformer/tabkanet_github_version_IR/SHAP_Summary_" + target_name + "_ext.png")
plt.close()


feature_importances = model.get_feature_importance()
feature_names = X_test_external.columns   



plt.figure(figsize=(10, 8))
sns.barplot(x=feature_importances, y=feature_names, palette='viridis')
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig("/data/gaowh/work/24process/tab-transformer/tabkanet_github_version_IR/Feature_Importance_" + target_name + "_ext.png")
plt.close()
