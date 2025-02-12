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

path='/data/gaowh/work/24process/tab-transformer/IR-github/'+savename

mkdir(path)




if args.dataset=="nhanes-homa":

    target_name = 'target1_bin'
    task = 'classification'
    continuous_features = ['RIDAGEYR', 'BMXWT', 'BMXHT', 'BMXBMI', 'BMXWAIST', 'BPXPLS', 'average_sy', 'average_di', 'LBXGLU']
    categorical_features = ['RIAGENDR','RIDRETH1']


elif args.dataset=="nhanes-tyg":

    target_name = 'target2_bin'
    task = 'classification'
    continuous_features = ['RIDAGEYR', 'BMXWT', 'BMXHT', 'BMXBMI', 'BMXWAIST', 'BPXPLS', 'average_sy', 'average_di', 'LBXGLU']
    categorical_features = ['RIAGENDR','RIDRETH1']

elif args.dataset=="nhanes-mets-strict":


    target_name = 'target3_bin'
    task = 'classification'
    continuous_features = ['RIDAGEYR', 'BMXWT', 'BMXHT', 'BMXBMI', 'BMXWAIST', 'BPXPLS', 'average_sy', 'average_di', 'LBXGLU']
    categorical_features = ['RIAGENDR','RIDRETH1']


elif args.dataset=="nhanes-mets-loose":


    target_name = 'target3_bin2'
    task = 'classification'
    continuous_features = ['RIDAGEYR', 'BMXWT', 'BMXHT', 'BMXBMI', 'BMXWAIST', 'BPXPLS', 'average_sy', 'average_di', 'LBXGLU']
    categorical_features = ['RIAGENDR','RIDRETH1']






test_data = pd.read_csv('/data/gaowh/work/24process/tab-transformer/IR-github/data/test.csv')




# 1	Mexican American		
# 2	Other Hispanic		
# 3	Non-Hispanic White		
# 4	Non-Hispanic Black	
# 5	Other Race - Including Multi-Racial	


# Gender
# 1 MALE
#  2 Female


X_test = test_data[categorical_features + continuous_features]
y_test = test_data[target_name]





model =CatBoostClassifier()
model.load_model(os.path.join(path,'catboost_model.cbm'))







preds = np.array(model.predict_proba(X_test))
test_auc = roc_auc_score(y_score=preds[:,1], y_true=y_test)



print(f"TEST AUC SCORE : {test_auc}")



print("Test report")

# 打印更详细的分类报告
print(classification_report(y_test, (preds[:,1] > 0.5).astype(int), digits=4))

test_f1 = f1_score(y_test, (preds[:,1] > 0.5).astype(int), average='macro')
print(f"TEST MACRO F1 SCORE : {test_f1}")
