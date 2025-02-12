import sys
import logging
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from tabkanet.models import BasicNet ,BasicNetKAN,TabularTransformer,TabKANet,FeatureTokenizerTransformer,TabMLPNet
import argparse
CUDA_LAUNCH_BLOCKING=1
from tabkanet.metrics import f1_score_macro,auc_score
from tabkanet.tools_regression import seed_everything, train, inference, get_dataset, get_data_loader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


batch_size = 64
inference_batch_size = 64
epochs = 30
early_stopping_patience = 20
seed = 0
# model_object = TabularTransformer # or FeatureTokenizerTransformer


save_path="/data/gaowh/work/24process/tab-transformer/tabkanet_github_version_IR/tabkanet_reg_simply/model.pth"





parser = argparse.ArgumentParser()

parser.add_argument('--fold', default="1",type=str)
parser.add_argument('--dataset', default="btc",type=str, help="choose from  [ bankmarketing   seismic credit onlineshoper biodeg      income  blastchar albert]")
parser.add_argument('--modelname',default="tabkanet", type=str, help="choose from  [BasicNet tabtransformer kan tabkanet")
parser.add_argument('--gpunum',default=4, type=int)
parser.add_argument('--dim',default=32, type=int)
parser.add_argument('--testpath',default="/data/gaowh/work/24process/tab-transformer/use_tabtransformers/templates/nhanes11-20/Fold1/CHARLS2015.csv", type=str, help="choose from  ")

args = parser.parse_args()

fold=args.fold


if args.modelname=="BasicNet":
    model_object =  BasicNet 
elif args.modelname=="tabtransformer":
    model_object =  TabularTransformer 
elif args.modelname=="kan":
    model_object =  BasicNetKAN 
elif args.modelname=="FeatureTokenizerTransformer":
    model_object =  FeatureTokenizerTransformer 
elif args.modelname=="tabmlpnet":
    model_object =  TabMLPNet 
elif args.modelname=="tabkanet":
    model_object =  TabKANet 


print(fold)

print(args.modelname)

print(args.dataset)


testpath=args.testpath


# Hyperparameters for the model
# The main adjustments will be made here
batch_size = 64
inference_batch_size = 128
epochs = 300
early_stopping_patience = 100
early_stopping_start_from = 200
seed = 0
# model_object = TabularTransformer
output_dim = 1
embedding_dim = 64
nhead = 8
num_layers = 3
dim_feedforward = 8
mlp_hidden_dims = [32]
activation = 'relu'

# RMSE
maximize = False
# Loss function for the model
criterion = torch.nn.MSELoss()

#seed_everything(seed) # seed_everything is not used in this snippet

attn_dropout_rate = 0.1
ffn_dropout_rate = 0.1




target_name = 'target3'
task = 'regression'
continuous_features = [ "BMXBMI", "LBDGLUSI"]
categorical_features = []
key="nhanes-ir"




if args.modelname=="tabkanet" or args.modelname=="tabmlpnet" :
    all_count=len(continuous_features)+len(categorical_features)
    if all_count<=10:
        mlp_hidden_dims = [32]
    elif 10<all_count<20 :
        mlp_hidden_dims = [256,32]
    else:
        mlp_hidden_dims = [512,32]




seed_everything(seed)

def train_model():



    train_data = pd.read_csv('/data/gaowh/work/24process/tab-transformer/use_tabtransformers/templates/'+key+'/Fold'+fold+'/train.csv').fillna('EMPTY')


    val_data = pd.read_csv('/data/gaowh/work/24process/tab-transformer/use_tabtransformers/templates/'+key+'/Fold'+fold+'/val.csv').fillna('EMPTY')



    test_data = pd.read_csv('/data/gaowh/work/24process/tab-transformer/use_tabtransformers/templates/'+key+'/Fold'+fold+'/test.csv').fillna('EMPTY')
    test_data_add = pd.read_csv(testpath)
    test_data_add = test_data




    train_dataset, test_dataset_add, val_dataset = \
        get_dataset(
        train_data, test_data_add, val_data, target_name, 
        task, categorical_features, continuous_features)
    
    train_loader, test_loader_add, val_loader = \
        get_data_loader(
        train_dataset, test_dataset_add, val_dataset, 
        train_batch_size=batch_size, inference_batch_size=inference_batch_size)


    train_dataset, test_dataset, val_dataset = \
        get_dataset(
        train_data, test_data, val_data, target_name, 
        task, categorical_features, continuous_features)
    
    train_loader, test_loader, val_loader = \
        get_data_loader(
        train_dataset, test_dataset, val_dataset, 
        train_batch_size=batch_size, inference_batch_size=inference_batch_size)

    vocabulary1=train_dataset.get_vocabulary()
    vocabulary2=test_dataset.get_vocabulary()
    vocabulary3=val_dataset.get_vocabulary()

    
    combined_vocabulary = {}

    # 合并 vocabulary1
    for column, mapping in vocabulary1.items():
        if column not in combined_vocabulary:
            combined_vocabulary[column] = mapping
        else:
            # 若存在相同列，更新映射
            combined_vocabulary[column].update(mapping)

    # 合并 vocabulary2
    for column, mapping in vocabulary2.items():
        if column not in combined_vocabulary:
            combined_vocabulary[column] = mapping
        else:
            combined_vocabulary[column].update(mapping)

    # 合并 vocabulary3
    for column, mapping in vocabulary3.items():
        if column not in combined_vocabulary:
            combined_vocabulary[column] = mapping
        else:
            combined_vocabulary[column].update(mapping)


    final_vocabulary = {}
    for column in combined_vocabulary:
        # 先获取所有键并转换为字符串
        unique_values = sorted(str(value) for value in combined_vocabulary[column].keys())
        final_vocabulary[column] = {value: i for i, value in enumerate(unique_values)}

    model = model_object(
        output_dim=output_dim, 
        vocabulary=final_vocabulary,
        num_continuous_features=len(continuous_features), 
        embedding_dim=embedding_dim, nhead=nhead, num_layers=num_layers, dim_feedforward=dim_feedforward, attn_dropout_rate=attn_dropout_rate,
        mlp_hidden_dims=mlp_hidden_dims, activation=activation, ffn_dropout_rate=ffn_dropout_rate

        ,learninable_noise=False,bins=None
        )

    # model.load_state_dict(torch.load(f'/data/gaowh/work/24process/tab-transformer/use_tabtransformers/templates/nhanes-ir/Fold1/target3.pth'))
    model.load_state_dict(torch.load(save_path))






    model.eval()
    
    predictions = inference(model, test_loader_add, task=task)


if __name__ == '__main__':
    train_model()