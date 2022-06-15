import pandas as pd
import glob
import os
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

path = r'GSHSD\data\hatebasetwitter\train.csv'
pd.read_csv(path)['label_id'].unique() 
path = r'D:\My_stuffs\Thesis\ALL CODE HERE\GSHSD\GSHSD\data\vlsp2019\orig_train.csv'
train_data = pd.read_csv(path)

Fold = GroupKFold(n_splits=10)
groups = train_data['free_text'].values
for n, (train_index, val_index) in enumerate(Fold.split(train_data, train_data['label_id'], groups)):
    train_data.loc[val_index, 'fold'] = int(n)

train_data['fold'] = train_data['fold'].astype(int)

dev_path = r'D:\My_stuffs\Thesis\ALL CODE HERE\GSHSD\GSHSD\data\vlsp2019\dev.csv'
test_path = r'D:\My_stuffs\Thesis\ALL CODE HERE\GSHSD\GSHSD\data\vlsp2019\test.csv'
train_path = r'D:\My_stuffs\Thesis\ALL CODE HERE\GSHSD\GSHSD\data\vlsp2019\train.csv'


train_data[train_data['fold'].isin([0,1])].drop(['Unnamed: 0', 'id', 'CLEAN', 'OFFENSIVE', 'HATE', 'fold'], 1)\
    .to_csv(dev_path, index=False)
train_data[train_data['fold'].isin([2,3])].drop(['Unnamed: 0', 'id', 'CLEAN', 'OFFENSIVE', 'HATE', 'fold'], 1)\
    .to_csv(test_path, index=False)
train_data[~train_data['fold'].isin([0,1,2,3])].drop(['Unnamed: 0', 'id', 'CLEAN', 'OFFENSIVE', 'HATE', 'fold'], 1)\
    .to_csv(train_path, index=False)


all_datasets = {}
for file in glob.glob(r"GSHSD\data\*"):
    if os.path.isdir(file):
        dir_name = file.split('\\')[-1]
        file = Path(file)
        all_datasets[dir_name] = {
            'test': pd.read_csv((file / "test.csv")),
            'dev': pd.read_csv((file / "dev.csv")),
            'train': pd.read_csv((file / "train.csv")),
        }

