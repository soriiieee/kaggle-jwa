# -*- coding: utf-8 -*-
# title  : [     ]
# date   : 2020.0x.xx
# editor : sori-machi
# action : kaggle cat 
# url : https://www.kaggle.com/c/cat-in-the-dat-ii/overview
#---------------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.impute import SimpleImputer
import scipy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from tqdm import tqdm_notebook

sys.path.append("/home/griduser/work/kaggle-jwa/tool")
from mk_df import load_data

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
# initial
#

DIR="/home/griduser/work/kaggle-jwa/dat/0422_cat"

train = load_data(DIR+"/train.csv")
# train.to_pickle(DIR+"/train.pkl")
# train = pd.read_pickle(DIR+"/train.pkl")
test = load_data(DIR+"/test.csv")

# train.to_csv(DIR+)

# sys.exit()
# ['id', 'bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'nom_0', 'nom_1',
#        'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9',       'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5', 'day', 'month',
#       #  'target']
# print(train.s)
# sys.exit()
# col ="bin_0"
use_col=['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'target']
use_col2=['nom_0', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6','nom_7' ,'nom_8','nom_9','target']
use_col3=['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5', 'day', 'month','target']
train = train[use_col]

f,ax = plt.subplots(5,5,figsize=(12,12))
ax = ax.flatten()

# for i in range(train.shape[1]):
#   col =train.columns[i]
#   sns.countplot(
#   data=train,
#   x = col,
#   hue="target",
#   ax = ax[i])

sns.pairplot(train[:2000])

plt.savefig(DIR+f"/tmp1.png")
# print(train.columns)
print(train.target.nunique())
# print(train.head())