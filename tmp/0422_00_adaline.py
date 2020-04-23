# -*- coding: utf-8 -*-
# title  : [     ]
# date   : 2020.0x.xx
# editor : sori-machi
# action : 
#---------------------------------------------------------------------------
# basic-module
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
# import conda
import numpy as np
# import itertools
#---------------------------------------------------
# scikit-learn
from sklearn.preprocessing import StandardScaler #標準化
from sklearn.preprocessing import MinMaxScaler #正規化
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

import warnings
warnings.simplefilter("ignore")

sys.path.append("/home/griduser/work/kaggle-jwa/tool")
from mk_df import load_data

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
# initial
#

x1 = np.arange(15)
x2 = np.arange(15)*2

# x3 = np.dot(x1,x2)
x3 = x1.T.dot(x2)
print(x3)
sys.exit()