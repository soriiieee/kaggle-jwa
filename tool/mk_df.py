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
import conda

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
# initial
#

def load_data(path):
  df = pd.read_csv(path)
  print(df.shape)
  # print(df.info())
  return df