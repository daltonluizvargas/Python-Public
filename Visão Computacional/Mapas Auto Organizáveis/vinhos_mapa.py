# -*- coding: utf-8 -*-
"""
Created on Sun May 12 13:45:37 2019

@author: ENIAC
"""

from minisom import MiniSom
import pandas as pd

base = pd.read_csv('wines.csv')

x = base.iloc[:, 1:14].values
y = base.iloc[:, 0]. values

#fazer a normalização
from sklearn.preprocessing import MinMaxScaler
normalizador = MinMaxScaler(feature_range=(0,1))

x = normalizador.fit_transform(x)