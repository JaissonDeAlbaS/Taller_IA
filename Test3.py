# -*- coding: utf-8 -*-
"""
@author: Jaisson De Alba Santos
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from warnings import simplefilter
from sklearn.tree import DecisionTreeClassifier
simplefilter(action='ignore', category=FutureWarning)

url = 'Dataset/weatherAUS.csv'
data = pd.read_csv(url)