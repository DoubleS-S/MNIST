# -*- encoding: utf-8 -*-
'''
@File    :   model_dispatcher.py
@Time    :   2023/07/31 19:00:37
@Author  :   Double S
@Version :   1.0
@Contact :   shishuoai@gmail.com
'''

from sklearn import tree
from sklearn import ensemble

models = {
    "decision_tree_gini": tree.DecisionTreeClassifier(
        criterion="gini"
    ),
    "decision_tree_entropy": tree.DecisionTreeClassifier(
        criterion="entropy"
    ),
    "rf": ensemble.RandomForestClassifier(),
}