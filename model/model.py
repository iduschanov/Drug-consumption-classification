import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from sklearn.metrics import roc_auc_score
from sklearn.multioutput import MultiOutputClassifier

def forest_importances(estimator, data):
  '''DataFrame с важными фичами''' 
  feat_dict= {}
  for col, val in sorted(zip(data.columns, estimator.feature_importances_),key=lambda x:x[1],reverse=True):
    feat_dict[col]=val 
  return pd.DataFrame({'Feature':feat_dict.keys(),'Importance':feat_dict.values()})

def model_grid(model, params, x_train, y_train, scoring, multi_label=None):
  '''Настраиваем гиперпараметры модели'''
	if multi_label is True:
		grid_model = GridSearchCV(model, param_grid=params, scoring='roc_auc', n_jobs=-1)
		multi_model = MultiOutputClassifier(grid_model).fit(x_train, y_train)
		return multi_model

	grid_model = GridSearchCV(model, param_grid=params, scoring=scoring, n_jobs=-1).fit(x_train, y_train)
	return grid_model

def cross_val_multilabel(estimator, X, y, n):

  '''estimator: модель
     X: данные для разбиения
     y: целевая переменная
     n: количество фолдов'''

  kf = KFold(n_splits=n)
  kf.get_n_splits(X)
  roc_auc = [] # Для хранения полученных значений roc-auc
  for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model = estimator
    model.fit(X_train, y_train)
    y_pred = np.transpose([pred[:, 1] for pred in model.predict_proba(X_test)])
    roc_auc.append(roc_auc_score(y_test, y_pred, average=None))
  return roc_auc
