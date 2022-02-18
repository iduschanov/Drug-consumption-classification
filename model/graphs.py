import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_counts(data, labels):
    ''' Отрисовка количество значений'''
    fig = plt.figure(figsize=(6,4))
    for i in labels:
      sns.countplot(data[i]);
      plt.xlabel(i)
      plt.xticks(rotation=90)
      plt.show();
    return None
    
def plot_hisplot(data, labels):
    ''' Построение гистограмм для визуализации распределений''' 
    sns.set(font_scale = 1.2)
    for idx in labels:
        fig = plt.figure(figsize=(6,4))
        sns.histplot(data[idx]);
        plt.title(f'Distribution of {idx}')
        plt.xlabel(idx)
        plt.ylabel('Frequency')
        plt.show()
    return None


def forest_importances_plot(data):
    '''Отрисовка важных признаков'''

    values = data.Importance    
    idx = data.Feature
    plt.figure(figsize=(20,15))
    clrs = ['green' if (x < max(values)) else 'red' for x in values ]
    sns.barplot(y=idx,x=values,palette=clrs).set(title='Important features to predict customer Churn')
    plt.show()

    return None

def plot_confusion_matrix(y_test, y_pred):
  '''Матрица ошибок'''

  cm = confusion_matrix(y_test, y_pred)
  fig, ax = plt.subplots(figsize=(5,3)) 
  sns.heatmap(cm/np.sum(cm), annot=True, 
              fmt='.2%', cmap='Blues');