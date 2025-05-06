# -*- coding: utf-8 -*-
# Importing required Library
import numpy as np
import pandas as pd
import random
from sklearn.datasets import make_classification
from sklearn.neighbors import NearestNeighbors


def create_dataset(n_sample=1000):
    ''' 
    Create a unevenly distributed sample data set multilabel  
    classification using make_classification function

    args
    nsample: int, Number of sample to be created

    return
    X: pandas.DataFrame, feature vector dataframe with 10 features 
    y: pandas.DataFrame, target vector dataframe with 5 labels
    '''
    X, y = make_classification(n_classes=5, class_sep=2,
                               weights=[0.1, 0.025, 0.205, 0.008, 0.9], n_informative=3, n_redundant=1, flip_y=0,
                               n_features=10, n_clusters_per_class=1, n_samples=1000, random_state=10)
    y = pd.get_dummies(y, prefix='class')
    return pd.DataFrame(X), y


def get_tail_label(df):
    """
    Give tail label columns of the given target dataframe
    """
    tail_label = []
    irpl_values = []
    for col in df.columns:
        value_counts = df[col].value_counts()
        count_1 = value_counts.get(1, 0)  # 安全获取正样本数
        irpl_values.append(count_1)

    max_count = max(irpl_values) if irpl_values else 0
    if max_count == 0:
        return []  # 所有标签无正样本

    irpl = [max_count / count if count != 0 else 0 for count in irpl_values]
    mir = np.mean(irpl)

    for idx, col in enumerate(df.columns):
        if irpl[idx] > mir:
            tail_label.append(col)
    return tail_label


def get_index(df):
    """
    Give the index of all tail_label rows
    """
    tail_labels = get_tail_label(df)
    if not tail_labels:
        return []
    index = set()
    for col in tail_labels:
        sub_index = set(df[df[col] == 1].index)
        index.update(sub_index)
    return list(index)


def get_minority_instace(X, y):
    """
    Give minority dataframe containing all the tail labels

    args
    X: pandas.DataFrame, the feature vector dataframe
    y: pandas.DataFrame, the target vector dataframe

    return
    X_sub: pandas.DataFrame, the feature vector minority dataframe
    y_sub: pandas.DataFrame, the target vector minority dataframe
    """
    index = get_index(y)
    if not index:  # 如果尾部标签为空，直接返回空
        return pd.DataFrame(), pd.DataFrame()
    X_sub = X.loc[index].reset_index(drop=True)
    y_sub = y.loc[index].reset_index(drop=True)
    return X_sub, y_sub


def nearest_neighbour(X):
    """
    Give index of 5 nearest neighbor of all the instance

    args
    X: np.array, array whose nearest neighbor has to find

    return
    indices: list of list, index of 5 NN of each element in X
    """
    nbs = NearestNeighbors(
        n_neighbors=5, metric='euclidean', algorithm='kd_tree').fit(X)
    euclidean, indices = nbs.kneighbors(X)
    return indices


def MLSMOTE(X, y, n_sample):
    """
    Give the augmented data using MLSMOTE algorithm

    args
    X: pandas.DataFrame, input vector DataFrame
    y: pandas.DataFrame, feature vector dataframe
    n_sample: int, number of newly generated sample

    return
    new_X: pandas.DataFrame, augmented feature vector data
    target: pandas.DataFrame, augmented target vector data
    """
    # 如果 X 是 numpy.ndarray 类型，将其转换为 pandas.DataFrame 类型
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    # 如果 y 是 numpy.ndarray 类型，将其转换为 pandas.DataFrame 类型
    if isinstance(y, np.ndarray):
        y = pd.DataFrame(y)

    indices2 = nearest_neighbour(X)
    n = len(indices2)
    new_X = np.zeros((n_sample, X.shape[1]))
    target = np.zeros((n_sample, y.shape[1]))
    for i in range(n_sample):
        reference = random.randint(0, n-1)
        neighbour = random.choice(indices2[reference, 1:])
        all_point = indices2[reference]
        nn_df = y[y.index.isin(all_point)]
        ser = nn_df.sum(axis=0, skipna=True)
        target[i] = np.array([1 if val > 2 else 0 for val in ser])
        ratio = random.random()
        gap = X.loc[reference, :] - X.loc[neighbour, :]
        new_X[i] = np.array(X.loc[reference, :] + ratio * gap)
    new_X = pd.DataFrame(new_X, columns=X.columns)
    target = pd.DataFrame(target, columns=y.columns)
    new_X = pd.concat([X, new_X], axis=0)
    target = pd.concat([y, target], axis=0)
    return new_X, target


# if __name__ == '__main__':
#     """
#     main function to use the MLSMOTE
#     """
#     X, y = create_dataset()  # Creating a Dataframe
#     # Getting minority instance of that datframe
#     X_sub, y_sub = get_minority_instace(X, y)
#     # Applying MLSMOTE to augment the dataframe
#     X_res, y_res = MLSMOTE(X_sub, y_sub, 100)
