import sys
import os
import math
import random
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, cross_validate
import joblib
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def trte_plot(y_train, y_train_pred, y_test, y_test_pred, # xlabel='Experimental yiled (%)', ylabel='Predicted yield (%)',\
              seed=0, method='FCN', target='yield', title='', filepath='',titlesize=10, fontsize=9, wt_value=True, save=False, close=False):
    if save is True:
        os.makedirs(filepath, exist_ok=True)

    print(y_train.shape, y_train_pred.shape)
    print(y_test.shape, y_test_pred.shape)

    range = [min(y_train.min(), y_test.min()) ,
             max(y_train.max(), y_test.max()) ]

    val0 = np.abs(y_train_pred - y_train)
    str0 = 'MAE(Train): %1.4f' % val0.mean()
    val1 = np.abs(y_test_pred - y_test)
    str1 = 'MAE(Test) : %1.4f' % val1.mean()

    print(str0)
    print(str1)

    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    
    # plot y=x
    ax.plot(range, range, color='0.5')
    ax.set_title(f'Seed: {seed}', fontsize=titlesize)

    # plot datapoint trte
    train_plot = ax.scatter(y_train, y_train_pred, s=5,
                facecolors='blue', edgecolors='blue', label='Training', alpha=0.4)
    test_plot = ax.scatter(y_test, y_test_pred, s=5,
                facecolors='red', edgecolors='red', label='Test', alpha=0.4)

    ax.legend(loc='upper left')
    # plt.show()

    # set plot lim
    ax.set_xlim([range[0], range[1]])
    ax.set_ylim([range[0], range[1]])
    
    # score legend
    if wt_value:
        ax.text(range[1] - 0.43 * (range[1] - range[0]),
                 range[1] - 0.92 * (range[1] - range[0]), str0, fontsize=8.5)
        ax.text(range[1] - 0.43 * (range[1] - range[0]),
                 range[1] - 0.97 * (range[1] - range[0]), str1, fontsize=8.5)

    xlabel = f'Experimental {target} (%)'
    ylabel = f'Predicted {target} (%)'

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)

    if save :
        plt.savefig(filepath + f'{target}_{method}_{seed}' + '.png', format='png', bbox_inches='tight', dpi=300)

        if close:
            plt.close()

if  __name__ == "__main__":
    n_feats = 26
    # seed = int(sys.argv[1])
    seed = 867
    np.random.seed(seed)

    ## load data
    dataset = np.array(pd.read_csv('propene_rxn_feat.v1.csv'))
    idx_rand = np.random.permutation(dataset.shape[0])
    n_train = int(0.8 * dataset.shape[0])

    dataset_train_x = dataset[idx_rand[:n_train], :n_feats]
    dataset_test_x = dataset[idx_rand[n_train:], :n_feats]

    dataset_train_y_scr = dataset[idx_rand[:n_train], n_feats + 2].reshape(-1,1)
    dataset_test_y_scr = dataset[idx_rand[n_train:], n_feats + 2].reshape(-1,1)

    X_train = dataset_train_x
    X_test = dataset_test_x
    y_train = dataset_train_y_scr
    y_test = dataset_test_y_scr

    ## load best parameters
    best_hyp_model_path ='./best_params/score/'

    model = joblib.load(os.path.join(best_hyp_model_path, f'best_model_xgb.pkl'))
    model = model.fit(dataset_train_x, dataset_train_y_scr.flatten())


    y_train_pred = model.predict(X_train).reshape(-1)
    y_test_pred  = model.predict(X_test).reshape(-1)

    train_mae = (np.abs(y_train_pred.reshape(-1) - y_train)).mean()
    test_mae  = (np.abs(y_test_pred.reshape(-1) - y_test)).mean()
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"MAE(Train) : {train_mae}")
    print(f"MAE(Test) : {test_mae}")
    print(f"r2(Train) : {train_r2}")
    print(f"r2(Test) : {test_r2}")

    rslt_dir = './plots'
    os.makedirs(rslt_dir, exist_ok=True)
    trte_plot(y_train=y_train, y_train_pred=y_train_pred, y_test=y_test, y_test_pred=y_test_pred,
                seed=seed, method='XGB', target='score', filepath='./plots/', save=True)
    