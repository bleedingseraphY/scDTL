import csv
import json

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import scanpypip.preprocessing as pp
from matplotlib.backends.backend_pdf import PdfPages
import os

from sklearn import metrics
import anndata

def RMSE(imputed_data, original_data):
    return np.sqrt(np.mean((imputed_data - original_data) ** 2))

def l1_distance(imputed_data, original_data):
    return np.mean(np.abs(original_data - imputed_data))

def pearson_corr(imputed_data, original_data):
    Y = original_data
    fake_Y = imputed_data
    fake_Y, Y = fake_Y.reshape(-1), Y.reshape(-1)
    fake_Y_mean, Y_mean = np.mean(fake_Y), np.mean(Y)
    corr = (np.sum((fake_Y - fake_Y_mean) * (Y - Y_mean))) / (
            np.sqrt(np.sum((fake_Y - fake_Y_mean) ** 2)) * np.sqrt(np.sum((Y - Y_mean) ** 2)))
    return corr

def indicator_drop():
    #dropout
    # sc_names = ['GSE112274','GSE117872','GSE134836','GSE134838','GSE134839','GSE134841','GSE149214', 'GSE81861', 'GSM3618014']
    sc_names = ['GSE134838']

    # pd_indicator = pd.DataFrame(columns=['GSE112274','GSE117872','GSE134836','GSE134838','GSE134839','GSE134841','GSE149214', 'GSE81861', 'GSM3618014'], index=[
    #                                                                 'impute_drop_RMSE','impute_drop_L1distance','impute_drop_PCCs'
    #                                                                 ])
    pd_indicator = pd.DataFrame(columns=['GSE134838'],
                                index=['impute_drop_RMSE','impute_drop_L1distance','impute_drop_PCCs'])
    # pd_indicator = pd.DataFrame(
    #     columns=['GSE112274', 'GSE117872', 'GSE134836', 'GSE134838', 'GSE134839', 'GSE134841', 'GSE149214', 'GSE81861',
    #              'GSM3618014'],
    #     index=['impute_drop_RMSE', 'impute_drop_L1distance', 'impute_drop_PCCs'])
    for sc_name in sc_names:
        dropout_data_value = sc.read_h5ad('./data/' + sc_name + '/' + sc_name + '_4000_drop_0.4.h5ad').X

        # predictions = pd.read_csv('./save/result/CL_Impute_result/' + sc_name + '/' + sc_name + '_impute_drop_0.4.csv', index_col=0).values
        predictions = pd.read_csv('./save/result/scrabble_result/' + sc_name + '/' + sc_name + '_drop_0.4_imputed.csv', index_col=0).values

        #inverse
        groundTruth = sc.read_h5ad('./data/' + sc_name + '/' + sc_name + '_4000.h5ad').X

        impute_all = predictions.copy()
        impute_zero = predictions.copy()
        impute_drop = predictions.copy()

        zero_i, zero_j = np.nonzero(groundTruth == 0)
        no_zero_i, no_zero_j = np.nonzero(dropout_data_value)

        impute_zero[no_zero_i, no_zero_j] = dropout_data_value[no_zero_i, no_zero_j]

        impute_drop[no_zero_i, no_zero_j] = dropout_data_value[no_zero_i, no_zero_j]
        impute_drop[zero_i, zero_j] = groundTruth[zero_i, zero_j]

        # pd_indicator.at['impute_all_RMSE', str(drop_rate)] = RMSE(groundTruth,impute_all)
        # pd_indicator.at['impute_all_L1distance', str(drop_rate)] = l1_distance(groundTruth,impute_all)
        # pd_indicator.at['impute_all_PCCs', str(drop_rate)] = pearson_corr(groundTruth,impute_all)
        #
        # pd_indicator.at['impute_zero_RMSE', str(drop_rate)] = RMSE(groundTruth, impute_zero)
        # pd_indicator.at['impute_zero_L1distance', str(drop_rate)] = l1_distance(groundTruth, impute_zero)
        # pd_indicator.at['impute_zero_PCCs', str(drop_rate)] = pearson_corr(groundTruth, impute_zero)

        pd_indicator.at['impute_drop_RMSE', sc_name] = RMSE(groundTruth, impute_drop)
        pd_indicator.at['impute_drop_L1distance', sc_name] = l1_distance(groundTruth, impute_drop)
        pd_indicator.at['impute_drop_PCCs', sc_name] = pearson_corr(groundTruth, impute_drop)

        # print(drop_rate,"_NoUnet_RMSE:",RMSE(groundTruth,impute_drop))
        # print(drop_rate,"_NoUnet_l1_distance:",l1_distance(groundTruth,impute_drop))
        # print(drop_rate,"_NoUnet_pearson_corr:",pearson_corr(groundTruth,impute_drop))
    # #write to csv
    pd_indicator.to_csv('./save/result/indicator_scrabble_impute.csv')

    return True

indicator_drop()
