import argparse
import logging
import sys
import time
import warnings
import os
import numpy as np
import pandas as pd
import scanpy
import torch
import scanpy as sc
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import  nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import GPUtil
import trainers as t
from models import (AEBase, PretrainedImputor, PretrainedVAEImputor, VAEBase, U_Net, AttentionVotor)
import matplotlib
import random
import json
seed=42
torch.manual_seed(seed)
#np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
#from transformers import *
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark=False

def RMSE(imputed_data, original_data):
    return np.sqrt(np.mean((imputed_data - original_data) ** 2))

def l1_distance(imputed_data, original_data):
    return np.mean(np.abs(original_data - imputed_data))

def pearson_corr(imputed_data, original_data):
    Y = original_data
    fake_Y = imputed_data
    # fake_Y, Y = fake_Y.reshape(-1), Y.reshape(-1)
    fake_Y_mean, Y_mean = np.mean(fake_Y), np.mean(Y)
    corr = (np.sum((fake_Y - fake_Y_mean) * (Y - Y_mean))) / (
            np.sqrt(np.sum((fake_Y - fake_Y_mean) ** 2)) * np.sqrt(np.sum((Y - Y_mean) ** 2)))
    return corr

def pearson_corr_Tensor(imputed_data, original_data):
    Y = original_data
    fake_Y = imputed_data
    corr = 0
    for i in range(Y.shape[0]):
        temp_Y = Y[i]
        temp_fake_Y = fake_Y[i]
        temp_fake_Y_mean, temp_Y_mean = torch.mean(temp_fake_Y), torch.mean(temp_Y)
        temp_corr = (torch.sum((temp_fake_Y - temp_fake_Y_mean) * (temp_Y - temp_Y_mean))) / (
                torch.sqrt(torch.sum((temp_fake_Y - temp_fake_Y_mean) ** 2)) * torch.sqrt(torch.sum((temp_Y - temp_Y_mean) ** 2)))
        corr = corr + temp_corr
    corr = corr/float(Y.shape[0])
    return corr

def cbam_indicator(X_eval_dataloader,drop_data_mmscaler, groundTruth_mmscaler,unet,drop_rate):
    column = str(drop_rate)
    pd_indicator = pd.DataFrame(columns=[column],
                                index=['impute_all_RMSE', 'impute_all_L1distance', 'impute_all_PCCs',
                                       'impute_zero_RMSE', 'impute_zero_L1distance', 'impute_zero_PCCs',
                                       'impute_drop_RMSE', 'impute_drop_L1distance', 'impute_drop_PCCs'
                                       ])

    # unet.eval()
    prediction_tensors = []
    drop_data_tensors = []
    groundTruth_tensor = []
    for batchidx, (x,x_drop,y) in enumerate(X_eval_dataloader):
        # generate_data_Tensor = torch.FloatTensor(generate_data.values).to(device)
        with torch.no_grad():
            prediction_tensor = unet(x,x_drop)
            if (batchidx == 0):
                prediction_tensors = prediction_tensor
                drop_data_tensors = x_drop
                groundTruth_tensor = y
            else:
                prediction_tensors = torch.cat([prediction_tensors, prediction_tensor], dim=0)
                drop_data_tensors = torch.cat([drop_data_tensors, x_drop], dim=0)
                groundTruth_tensor = torch.cat([groundTruth_tensor, y], dim=0)
    #
    predictions = prediction_tensors.detach().cpu().numpy().astype(float)
    drop_data = drop_data_tensors.detach().cpu().numpy().astype(float)
    groundTruth = groundTruth_tensor.detach().cpu().numpy().astype(float)
    #find no zero value of selected_data_single
    #反归一化
    drop_data = drop_data_mmscaler.inverse_transform(drop_data.T).T

    impute_all = predictions.copy()
    impute_zero = predictions.copy()
    impute_drop = predictions.copy()

    no_zero_i, no_zero_j = np.nonzero(drop_data)
    # no_zero_i_groundTruth, no_zero_j_groundTruth = np.nonzero(groundTruth.X)
    zero_i,zero_j = np.nonzero(groundTruth == 0)

    impute_zero[no_zero_i, no_zero_j] = drop_data[no_zero_i, no_zero_j]

    impute_drop[no_zero_i, no_zero_j] = drop_data[no_zero_i, no_zero_j]
    impute_drop[zero_i, zero_j] = groundTruth[zero_i, zero_j]
    #

    return impute_all,impute_zero,impute_drop

def show_gpu_info():
    gpus = GPUtil.getGPUs()

    if not gpus:
        print("No GPUs found.")
        return

    for i, gpu in enumerate(gpus):
        print(f"GPU {i + 1} - ID: {gpu.id}, Name: {gpu.name}")
        print(f"    GPU Memory Total: {gpu.memoryTotal} MB")
        print(f"    GPU Memory Free: {gpu.memoryFree} MB")
        print(f"    GPU Memory Used: {gpu.memoryUsed} MB")
        print(f"    GPU Load: {gpu.load * 100}%\n")

def run_main(args):

    sc_data_name = args.data_name
    # Extract parameters
    epochs = args.epochs
    log_path = args.log
    batch_size = args.batch_size
    con_kernel = args.con_kernel.split(",")
    reduce_model = args.dimreduce

    con_kernel = list(map(int, con_kernel) )

    para = "data_"+str(args.data_name)+"_con_kernel_"+str(args.con_kernel)+"_dropRate_" + str(args.drop_rate) +"_model_"+reduce_model   #(para)
    now=time.strftime("%Y-%m-%d-%H-%M-%S")

    for path in [args.log,args.bulk_model,args.cbam_pre]:
        if not os.path.exists(path):
            # Create a new directory because it does not exist
            os.makedirs(path)
            print("The new directory is created!")

    # Load model from checkpoint
    if(args.checkpoint not in ["False","True"]):
        # para = os.path.basename(selected_model).split("_DaNN.pkl")[0]
        args.checkpoint = 'True'

    cbam_pre = args.cbam_pre + para
    # Read data
    generate_data = pd.read_csv('./save/result/' + sc_data_name + '/P_impute_all_' + str(args.drop_rate) + '.csv', index_col=0)

    groundTruth_adata = sc.read_h5ad('./data/' + sc_data_name + '/' + sc_data_name + '_4000.h5ad')

    labels = groundTruth_adata.obs['leiden']
    ##LabelEncoder
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    # unique_labels, label_counts = np.unique(labels, return_counts=True)

    # Select the Training device
    if(args.device == "gpu"):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(device)
    else:
        device = 'cpu'

    unique_labels = np.unique(encoded_labels)
    center_cells = {}
    for unique_label in unique_labels:
        label_data = groundTruth_adata[encoded_labels == unique_label, :]
        label_data = label_data.to_df()
        center_cell = label_data.apply(lambda col: np.sum(col[col != 0]) / np.count_nonzero(col[col != 0]),
                                               axis=0).fillna(0)
        center_cell = torch.FloatTensor(center_cell).to(device)
        # center_cell = center_cell.to_dict()
        center_cells[unique_label] = center_cell

    if(args.drop_rate != 0.0):
        drop_adata = sc.read_h5ad(
            './data/' + args.data_name + '/' + args.data_name + '_4000_drop_' + str(args.drop_rate) + '.h5ad')
    else:
        drop_adata = groundTruth_adata
    # Preprocess data if spcific process is required
    single_data=generate_data.values
    drop_data = drop_adata.X
    # label_r=label_r.fillna(na)

    # Initialize logging and std out
    out_path = log_path+now+"bulk.err"
    log_path = log_path+now+"bulk.log"

    out=open(out_path,"w")
    sys.stderr=out

    logging.basicConfig(level=logging.INFO,
                    filename=log_path,
                    filemode='a',
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )
    logging.getLogger('matplotlib.font_manager').disabled = True

    logging.info(args)
    # Scaling data
    single_data_mmscaler = preprocessing.MinMaxScaler().fit(single_data.T)
    single_data = single_data_mmscaler.transform(single_data.T).T
    groundTruth_data = groundTruth_adata.X


    # Split traning valid test set
    X_train_all,X_test,X_drop_train_all,X_drop_test_orgin,Y_train_all,Y_test,label_train_all,label_test = train_test_split(single_data,drop_data,groundTruth_data,encoded_labels,test_size=args.test_size,random_state=42)

    X_drop_train_all_mmscaler = preprocessing.MinMaxScaler().fit(X_drop_train_all.T)
    X_drop_test_orgin_mmscaler = preprocessing.MinMaxScaler().fit(X_drop_test_orgin.T)
    Y_test_mmscaler = preprocessing.MinMaxScaler().fit(Y_test.T)
    #transform
    X_drop_train_all = X_drop_train_all_mmscaler.transform(X_drop_train_all.T).T
    X_drop_test = X_drop_test_orgin_mmscaler.transform(X_drop_test_orgin.T).T

    X_train,X_valid,X_drop_train,X_drop_valid,Y_train,Y_valid,label_train,label_valid = train_test_split(X_train_all,X_drop_train_all,Y_train_all,label_train_all,test_size=args.test_size,random_state=42)


    #print(device)
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    #logging.info(device)
    print(device)
    # Construct datasets and data loaders
    X_trainTensor = torch.FloatTensor(X_train).to(device)
    X_drop_trainTensor = torch.FloatTensor(X_drop_train).to(device)
    Y_trainTensor = torch.FloatTensor(Y_train).to(device)
    label_trainTensor = torch.tensor(label_train, dtype=torch.int32).to(device)
    # label_train.to(device)

    X_validTensor = torch.FloatTensor(X_valid).to(device)
    X_drop_validTensor = torch.FloatTensor(X_drop_valid).to(device)
    Y_validTensor = torch.FloatTensor(Y_valid).to(device)
    label_validTensor = torch.tensor(label_valid, dtype=torch.int32).to(device)
    # label_valid.to(device)

    X_testTensor = torch.FloatTensor(X_test).to(device)
    X_drop_testTensor = torch.FloatTensor(X_drop_test).to(device)
    Y_testTensor = torch.FloatTensor(Y_test).to(device)
    label_testTensor = torch.tensor(label_test, dtype=torch.int32).to(device)
    # label_test.to(device)

    # Preprocess data to tensor
    train_dataset = TensorDataset(X_trainTensor,X_drop_trainTensor,Y_trainTensor,label_trainTensor)
    # train_dataset = TensorDataset(X_testTensor,X_drop_testTensor,Y_testTensor,label_testTensor)
    valid_dataset = TensorDataset(X_validTensor, X_drop_validTensor,Y_validTensor,label_validTensor)
    test_dataset = TensorDataset(X_testTensor, X_drop_testTensor,Y_testTensor)

    trainDataLoader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    validDataLoader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)
    testDataLoader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    if(str(args.pretrain)!="False"):
        dataloaders_pretrain = {'train':trainDataLoader,'val':validDataLoader}

        Votor_net = AttentionVotor(filters=con_kernel, in_channel=2)

        #logging.info(unet)
        Votor_net.to(device)
        #print(unet)
        optimizer_e = optim.Adam(Votor_net.parameters(), lr=1e-2)
        loss_function_e = nn.MSELoss()
        exp_lr_scheduler_e = lr_scheduler.ReduceLROnPlateau(optimizer_e)

        # Load from checkpoint if checkpoint path is provided
        if(args.checkpoint != "False"):
            load = cbam_pre
        else:
            load = False

        if reduce_model == "AE":
            net,loss_report_en = t.train_CBAM_model(net=Votor_net, data_loaders=dataloaders_pretrain,
                                                    optimizer=optimizer_e, loss_function=loss_function_e,
                                                    n_epochs=epochs, scheduler=exp_lr_scheduler_e, load=load,
                                                    save_path=cbam_pre + '.pkl', center_cells=center_cells)
        elif reduce_model == "VAE":
            net,loss_report_en = t.train_CBAM_model(net=Votor_net, data_loaders=dataloaders_pretrain,
                                                    optimizer=optimizer_e, n_epochs=epochs,
                                                    scheduler=exp_lr_scheduler_e, load=False,
                                                    save_path=cbam_pre + '.pkl', center_cells=center_cells)
        if reduce_model == "DAE":
            net,loss_report_en = t.train_CBAM_model(net=Votor_net, data_loaders=dataloaders_pretrain,
                                                    optimizer=optimizer_e, loss_function=loss_function_e,
                                                    n_epochs=epochs, scheduler=exp_lr_scheduler_e, load=load,
                                                    save_path=cbam_pre + '.pkl', center_cells=center_cells)
                                    
        
        logging.info("Encoder Pretrained finished")
    torch.cuda.empty_cache()

    if(args.drop_rate == 0.0):
        torch.cuda.empty_cache()

        groundTruth_data_mmscaler = preprocessing.MinMaxScaler().fit(groundTruth_data.T)
        X_drop_data_mmscaler = preprocessing.MinMaxScaler().fit(drop_data.T)
        X_all = single_data
        X_drop = X_drop_data_mmscaler.transform(drop_data.T).T
        Y_all = groundTruth_data

        X_allTensor = torch.FloatTensor(X_all).to(device)
        X_drop_allTensor = torch.FloatTensor(X_drop).to(device)
        Y_allTensor = torch.FloatTensor(Y_all).to(device)

        all_dataset = TensorDataset(X_allTensor, X_drop_allTensor, Y_allTensor)
        allDataLoader = DataLoader(dataset=all_dataset, batch_size=batch_size, shuffle=False)
        impute_all, impute_zero, impute_drop = cbam_indicator(allDataLoader, X_drop_data_mmscaler,
                                                              groundTruth_data_mmscaler, net, args.drop_rate)
        #
        row_labels = groundTruth_adata.obs_names
        col_labels = groundTruth_adata.var_names

        #
        df_impute_all = pd.DataFrame(impute_all, index=row_labels, columns=col_labels)
        df_impute_zero = pd.DataFrame(impute_zero, index=row_labels, columns=col_labels)

        #
        df_impute_all.to_csv('./save/result/' + sc_data_name + '/cbam_impute_all_leiden_' +sc_data_name+ '.csv')
        df_impute_zero.to_csv('./save/result/' + sc_data_name + '/cbam_impute_zero_leiden_' +sc_data_name+ '.csv')
    else:

        impute_all, impute_zero, impute_drop = cbam_indicator(testDataLoader, X_drop_test_orgin_mmscaler,
                                                              Y_test_mmscaler, net, args.drop_rate)


        df_impute_all = pd.DataFrame(impute_all)
        df_impute_zero = pd.DataFrame(impute_zero)
        df_impute_drop = pd.DataFrame(impute_drop)

        df_impute_all.to_csv('./save/result/' + sc_data_name + '/Attention_impute_all_drop_'+str(args.drop_rate)+ '_leiden_' + sc_data_name + '.csv')
        df_impute_zero.to_csv(
            './save/result/' + sc_data_name + '/cbam_impute_zero_drop_'+str(args.drop_rate)+ '_leiden_' + sc_data_name + '.csv')
        df_impute_drop.to_csv(
            './save/result/' + sc_data_name + '/cbam_impute_drop_drop_'+str(args.drop_rate)+ '_leiden_' + sc_data_name + '.csv')
    print("cbam_model finished")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--data_name', type=str, default="GSE112274", help='Accession id for testing data, only support pre-built data.')
    parser.add_argument('--drop_rate', type=float, default=0.0,help='Path of the bulk RNA-Seq expression profile')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epoches training. Default: 500')
    parser.add_argument('--test_size', type=float, default=0.2,help='Size of the test set for the bulk model traning, default: 0.2')
    parser.add_argument('--valid_size', type=float, default=0.2,help='Size of the validation set for the bulk model traning, default: 0.2')
    #
    parser.add_argument('--device', type=str, default="gpu",help='Device to train the model. Can be cpu or gpu. Deafult: cpu')
    parser.add_argument('--cbam_pre','-e', type=str, default='save/cbam_pre/',help='Path of the pre-trained encoder in the bulk level')
    parser.add_argument('--pretrain', type=str, default="True",help='Whether to perform pre-training of the encoder,str. False: do not pretraing, True: pretrain. Default: True')
    parser.add_argument('--lr', type=float, default=0.0001,help='Learning rate of model training. Default: 1e-2')
    parser.add_argument('--batch_size', type=int, default=200,help='Number of batch size when training. Default: 200')
    parser.add_argument('--dimreduce', type=str, default="DAE",help='Encoder model type. Can be AE or VAE. Default: AE')
    parser.add_argument('--con_kernel', type=str, default="16,32,64,128",help='Shape of the encoder. Each number represent the number of neuron in a layer. \
                        Layers are seperated by a comma. Default: 16,32,64,128')
    parser.add_argument('--checkpoint', type=str, default='False',help='Load weight from checkpoint files, can be True,False, or file path. Checkpoint files can be paraName1_para1_paraName2_para2... Default: True')
    # misc
    parser.add_argument('--bulk_model', '-p',  type=str, default='save/bulk_pre/',help='Path of the trained prediction model in the bulk level')
    parser.add_argument('--log', '-l',  type=str, default='save/logs/',help='Path of training log')
    parser.add_argument('--dropout', type=float, default=0.0,help='Dropout of neural network. Default: 0.3')
    warnings.filterwarnings("ignore")

    args, unknown = parser.parse_known_args()
    matplotlib.use('Agg')

    # #LOOP
    # # drop_rates = [0.0,0.1,0.2,0.4]
    # drop_rates = [0.0]
    # # # #
    # # sc_data_folder_name = ['GSE112274', 'GSE117872','GSE134836',  'GSE134838', 'GSE134839', 'GSE134841', 'GSE140440','GSE149214']
    # sc_data_folder_name = ['GSE117872']
    # for data_name in sc_data_folder_name:
    #     args.data_name = data_name
    #     print("____________________________________",args.data_name,"________________________________")
    #     for temp_drop_rate in drop_rates:
    #         args.drop_rate = temp_drop_rate
    #         print("____________________________________",args.drop_rate,"________________________________")
    #         torch.cuda.empty_cache()
    #         run_main(args)

    torch.cuda.empty_cache()
    args.data_name = 'GSE134838'
    args.epochs = 200
    # for temp_drop_rate in drop_rates:
    #     print("____________________________________",args.drop_rate,"________________________________")
    #
    args.drop_rate = 0.0
    args.checkpoint = 'data_GSE134838_con_kernel_16,32,64,128_dropRate_0.0_model_DAE.pkl'

    run_main(args)