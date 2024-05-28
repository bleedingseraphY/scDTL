#!/usr/bin/env python
# coding: utf-8
import argparse
import pandas as pd
from pandas.core.frame import DataFrame
import logging
import os
import sys
import time
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
import DTL.mmd as mmd
import scanpypip.preprocessing as pp
import trainers as t
from captum.attr import IntegratedGradients
from models import (AEBase, DTL, PretrainedImputor,
                    PretrainedVAEImputor, VAEBase)
from scipy.spatial import distance_matrix, minkowski_distance, distance
import random
seed = 42
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
# sc_data_names = ['GSE112274', 'GSE117872', 'GSE134836','GSE134838', 'GSE134839', 'GSE134841', 'GSE140440',
#                      'GSE149214']

def output_to_csv(sc_data_path,bulk_select_mmscaler,device,encoder,source_model,X_allTensor,single_adata):
    #dropout
    drop_List = [0.0, 0.4]
    #
    row_labels = single_adata.obs.axes[0]
    col_labels = single_adata.var.axes[0]

    for drop_rate in drop_List:
        if(drop_rate != 0.0):
            dropout_data = sc.read_h5ad(sc_data_path[:-5]+'_drop_'+str(drop_rate)+'.h5ad')
            dropout_data_value = bulk_select_mmscaler.transform(dropout_data.X)
            X_allTensor = torch.FloatTensor(dropout_data_value).to(device)

        X_all_dataset =  TensorDataset(X_allTensor)
        X_all_dataloader =  DataLoader(dataset=X_all_dataset, batch_size=args.batch_size, shuffle=False)

        prediction_tensors = []
        for batchidx, x in enumerate(X_all_dataloader):
            x = x[0]
            with torch.no_grad():
                embedding_tensors = encoder.encode(x)
                prediction_tensor = source_model.predictor(embedding_tensors)

                if (batchidx == 0):
                    prediction_tensors = prediction_tensor
                else:
                    prediction_tensors = torch.cat([prediction_tensors, prediction_tensor], dim=0)

        predictions = prediction_tensors.detach().cpu().numpy()

        #inverse_transform
        predictions = bulk_select_mmscaler.inverse_transform(predictions)
        impute_all = predictions.copy()
        impute_zero = predictions.copy()

        groundTruth = X_allTensor.detach().cpu().numpy()
        groundTruth = bulk_select_mmscaler.inverse_transform(groundTruth)
        no_zero_i, no_zero_j = np.nonzero(groundTruth)
        impute_zero[no_zero_i, no_zero_j] = groundTruth[no_zero_i, no_zero_j]

        # df_predictions = pd.DataFrame(predictions, index=row_labels, columns=col_labels)
        df_impute_all = pd.DataFrame(impute_all, index=row_labels, columns=col_labels)
        df_impute_zero = pd.DataFrame(impute_zero, index=row_labels, columns=col_labels)
        # save
        csv_file_path_impute_all = './save/result/' + args.sc_data_name + '/P_impute_all_' + str(drop_rate)+ '.csv'
        csv_file_path_impute_zero = './save/result/' + args.sc_data_name + '/P_impute_zero_' + str(drop_rate)+ '.csv'
        df_impute_all.to_csv(csv_file_path_impute_all)
        df_impute_zero.to_csv(csv_file_path_impute_zero)
    return True
def run_main(args):
    t0 = time.time()
    # checkpoint
    if(args.checkpoint not in ["False","True"]):
        selected_model = args.checkpoint
    # Laod parameters from args
    epochs = args.epochs
    dim_au_out = args.bottleneck #8, 16, 32, 64, 128, 256,512
    sc_data_path = './data/' + args.sc_data_name + '/' + args.sc_data_name + '_4000.h5ad'
    test_size = args.test_size
    freeze = args.freeze_pretrain
    valid_size = args.valid_size
    log_path = args.logging_file
    batch_size = args.batch_size
    encoder_hdims = args.bulk_h_dims.split(",")
    encoder_hdims = list(map(int, encoder_hdims))
    sc_data_name = args.sc_data_name
    reduce_model = args.dimreduce
    imputor_hdims = args.imputor_h_dims.split(",")
    imputor_hdims = list(map(int, imputor_hdims))
    # Merge parameters as string for saving model and logging
    para = "data_"+str(sc_data_name)+"_4000_bottle_"+str(args.bottleneck)+"_edim_"+str(args.bulk_h_dims)+"_idim_"+str(args.imputor_h_dims)+"_model_"+reduce_model
    bulk_data_path = args.bulk_data
    
    # Record time
    now=time.strftime("%Y-%m-%d-%H-%M-%S")
    # Initialize logging and std out
    out_path = log_path+now+"transfer.err"
    log_path = log_path+now+"transfer.log"
    out=open(out_path,"w")
    sys.stderr=out
    #Logging parameters
    logging.basicConfig(level=logging.INFO,
                    filename=log_path,
                    filemode='a',
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.info(args)
    logging.info("Start at " + str(t0))

    # Create directories if they do not exist
    for path in [args.logging_file,args.bulk_model_path,args.sc_model_path,args.sc_encoder_path]:
        if not os.path.exists(path):
            # Create a new directory because it does not exist
            os.makedirs(path)
            print("The new directory is created!")
    
    # Save arguments
    # Overwrite params if checkpoint is provided
    if(args.checkpoint not in ["False","True"]):
        args.checkpoint = "True"

    sc_encoder_path = args.sc_encoder_path+para
    bulk_model_path = args.bulk_model_path + para
    print(bulk_model_path)
    target_model_path = args.sc_model_path +para
    # Load single single_data and preprocessing
    single_adata = sc.read_h5ad(sc_data_path)
    # Read select bulk data
    bulk_data = pd.read_csv(bulk_data_path, index_col=0)
    select_bulk_data = pd.read_csv(bulk_data_path[:-4]+'_'+sc_data_name+'_4000_sort.csv', index_col=0)
    # Process source single_data
    bulk_all_mmscaler = preprocessing.MinMaxScaler().fit(bulk_data.values)
    bulk_select_mmscaler = preprocessing.MinMaxScaler().fit(select_bulk_data.values)

    bulk_all_data = bulk_all_mmscaler.transform(bulk_data.values)
    bulk_select_data = bulk_select_mmscaler.transform(select_bulk_data.values)
    single_data = bulk_select_mmscaler.transform(single_adata.X)

    # Split single_data to train and valid set
    Xtarget_train, Xtarget_valid, Ytarget_train, Ytarget_valid = train_test_split(single_data, single_data, test_size=valid_size, random_state=42)
    # Select the device of gpu
    if(args.device == "gpu"):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(device)
    else:
        device = 'cpu'
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    logging.info(device)
    # Construct datasets and single_data loaders
    Xtarget_trainTensor = torch.FloatTensor(Xtarget_train).to(device)
    Xtarget_validTensor = torch.FloatTensor(Xtarget_valid).to(device)

    Ctarget_trainTensor = torch.FloatTensor(Ytarget_train).to(device)
    Ctarget_validTensor = torch.FloatTensor(Ytarget_valid).to(device)
    #print("C",Ctarget_validTensor )
    X_allTensor = torch.FloatTensor(single_data).to(device)
    
    train_dataset = TensorDataset(Xtarget_trainTensor, Ctarget_trainTensor)
    valid_dataset = TensorDataset(Xtarget_validTensor, Ctarget_validTensor)

    Xtarget_trainDataLoader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    Xtarget_validDataLoader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)

    dataloaders_pretrain = {'train':Xtarget_trainDataLoader,'val':Xtarget_validDataLoader}

    # Split source single_data
    Xsource_train_all, Xsource_test, Ysource_train_all, Ysource_test = train_test_split(bulk_all_data, bulk_select_data, test_size=test_size, random_state=42)
    Xsource_train, Xsource_valid, Ysource_train, Ysource_valid = train_test_split(Xsource_train_all,Ysource_train_all, test_size=valid_size, random_state=42)

    # Transform source single_data
    # Construct datasets and single_data loaders
    Xsource_trainTensor = torch.FloatTensor(Xsource_train).to(device)
    Xsource_validTensor = torch.FloatTensor(Xsource_valid).to(device)

    Ysource_trainTensor = torch.FloatTensor(Ysource_train).to(device)
    Ysource_validTensor = torch.FloatTensor(Ysource_valid).to(device)

    sourcetrain_dataset = TensorDataset(Xsource_trainTensor, Ysource_trainTensor)
    sourcevalid_dataset = TensorDataset(Xsource_validTensor, Ysource_validTensor)

    Xsource_trainDataLoader = DataLoader(dataset=sourcetrain_dataset, batch_size=batch_size, shuffle=True)
    Xsource_validDataLoader = DataLoader(dataset=sourcevalid_dataset, batch_size=batch_size, shuffle=True)

    dataloaders_source = {'train':Xsource_trainDataLoader,'val':Xsource_validDataLoader}
    # Construct target encoder
    if reduce_model == "AE":
        encoder = AEBase(input_dim=single_data.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims,drop_out=args.dropout)

    elif reduce_model == "VAE":
        encoder = VAEBase(input_dim=single_data.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims,drop_out=args.dropout)
    if reduce_model == "DAE":
        encoder = AEBase(input_dim=single_data.shape[1],latent_dim=dim_au_out,h_dims=encoder_hdims,drop_out=args.dropout)


    logging.info("Target encoder structure is: ")
    logging.info(encoder)
    
    encoder.to(device)
    optimizer_e = optim.Adam(encoder.parameters(), lr=1e-2)
    loss_function_e = nn.MSELoss()
    exp_lr_scheduler_e = lr_scheduler.ReduceLROnPlateau(optimizer_e)

    dim_model_out = Ysource_trainTensor.shape[1]
    # Load the trained source encoder and imputor
    if reduce_model == "AE":
        source_model = PretrainedImputor(input_dim=Xsource_train.shape[1], latent_dim=dim_au_out, h_dims=encoder_hdims,
                                         hidden_dims_predictor=imputor_hdims, output_dim=dim_model_out,
                                         pretrained_weights=None, freezed=freeze, drop_out=args.dropout, drop_out_predictor=args.dropout)
        
        source_model.load_state_dict(torch.load(bulk_model_path))
        source_encoder = source_model
    if reduce_model == "DAE":
        source_model = PretrainedImputor(input_dim=Xsource_train.shape[1], latent_dim=dim_au_out, h_dims=encoder_hdims,
                                         hidden_dims_predictor=imputor_hdims, output_dim=dim_model_out,
                                         pretrained_weights=None, freezed=freeze, drop_out=args.dropout, drop_out_predictor=args.dropout)

        source_model.load_state_dict(torch.load(bulk_model_path))
        source_encoder = source_model    
    # Load VAE model
    elif reduce_model in ["VAE"]:
        source_model = PretrainedVAEImputor(input_dim=Xsource_train.shape[1], latent_dim=dim_au_out, h_dims=encoder_hdims,
                                            hidden_dims_predictor=imputor_hdims, output_dim=dim_model_out,
                                            pretrained_weights=None, freezed=freeze, z_reparam=bool(args.VAErepram), drop_out=args.dropout, drop_out_predictor=args.dropout)
        source_model.load_state_dict(torch.load(bulk_model_path))
        source_encoder = source_model

    source_encoder.to(device)
    # Pretrain target encoder training
    # Pretain using autoencoder is pretrain is not False
    if(str(args.sc_encoder_path)!='False'):
        # Pretrained target encoder if there are not stored files in the harddisk
        train_flag = True
        sc_encoder_path = str(sc_encoder_path)
        print("Pretrain=="+sc_encoder_path)
        
        # If pretrain is not False load from check point
        if(args.checkpoint!="False"):
            # if checkpoint is not False, load the pretrained model
            try:
                encoder.load_state_dict(torch.load(sc_encoder_path))
                logging.info("Load pretrained target encoder from "+sc_encoder_path)
                train_flag = False

            except:
                logging.info("Loading failed, procceed to re-train model")
                train_flag = True

        # If pretrain is not False and checkpoint is False, retrain the model
        if train_flag == True:

            if reduce_model == "AE":
                encoder,loss_report_en = t.train_AE_model(net=encoder,data_loaders=dataloaders_pretrain,
                                            optimizer=optimizer_e,loss_function=loss_function_e,load=False,
                                            n_epochs=epochs,scheduler=exp_lr_scheduler_e,save_path=sc_encoder_path)
            if reduce_model == "DAE":
                encoder,loss_report_en = t.train_DAE_model(net=encoder,data_loaders=dataloaders_pretrain,
                                            optimizer=optimizer_e,loss_function=loss_function_e,load=False,
                                            n_epochs=epochs,scheduler=exp_lr_scheduler_e,save_path=sc_encoder_path)
                                            
            elif reduce_model == "VAE":
                encoder,loss_report_en = t.train_VAE_model(net=encoder,data_loaders=dataloaders_pretrain,
                                optimizer=optimizer_e,load=False,
                                n_epochs=epochs,scheduler=exp_lr_scheduler_e,save_path=sc_encoder_path)
            # print(loss_report_en)
            logging.info("Pretrained finished")
        #single_data:single bulk_data:bulk  encoder:target  source_encoder  dataloaders_pretrain:target
        # Before Transfer learning, we test the performance of using no transfer performance:
        # Use vae result to predict
        #test
        embeddings_pretrain = encoder.encode(X_allTensor)
        # print(embeddings_pretrain)
        pretrain_prob_prediction = source_model.predict(embeddings_pretrain).detach().cpu().numpy()
        single_adata.obs["sens_preds_pret"] = pretrain_prob_prediction[:, 1]
        single_adata.obs["sens_label_pret"] = pretrain_prob_prediction.argmax(axis=1)

        # Add embeddings to the single_adata object
        embeddings_pretrain = embeddings_pretrain.detach().cpu().numpy()
        single_adata.obsm["X_pre"] = embeddings_pretrain
    # Using DTL transfer learning
    # DTL model
    # Set predictor loss
    loss_d = nn.MSELoss()
    optimizer_d = optim.Adam(encoder.parameters(), lr=1e-2)
    exp_lr_scheduler_d = lr_scheduler.ReduceLROnPlateau(optimizer_d)
       
    # Set DTL model
    DTL_model = DTL(source_model=source_encoder, target_model=encoder, fix_source=bool(args.fix_source))
    DTL_model.to(device)

    # Set distribution loss 
    def loss(x,y,GAMMA=args.mmd_GAMMA):
        result = mmd.mmd_loss(x,y,GAMMA)
        return result

    loss_disrtibution = loss

    # Train DTL model with new loss function
    if args.checkpoint == 'True':
        DTL_model, report_, _, _ = t.train_DTL_model(DTL_model, dataloaders_source, dataloaders_pretrain, optimizer_d,
                                                     loss_d, epochs, exp_lr_scheduler_d, dist_loss=loss_disrtibution,
                                                     weight=args.mmd_weight, load=selected_model,
                                                     save_path=target_model_path + "_DaNN.pkl")
    else:
        DTL_model, report_, _, _ = t.train_DTL_model(DTL_model, dataloaders_source, dataloaders_pretrain, optimizer_d,
                                                     loss_d, epochs, exp_lr_scheduler_d, dist_loss=loss_disrtibution,
                                                     weight=args.mmd_weight, load=False,
                                                     save_path=target_model_path + "_DaNN.pkl", device=device)
    encoder = DTL_model.target_model
    source_model = DTL_model.source_model

    torch.cuda.empty_cache()
    output_to_csv(sc_data_path,bulk_select_mmscaler,device,encoder,source_model,X_allTensor,single_adata)
    print("sc model finished")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data 
    parser.add_argument('--bulk_data', type=str, default='data/CCLE.csv',help='Path of the bulk RNA-Seq expression profile')
    parser.add_argument('--sc_data_name', type=str, default="GSE117872",help='Accession id for testing data, only support pre-built data.')

    parser.add_argument('--test_size', type=float, default=0.2,help='Size of the test set for the bulk model traning, default: 0.2')
    parser.add_argument('--valid_size', type=float, default=0.2,help='Size of the validation set for the bulk model traning, default: 0.2')

    parser.add_argument('--mmd_weight', type=float, default=0.25,help="Weight of the MMD loss of the transfer learning, default: 0.25")
    parser.add_argument('--mmd_GAMMA', type=int, default=1000,help="Gamma parameter in the kernel of the MMD loss of the transfer learning, default: 1000")
    # train
    parser.add_argument('--device', type=str, default="gpu",help='Device to train the model. Can be cpu or gpu. Deafult: cpu')
    parser.add_argument('--bulk_model_path','-s', type=str, default='save/bulk_pre/',help='Path of the trained predictor in the bulk level')
    parser.add_argument('--sc_model_path', '-p',  type=str, default='save/sc_pre/',help='Path (prefix) of the trained predictor in the single cell level')
    parser.add_argument('--sc_encoder_path', type=str, default='save/sc_encoder/',help='Path of the pre-trained encoder in the single-cell level')
    parser.add_argument('--checkpoint', type=str, default='True',help='Load weight from checkpoint files, can be True,False, or a file path. Checkpoint files can be paraName1_para1_paraName2_para2... Default: True')

    parser.add_argument('--lr', type=float, default=1e-2,help='Learning rate of model training. Default: 1e-2')
    parser.add_argument('--epochs', type=int, default=50,help='Number of epoches training. Default: 500')
    parser.add_argument('--batch_size', type=int, default=200,help='Number of batch size when training. Default: 200')
    parser.add_argument('--bottleneck', type=int, default=512,help='Size of the bottleneck layer of the model. Default: 32')
    parser.add_argument('--dimreduce', type=str, default="DAE",help='Encoder model type. Can be DAE or VAE. Default: DAE')
    parser.add_argument('--freeze_pretrain', type=int,default=0,help='Fix the prarmeters in the pretrained model. 0: do not freeze, 1: freeze. Default: 0')
    parser.add_argument('--bulk_h_dims', type=str, default="4096,2048",help='Shape of the source encoder. Each number represent the number of neuron in a layer. \
                        Layers are seperated by a comma. Default: 512,256')
    parser.add_argument('--sc_h_dims', type=str, default="4096,2048",help='Shape of the encoder. Each number represent the number of neuron in a layer. \
                        Layers are seperated by a comma. Default: 512,256')
    parser.add_argument('--imputor_h_dims', type=str, default="16,8",help='Shape of the predictor. Each number represent the number of neuron in a layer. \
                        Layers are seperated by a comma. Default: 16,8')
    parser.add_argument('--VAErepram', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.3,help='Dropout of neural network. Default: 0.3')
    # miss
    parser.add_argument('--logging_file', '-l',  type=str, default='save/logs/',help='Path of training log')
    parser.add_argument('--fix_source', type=int, default=0,help='Fix the bulk level model. Default: 0')
    #
    args, unknown = parser.parse_known_args()


    args.sc_data_name = "GSE134838"
    args.dimreduce = "DAE"
    args.bulk_h_dims = "4096,2048"
    args.bottleneck = 2048
    args.imputor_h_dims = "2048,4096"
    args.dropout = 0.0
    args.checkpoint = "False"
    args.epochs = 200
    run_main(args)
    # sc_data_folder_name = ['GSE81861', 'GSM3618014']
    # for data_name in sc_data_folder_name:
    #     args.sc_data_name = data_name
    #     print(args.sc_data_name)
    #     torch.cuda.empty_cache()
    #     run_main(args)

