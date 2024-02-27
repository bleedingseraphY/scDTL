import argparse
import logging
import sys
import time
import warnings
import os
import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch import  nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset

import trainers as t
from models import (AEBase, PretrainedImputor, PretrainedVAEImputor, VAEBase)
import matplotlib
import random
seed=42
torch.manual_seed(seed)
#np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
#from transformers import *
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
#torch.manual_seed(seed)
#torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark=False
def run_main(args):
    if(args.checkpoint not in ["False","True"]):
        selected_model = args.checkpoint
    sc_data_name = args.sc_data_name
    sc_data_path = './data/' + sc_data_name + '/' + sc_data_name + '_4000.h5ad'
    # Extract parameters
    epochs = args.epochs
    dim_au_out = args.bottleneck #8, 16, 32, 64, 128, 256,512
    bulk_data_path = args.bulk_data_path
    test_size = args.test_size
    valid_size = args.valid_size
    log_path = args.log
    batch_size = args.batch_size
    encoder_hdims = args.encoder_h_dims.split(",")
    imputor_hdims = args.imputor_h_dims.split(",")
    reduce_model = args.dimreduce

    encoder_hdims = list(map(int, encoder_hdims))
    imputor_hdims = list(map(int, imputor_hdims))

    para = "data_"+str(sc_data_name)+"_4000_bottle_"+str(args.bottleneck)+"_edim_"+str(args.encoder_h_dims)+"_idim_"+str(args.imputor_h_dims)+"_model_"+reduce_model   #(para)
    now=time.strftime("%Y-%m-%d-%H-%M-%S")

    for path in [args.log,args.bulk_model,args.bulk_encoder,'save/result']:
        if not os.path.exists(path):
            # Create a new directory because it does not exist
            os.makedirs(path)
            print("The new directory is created!")

    # Load model from checkpoint
    if(args.checkpoint not in ["False","True"]):
        args.checkpoint = 'True'

    imputor_path = args.bulk_model + para
    bulk_encoder = args.bulk_encoder+para
    # Read bulk all_data_bulk_value
    data_bulk_all=pd.read_csv(bulk_data_path, index_col=0)
    selected_data_bulk = pd.read_csv(bulk_data_path[:-4]+'_'+sc_data_name+'_4000_sort.csv', index_col=0)

    # Initialize logging
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

    all_data_bulk_value = data_bulk_all.values
    selected_data_bulk_value = selected_data_bulk.values
    # Scaling all_data_bulk_value
    bulk_all_mmscaler = preprocessing.MinMaxScaler().fit(all_data_bulk_value)
    bulk_select_mmscaler = preprocessing.MinMaxScaler().fit(selected_data_bulk_value)
    # selected_data_bulk_value_mmscaler = preprocessing.MinMaxScaler()
    #
    all_data_bulk_value = bulk_all_mmscaler.transform(all_data_bulk_value)
    selected_data_bulk_value = bulk_select_mmscaler.transform(selected_data_bulk_value)

    dim_model_out = selected_data_bulk_value.shape[1]

    logging.info(np.std(all_data_bulk_value))
    logging.info(np.mean(all_data_bulk_value))
    # Split traning valid test set
    X_train_all, X_test = train_test_split(all_data_bulk_value, test_size=test_size, random_state=42)
    Y_train_all, Y_test = train_test_split(selected_data_bulk_value, test_size=test_size, random_state=42)
    X_train, X_valid = train_test_split(X_train_all, test_size=valid_size, random_state=42)
    Y_train, Y_valid = train_test_split(Y_train_all, test_size=valid_size, random_state=42)

    # Select the Training device
    if(args.device == "gpu"):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(device)
    else:
        device = 'cpu'

    # Construct datasets and all_data_bulk_value loaders
    X_trainTensor = torch.FloatTensor(X_train).to(device)
    X_validTensor = torch.FloatTensor(X_valid).to(device)
    X_testTensor = torch.FloatTensor(X_test).to(device)

    Y_trainTensor = torch.FloatTensor(Y_train).to(device)
    Y_validTensor = torch.FloatTensor(Y_valid).to(device)
    Y_testTensor = torch.FloatTensor(Y_test).to(device)

    # Preprocess all_data_bulk_value to tensor
    train_dataset = TensorDataset(X_trainTensor, X_trainTensor)
    valid_dataset = TensorDataset(X_validTensor, X_validTensor)

    X_trainDataLoader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    X_validDataLoader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)

    # construct TensorDataset
    trainreducedDataset = TensorDataset(X_trainTensor, Y_trainTensor)
    validreducedDataset = TensorDataset(X_validTensor, Y_validTensor)

    trainDataLoader_p = DataLoader(dataset=trainreducedDataset, batch_size=batch_size, shuffle=True)
    validDataLoader_p = DataLoader(dataset=validreducedDataset, batch_size=batch_size, shuffle=True)
    bulk_X_allTensor = torch.FloatTensor(all_data_bulk_value).to(device)
    dataloaders_train = {'train':trainDataLoader_p,'val':validDataLoader_p}

    print("bulk_X_allRensor",bulk_X_allTensor.shape)
    if(str(args.pretrain)!="False"):
        dataloaders_pretrain = {'train':X_trainDataLoader,'val':X_validDataLoader}
        if reduce_model == "VAE":
            encoder = VAEBase(input_dim=all_data_bulk_value.shape[1], latent_dim=dim_au_out, h_dims=encoder_hdims, drop_out=args.dropout)
        if reduce_model == 'AE':
            encoder = AEBase(input_dim=all_data_bulk_value.shape[1], latent_dim=dim_au_out, h_dims=encoder_hdims, drop_out=args.dropout)
        if reduce_model =='DAE':            
            encoder = AEBase(input_dim=all_data_bulk_value.shape[1], latent_dim=dim_au_out, h_dims=encoder_hdims, drop_out=args.dropout)

    # Defined the model of imputor
    if reduce_model == "AE":
        model = PretrainedImputor(input_dim=X_train.shape[1], latent_dim=dim_au_out, h_dims=encoder_hdims,
                                  hidden_dims_predictor=imputor_hdims, output_dim=dim_model_out,
                                  pretrained_weights=bulk_encoder, freezed=bool(args.freeze_pretrain), drop_out=args.dropout, drop_out_predictor=args.dropout)
    if reduce_model == "DAE":
        model = PretrainedImputor(input_dim=X_train.shape[1], latent_dim=dim_au_out, h_dims=encoder_hdims,
                                  hidden_dims_predictor=imputor_hdims, output_dim=dim_model_out,
                                  pretrained_weights=bulk_encoder, freezed=bool(args.freeze_pretrain), drop_out=args.dropout, drop_out_predictor=args.dropout)
    elif reduce_model == "VAE":
        model = PretrainedVAEImputor(input_dim=X_train.shape[1], latent_dim=dim_au_out, h_dims=encoder_hdims,
                                     hidden_dims_predictor=imputor_hdims, output_dim=dim_model_out,
                                     pretrained_weights=bulk_encoder, freezed=bool(args.freeze_pretrain), z_reparam=bool(args.VAErepram), drop_out=args.dropout, drop_out_predictor=args.dropout)

    logging.info("Current model is:")
    logging.info(model)

    model.to(device)
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    loss_function = nn.MSELoss()

    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)

    # Train imputor model if load is not false
    if(args.checkpoint != "False"):
        load = True
    else:
        load = False
    print("train imputor")
    model,report = t.train_imputor_model(model, dataloaders_train, optimizer, loss_function, epochs, exp_lr_scheduler,
                                         load=load, save_path=imputor_path)

    print("bulk_model finished")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # data 
    parser.add_argument('--bulk_data_path', type=str, default='data/CCLE.csv',help='Path of the bulk RNA-Seq expression profile')
    parser.add_argument('--sc_data_name', type=str, default="GSE140440",help='Accession id for testing data, only support pre-built data.')

    parser.add_argument('--test_size', type=float, default=0.2,help='Size of the test set for the bulk model traning, default: 0.2')
    parser.add_argument('--valid_size', type=float, default=0.2,help='Size of the validation set for the bulk model traning, default: 0.2')
    # trainv
    parser.add_argument('--device', type=str, default="gpu",help='Device to train the model. Can be cpu or gpu. Deafult: cpu')
    parser.add_argument('--bulk_encoder','-e', type=str, default='save/bulk_encoder/',help='Path of the pre-trained encoder in the bulk level')
    parser.add_argument('--pretrain', type=str, default="True",help='Whether to perform pre-training of the encoder,str. False: do not pretraing, True: pretrain. Default: True')
    parser.add_argument('--lr', type=float, default=1e-2,help='Learning rate of model training. Default: 1e-2')
    parser.add_argument('--epochs', type=int, default=50,help='Number of epoches training. Default: 500')
    parser.add_argument('--batch_size', type=int, default=200,help='Number of batch size when training. Default: 200')
    parser.add_argument('--bottleneck', type=int, default=2048,help='Size of the bottleneck layer of the model. Default: 2048')
    parser.add_argument('--dimreduce', type=str, default="DAE",help='Encoder model type. Can be AE or VAE or DAE. Default: DAE')
    parser.add_argument('--freeze_pretrain', type=int, default=0,help='Fix the prarmeters in the pretrained model. 0: do not freeze, 1: freeze. Default: 0')
    parser.add_argument('--encoder_h_dims', type=str, default="4096,2048",help='Shape of the encoder. Each number represent the number of neuron in a layer. \
                        Layers are seperated by a comma. Default: 512,256')
    parser.add_argument('--imputor_h_dims', type=str, default="2048,4096",help='Shape of the predictor. Each number represent the number of neuron in a layer. \
                        Layers are seperated by a comma. Default: 16,8')
    parser.add_argument('--VAErepram', type=int, default=1)
    parser.add_argument('--checkpoint', type=str, default='True',help='Load weight from checkpoint files, can be True,False, or file path. Checkpoint files can be paraName1_para1_paraName2_para2... Default: True')
    # misc
    parser.add_argument('--bulk_model', '-p',  type=str, default='save/bulk_pre/',help='Path of the trained prediction model in the bulk level')
    parser.add_argument('--log', '-l',  type=str, default='save/logs/',help='Path of training log')
    parser.add_argument('--dropout', type=float, default=0.3,help='Dropout of neural network. Default: 0.3')
    warnings.filterwarnings("ignore")
    args, unknown = parser.parse_known_args()
    matplotlib.use('Agg')

    args.sc_data_name = 'GSE134838'
    args.checkpoint = 'data_GSE134838_4000_bottle_2048_edim_4096,2048_idim_2048,4096_model_DAE'
    run_main(args)

