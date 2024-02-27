import copy
import logging
import os

import numpy as np

import torch
from torch import nn
from tqdm import tqdm

from models import vae_loss
from sklearn.metrics.pairwise import cosine_similarity
### loss2
import copy

from scipy.spatial import distance_matrix, minkowski_distance, distance
import networkx as nx
from igraph import *



def train_AE_model(net,data_loaders={},optimizer=None,loss_function=None,n_epochs=100,scheduler=None,load=False,save_path="model.pkl"):
    if(load!=False):
        if(os.path.exists(save_path)):
            net.load_state_dict(torch.load(save_path))           
            return net, 0
        else:
            logging.warning("Failed to load existing file, proceed to the trainning process.")
    
    dataset_sizes = {x: data_loaders[x].dataset.tensors[0].shape[0] for x in ['train', 'val']}
    loss_train = {}
    
    best_model_wts = copy.deepcopy(net.state_dict())
    best_loss = np.inf

    for epoch in range(n_epochs):
        logging.info('Epoch {}/{}'.format(epoch, n_epochs - 1))
        logging.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #optimizer = scheduler(optimizer, epoch)
                net.train()  # Set model to training mode
            else:
                net.eval()  # Set model to evaluate mode

            running_loss = 0.0

            n_iters = len(data_loaders[phase])
            # Iterate over data.
            # for data in data_loaders[phase]:
            for batchidx, (x, _) in enumerate(data_loaders[phase]):

                x.requires_grad_(True)
                # encode and decode 
                #print(x)
                output = net(x)
                # compute loss
                loss = loss_function(output, x)      

                # zero the parameter (weight) gradients
                optimizer.zero_grad()

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    # update the weights
                    optimizer.step()

                # print loss statistics
                running_loss += loss.item()
            
  
            epoch_loss = running_loss / n_iters
            #print(epoch_loss)
            
            if phase == 'train':
                scheduler.step(epoch_loss)
                
            last_lr = scheduler.optimizer.param_groups[0]['lr']
            loss_train[epoch,phase] = epoch_loss
            logging.info('{} Loss: {:.8f}. Learning rate = {}'.format(phase, epoch_loss,last_lr))
            
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(net.state_dict())
    
    # Select best model wts
    torch.save(best_model_wts, save_path)
    net.load_state_dict(best_model_wts)           
    
    return net, loss_train
    
def train_DAE_model(net,data_loaders={},optimizer=None,loss_function=None,n_epochs=100,scheduler=None,load=False,save_path="model.pkl"):
    
    if(load!=False):
        if(os.path.exists(save_path)):
            net.load_state_dict(torch.load(save_path))           
            return net, 0
        else:
            logging.warning("Failed to load existing file, proceed to the trainning process.")
    
    dataset_sizes = {x: data_loaders[x].dataset.tensors[0].shape[0] for x in ['train', 'val']}
    loss_train = {}
    
    best_model_wts = copy.deepcopy(net.state_dict())
    best_loss = np.inf

    for epoch in range(n_epochs):
        logging.info('Epoch {}/{}'.format(epoch, n_epochs - 1))
        logging.info('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #optimizer = scheduler(optimizer, epoch)
                net.train()  # Set model to training mode
            else:
                net.eval()  # Set model to evaluate mode

            running_loss = 0.0

            n_iters = len(data_loaders[phase])


            # Iterate over data.
            # for data in data_loaders[phase]:
            for batchidx, (x, _) in enumerate(data_loaders[phase]):
     
                z = x
                y = np.random.binomial(1, 0.4, (z.shape[0], z.shape[1]))
                z[np.array(y, dtype= bool),] = 0
                x.requires_grad_(True)
                # encode and decode 
                output = net(z)
                # compute loss
                # print("bulk_AE:", output)
                loss = loss_function(output, x)
                # print("bulk_AE_Loss:", loss)

                # zero the parameter (weight) gradients
                optimizer.zero_grad()
                # for param in net.parameters():
                #     print(param.grad)
                # backward + optimize only if in training phase
                if phase == 'train':
                    try:
                        loss.backward()
                    except Exception as e:
                        print(e)
                        import traceback
                        traceback.print_exc()

                    # update the weights
                    optimizer.step()

                # print loss statistics
                running_loss += loss.item()
            
  
            epoch_loss = running_loss / n_iters

            print(epoch_loss)
            if phase == 'train':
                scheduler.step(epoch_loss)
                
            last_lr = scheduler.optimizer.param_groups[0]['lr']
            loss_train[epoch,phase] = epoch_loss
            logging.info('{} Loss: {:.8f}. Learning rate = {}'.format(phase, epoch_loss,last_lr))
            
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(net.state_dict())
    
    # Select best model wts
    torch.save(best_model_wts, save_path)
    net.load_state_dict(best_model_wts)           
    
    return net, loss_train

def pearson_corr(imputed_data, original_data):
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
def train_Unet_model(net, data_loaders={}, optimizer=None, loss_function=None, n_epochs=100, scheduler=None, load=False,
                    save_path="model.pkl"):
    if (load != False):
        if (os.path.exists(save_path)):
            net.load_state_dict(torch.load(save_path))
            # return net, 0
        else:
            logging.warning("Failed to load existing file, proceed to the trainning process.")

    # dataset_sizes = {x: data_loaders[x].dataset.tensors[0].shape[0] for x in ['train', 'val']}
    loss_train = {}

    best_model_wts = copy.deepcopy(net.state_dict())
    best_loss = np.inf

    for epoch in range(n_epochs):
        logging.info('Epoch {}/{}'.format(epoch, n_epochs - 1))
        logging.info('-' * 10)

        # Each epoch has a training and validation phase
        # for phase in ['train', 'val']:
        for phase in ['train']:
            if phase == 'train':
                # optimizer = scheduler(optimizer, epoch)
                net.train()  # Set model to training mode
            else:
                net.eval()  # Set model to evaluate mode

            running_loss = 0.0

            n_iters = len(data_loaders[phase])

            # Iterate over data.
            # for data in data_loaders[phase]:
            outputs = []
            ys = []
            for batchidx, (x,y) in enumerate(data_loaders[phase]):

                # encode and decode
                output = net(x)
                if(batchidx == 0):
                    outputs = output
                    ys = y
                else:
                    outputs = torch.cat([outputs,output],dim=0)
                    ys = torch.cat([ys,y],dim=0)
                # print(output)
                loss_mse = loss_function(output,y)
                loss_pear = 1 - pearson_corr(output,y)

                print(loss_pear)
                edgeList = calculateKNNgraphDistanceMatrix(y.cpu().detach().numpy(), distanceType='euclidean',
                                                           k=10)
                listResult, size = generateLouvainCluster(edgeList)
                # sc sim loss
                loss_s = 0.0
                for i in range(size):
                    # print(i)
                    s = cosine_similarity(output[np.asarray(listResult) == i, :].cpu().detach().numpy())
                    s = 1 - s
                    loss_s += np.sum(np.triu(s, 1)) / ((s.shape[0] * s.shape[0]) * 2 - s.shape[0])
                # loss_s = torch.tensor(loss_s).cuda()

                loss_s = torch.tensor(loss_s).cuda()
                loss_s.requires_grad_(True)

                loss = loss_pear + loss_mse
                # compute loss
                # print("bulk_AE_Loss:", loss)

                # zero the parameter (weight) gradients
                optimizer.zero_grad()
                # for param in net.parameters():
                #     print(param.grad)
                # backward + optimize only if in training phase
                # if phase == 'train':
                #     try:
                #         loss.backward()
                #     except Exception as e:
                #         print(e)
                #         import traceback
                #         traceback.print_exc()
                loss.backward()
                # update the weights
                optimizer.step()

                # print loss statistics
                # running_loss += loss.item()
                running_loss += loss.item()
            print("all_PCCS:",pearson_corr(outputs,ys))
            epoch_loss = running_loss / n_iters

            # print(epoch_loss)
            if phase == 'train':
                scheduler.step(epoch_loss)

            last_lr = scheduler.optimizer.param_groups[0]['lr']
            loss_train[epoch, phase] = epoch_loss
            logging.info('{} Loss: {:.8f}. Learning rate = {}'.format(phase, epoch_loss, last_lr))

            if phase == 'train' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(net.state_dict())

    # Select best model wts
    # torch.save(net.state_dict(), save_path)
    # net.load_state_dict(net.state_dict())
    torch.save(best_model_wts, save_path)
    net.load_state_dict(best_model_wts)

    return net, loss_train

def train_CBAM_model(net, data_loaders={}, optimizer=None, loss_function=None, n_epochs=100, scheduler=None, load=False,
                    save_path="model.pkl",center_cells = None):
    if (load != False):
        if (os.path.exists(save_path)):
            net.load_state_dict(torch.load(save_path))
            return net, 0
        else:
            logging.warning("Failed to load existing file, proceed to the trainning process.")

    # dataset_sizes = {x: data_loaders[x].dataset.tensors[0].shape[0] for x in ['train', 'val']}
    loss_train = {}

    best_model_wts = copy.deepcopy(net.state_dict())
    best_loss = np.inf

    for epoch in range(n_epochs):
        logging.info('Epoch {}/{}'.format(epoch, n_epochs - 1))
        logging.info('-' * 10)

        # Each epoch has a training and validation phase
        # for phase in ['train', 'val']:
        for phase in ['train','val']:
            if phase == 'train':
                # optimizer = scheduler(optimizer, epoch)
                net.train()  # Set model to training mode
            else:
                net.eval()  # Set model to evaluate mode

            running_loss = 0.0

            n_iters = len(data_loaders[phase])

            for batchidx, (x,x_drop,y,labels) in enumerate(data_loaders[phase]):

                output = net(x,x_drop)
                # if(batchidx == 0):
                #     outputs = output
                #     ys = y
                # else:
                #     outputs = torch.cat([outputs,output],dim=0)
                #     ys = torch.cat([ys,y],dim=0)
                # print(output)
                # weight = torch.ones_like(x_drop)
                # nonzero_indices = torch.nonzero(x_drop)
                # zero_indices = torch.nonzero(y==0)
                # nozero_i, nozero_j = nonzero_indices.t()
                # zero_i, zero_j = zero_indices.t()
                # weight[nozero_i,nozero_j] = 0.0
                # weight[zero_i,zero_j] = 0.0
                #
                # output = torch.mul(output, weight)
                # y = torch.mul(y, weight)

                # temp = (output-y)**2
                # temp = temp * weight
                # loss_mse = torch.mean(temp)
                loss_mse = loss_function(output,y)
                loss_pear = 1 - pearson_corr(output,y)

                # if(y.shape[0] >10):
                #     edgeList = calculateKNNgraphDistanceMatrix(y.cpu().detach().numpy(), distanceType='euclidean',
                #                                                k=10)
                #     listResult, size = generateLouvainCluster(edgeList)
                #     # sc sim loss
                #     loss_s = 0.0
                #     for i in range(size):
                #         # print(i)
                #         s = cosine_similarity(output[np.asarray(listResult) == i, :].cpu().detach().numpy())
                #         s = 1 - s
                #         loss_s += np.sum(np.triu(s, 1)) / ((s.shape[0] * s.shape[0]) * 2 - s.shape[0])
                # # loss_s = torch.tensor(loss_s).cuda()
                # else:
                #     loss_s=0.0
                loss_s = 0.0
                loss_o = 0.0
                unique_labels, label_counts = torch.unique(labels, return_counts=True)
                label_counts = label_counts.detach().cpu().numpy()

                for i in range(unique_labels.shape[0]):
                    # temp = output[labels == unique_label, :]
                    # s = cosine_similarity(output[labels == unique_label, :].cpu().detach().numpy())
                    # s = 1 - s
                    # loss_s += np.sum(np.triu(s, 1)) / ((s.shape[0] * s.shape[0] - s.shape[0])/2)
                    excluded_key = i
                    center_other_cells = {key: value for key, value in center_cells.items() if key != excluded_key}

                    loss_temp_o = 0.0
                    for center_other_cell in center_other_cells.values():
                        loss_temp_o = loss_temp_o + loss_function(output[labels == unique_labels[i], :], center_other_cell.repeat(label_counts[i], 1))
                    loss_o = loss_o + loss_temp_o
                    loss_s = loss_s + loss_function(output[labels == unique_labels[i], :], center_cells[i].repeat(label_counts[i], 1))
                # loss_s = torch.tensor(loss_s).cuda()
                loss_s.requires_grad_(True)
                # loss_o.requires_grad_(True)

                # loss = loss_pear + loss_mse + 100*loss_s
                loss = loss_pear + loss_mse + loss_s
                print(f"total_loss: {loss.item()}, loss_s: {loss_s.item()}, loss_o: {loss_o.item()}, loss_pear: {loss_pear.item()}, loss_mse: {loss_mse.item()}")
                # compute loss
                # print("bulk_AE_Loss:", loss)

                # zero the parameter (weight) gradients
                optimizer.zero_grad()
                # for param in net.parameters():
                #     print(param.grad)
                # backward + optimize only if in training phase
                loss.backward()
                # loss.backward()
                # update the weights
                optimizer.step()

                # print loss statistics
                # running_loss += loss.item()
                running_loss += loss.item()
            # print("all_PCCS:",pearson_corr(outputs,ys))
            epoch_loss = running_loss / n_iters

            # print(epoch_loss)
            # if phase == 'train':
            #     scheduler.step(epoch_loss)

            last_lr = scheduler.optimizer.param_groups[0]['lr']
            loss_train[epoch, phase] = epoch_loss
            logging.info('{} Loss: {:.8f}. Learning rate = {}'.format(phase, epoch_loss, last_lr))

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(net.state_dict())
    # Select best model wts
    # torch.save(net.state_dict(), save_path)
    # net.load_state_dict(net.state_dict())
    torch.save(best_model_wts, save_path)
    net.load_state_dict(best_model_wts)

    return net, loss_train
def train_VAE_model(net,data_loaders={},optimizer=None,n_epochs=100,scheduler=None,load=False,save_path="model.pkl",best_model_cache = "drive"):
    
    if(load!=False):
        if(os.path.exists(save_path)):
            net.load_state_dict(torch.load(save_path))           
            return net, 0
        else:
            logging.warning("Failed to load existing file, proceed to the trainning process.")
    
    dataset_sizes = {x: data_loaders[x].dataset.tensors[0].shape[0] for x in ['train', 'val']}
    loss_train = {}
    
    if best_model_cache == "memory":
        best_model_wts = copy.deepcopy(net.state_dict())
    else:
        torch.save(net.state_dict(), save_path+"_bestcahce.pkl")
    
    best_loss = np.inf

    for epoch in range(n_epochs):
        logging.info('Epoch {}/{}'.format(epoch, n_epochs - 1))
        logging.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #optimizer = scheduler(optimizer, epoch)
                net.train()  # Set model to training mode
            else:
                net.eval()  # Set model to evaluate mode

            running_loss = 0.0

            n_iters = len(data_loaders[phase])


            # Iterate over data.
            # for data in data_loaders[phase]:
            for batchidx, (x, _) in enumerate(data_loaders[phase]):

                x.requires_grad_(True)
                # encode and decode 
                output = net(x)
                # compute loss

                #losses = net.loss_function(*output, M_N=data_loaders[phase].batch_size/dataset_sizes[phase])      
                #loss = losses["loss"]

                recon_loss = nn.MSELoss(reduction="sum")

                loss = vae_loss(output[0],output[1],output[2],output[3],recon_loss,data_loaders[phase].batch_size/dataset_sizes[phase])

                # zero the parameter (weight) gradients
                optimizer.zero_grad()

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    # update the weights
                    optimizer.step()

                # print loss statistics
                running_loss += loss.item()
            
                
            epoch_loss = running_loss / dataset_sizes[phase]
            #epoch_loss = running_loss / n_iters

            
            if phase == 'train':
                scheduler.step(epoch_loss)
                
            last_lr = scheduler.optimizer.param_groups[0]['lr']
            loss_train[epoch,phase] = epoch_loss
            logging.info('{} Loss: {:.8f}. Learning rate = {}'.format(phase, epoch_loss,last_lr))
            
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss

                if best_model_cache == "memory":
                    best_model_wts = copy.deepcopy(net.state_dict())
                else:
                    torch.save(net.state_dict(), save_path+"_bestcahce.pkl")
    
    # Select best model wts if use memory to cahce models
    if best_model_cache == "memory":
        torch.save(best_model_wts, save_path)
        net.load_state_dict(best_model_wts)  
    else:
        net.load_state_dict((torch.load(save_path+"_bestcahce.pkl")))
        torch.save(net.state_dict(), save_path)

    return net, loss_train

def train_CVAE_model(net,data_loaders={},optimizer=None,n_epochs=100,scheduler=None,load=False,save_path="model.pkl",best_model_cache = "drive"):
    
    if(load!=False):
        if(os.path.exists(save_path)):
            net.load_state_dict(torch.load(save_path))           
            return net, 0
        else:
            logging.warning("Failed to load existing file, proceed to the trainning process.")
    
    dataset_sizes = {x: data_loaders[x].dataset.tensors[0].shape[0] for x in ['train', 'val']}
    loss_train = {}
    
    if best_model_cache == "memory":
        best_model_wts = copy.deepcopy(net.state_dict())
    else:
        torch.save(net.state_dict(), save_path+"_bestcahce.pkl")
    
    best_loss = np.inf

    for epoch in range(n_epochs):
        logging.info('Epoch {}/{}'.format(epoch, n_epochs - 1))
        logging.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #optimizer = scheduler(optimizer, epoch)
                net.train()  # Set model to training mode
            else:
                net.eval()  # Set model to evaluate mode

            running_loss = 0.0

            n_iters = len(data_loaders[phase])


            # Iterate over data.
            # for data in data_loaders[phase]:
            for batchidx, (x, c) in enumerate(data_loaders[phase]):

                x.requires_grad_(True)
                # encode and decode 
                output = net(x,c)
                # compute loss

                #losses = net.loss_function(*output, M_N=data_loaders[phase].batch_size/dataset_sizes[phase])      
                #loss = losses["loss"]

                recon_loss = nn.MSELoss(reduction="sum")

                loss = vae_loss(output[0],output[1],output[2],output[3],recon_loss,data_loaders[phase].batch_size/dataset_sizes[phase])

                # zero the parameter (weight) gradients
                optimizer.zero_grad()

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    # update the weights
                    optimizer.step()

                # print loss statistics
                running_loss += loss.item()
            
                
            epoch_loss = running_loss / dataset_sizes[phase]
            #epoch_loss = running_loss / n_iters

            
            if phase == 'train':
                scheduler.step(epoch_loss)
                
            last_lr = scheduler.optimizer.param_groups[0]['lr']
            loss_train[epoch,phase] = epoch_loss
            logging.info('{} Loss: {:.8f}. Learning rate = {}'.format(phase, epoch_loss,last_lr))
            
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss

                if best_model_cache == "memory":
                    best_model_wts = copy.deepcopy(net.state_dict())
                else:
                    torch.save(net.state_dict(), save_path+"_bestcahce.pkl")
    
    # Select best model wts if use memory to cahce models
    if best_model_cache == "memory":
        torch.save(best_model_wts, save_path)
        net.load_state_dict(best_model_wts)  
    else:
        net.load_state_dict((torch.load(save_path+"_bestcahce.pkl")))
        torch.save(net.state_dict(), save_path)

    return net, loss_train

def train_imputor_model(net,data_loaders,optimizer,loss_function,n_epochs,scheduler,load=False,save_path="model.pkl"):

    if(load!=False):
        if(os.path.exists(save_path)):
            net.load_state_dict(torch.load(save_path))           
            return net, 0
        else:
            logging.warning("Failed to load existing file, proceed to the trainning process.")
    
    dataset_sizes = {x: data_loaders[x].dataset.tensors[0].shape[0] for x in ['train', 'val']}
    loss_train = {}
    
    best_model_wts = copy.deepcopy(net.state_dict())
    best_loss = np.inf



    for epoch in range(n_epochs):
        logging.info('Epoch {}/{}'.format(epoch, n_epochs - 1))
        logging.info('-' * 10)


        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #optimizer = scheduler(optimizer, epoch)
                net.train()  # Set model to training mode
            else:
                net.eval()  # Set model to evaluate mode

            running_loss = 0.0

            # N iter s calculated
            n_iters = len(data_loaders[phase])

            # Iterate over data.
            # for data in data_loaders[phase]:
            for batchidx, (x, y) in enumerate(data_loaders[phase]):

                x.requires_grad_(True)
                # encode and decode 
                output = net(x)
                # compute loss
                loss = loss_function(output, y)      

                # zero the parameter (weight) gradients
                optimizer.zero_grad()

                # backward + optimize only if in training phase
                if phase == 'train':
                    try:
                        # 在这里执行可能引发错误的代码
                        loss.backward()
                        # update the weights
                        optimizer.step()
                    except Exception as e:
                        print(e)
                        import traceback
                        traceback.print_exc()

                # print loss statistics
                running_loss += loss.item()
            

            epoch_loss = running_loss / n_iters
            print(epoch_loss)
            if phase == 'train':
                scheduler.step(epoch_loss)
                
            last_lr = scheduler.optimizer.param_groups[0]['lr']
            loss_train[epoch,phase] = epoch_loss
            logging.info('{} Loss: {:.8f}. Learning rate = {}'.format(phase, epoch_loss,last_lr))
            
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(net.state_dict())
    
    # Select best model wts
        torch.save(best_model_wts, save_path)
        
    net.load_state_dict(best_model_wts)           
    
    return net, loss_train

def calculateKNNgraphDistanceMatrix(featureMatrix, distanceType='euclidean', k=10):
    distMat = distance.cdist(featureMatrix,featureMatrix, distanceType)
        #print(distMat)
    edgeList=[]
    
    for i in np.arange(distMat.shape[0]):
        res = distMat[:,i].argsort()[:k]
        for j in np.arange(k):
            edgeList.append((i,res[j],distMat[i,j]))
    
    return edgeList

def generateLouvainCluster(edgeList):
   
    """
    Louvain Clustering using igraph
    """
    Gtmp = nx.Graph()
    Gtmp.add_weighted_edges_from(edgeList)
    W = nx.adjacency_matrix(Gtmp)
    W = W.todense()
    graph = Graph.Weighted_Adjacency(
        W.tolist(), mode=ADJ_UNDIRECTED, attr="weight", loops=False)
    louvain_partition = graph.community_multilevel(
        weights=graph.es['weight'], return_levels=False)
    size = len(louvain_partition)
    hdict = {} #200 dic show cell's cluster number
    count = 0
    for i in range(size):
        tlist = louvain_partition[i]
        for j in range(len(tlist)):
            hdict[tlist[j]] = i
            count += 1

    listResult = [] #200len: cell index's cluster number
    for i in range(count):
        listResult.append(hdict[i])

    return listResult, size

def train_DTL_model(net,source_loader,target_loader,
                    optimizer,loss_function,n_epochs,scheduler,dist_loss,weight=0.25,GAMMA=1000,epoch_tail=0.90,
                    load=False,save_path="save/model.pkl",best_model_cache = "drive",top_models=5,k=10,device="cuda"):

    if(load!=False):
        if(os.path.exists(save_path)):
            try:
                net.load_state_dict(torch.load(save_path))           
                return net, 0,0,0
            except:
                logging.warning("Failed to load existing file, proceed to the trainning process.")

        else:
            logging.warning("Failed to load existing file, proceed to the trainning process.")
    
    # dataset_sizes = {x: source_loader[x].dataset.tensors[0].shape[0] for x in ['train', 'val']}
    loss_train = {}
    mmd_train = {}
    sc_train = {}
    best_model_wts = copy.deepcopy(net.state_dict())
    best_loss = np.inf


    g_tar_outputs = []
    g_src_outputs = []

    for epoch in range(n_epochs):
        logging.info('Epoch {}/{}'.format(epoch, n_epochs - 1))
        logging.info('-' * 10)


        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #optimizer = scheduler(optimizer, epoch)
                net.train()  # Set model to training mode
            else:
                net.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_mmd = 0.0
            running_sc =0.0
            
            batch_j = 0
            list_src, list_tar = list(enumerate(source_loader[phase])), list(enumerate(target_loader[phase]))
            n_iters = max(len(source_loader[phase]), len(target_loader[phase]))

            for batchidx, (x_src, y_src) in enumerate(source_loader[phase]):#x_src:bulk y_src:bulk
                _, (x_tar, y_tar) = list_tar[batch_j]
                
                x_tar.requires_grad_(True)
                x_src.requires_grad_(True)

                min_size = min(x_src.shape[0],x_tar.shape[0])

                if (x_src.shape[0]!=x_tar.shape[0]):
                    x_src = x_src[:min_size,]
                    y_src = y_src[:min_size,]
                    x_tar = x_tar[:min_size,]
                    y_tar = y_tar[:min_size,]

                #x.requires_grad_(True)
                # encode and decode 
                
                
                
                if(net.target_model._get_name()=="CVAEBase"):
                    y_pre, x_src_mmd, x_tar_mmd = net(x_src, x_tar,y_tar)
                else:
                    y_pre, x_src_mmd, x_tar_mmd = net(x_src, x_tar)
                # compute loss
                encoderrep = net.target_model.encoder(x_tar)
                #print(x_tar.shape)
                if encoderrep.shape[0]<k:
                    next
                else:    
                    edgeList = calculateKNNgraphDistanceMatrix(encoderrep.cpu().detach().numpy(), distanceType='euclidean', k=10)
                    listResult, size = generateLouvainCluster(edgeList)
                    # sc sim loss
                    loss_s = 0
                    for i in range(size):
                        #print(i)
                        s = cosine_similarity(x_tar[np.asarray(listResult) == i,:].cpu().detach().numpy())
                        s = 1-s
                        loss_s += np.sum(np.triu(s,1))/((s.shape[0]*s.shape[0])*2-s.shape[0])
                    #loss_s = torch.tensor(loss_s).cuda()
                    if(device=="cuda"):
                        loss_s = torch.tensor(loss_s).cuda()
                    else:
                        loss_s = torch.tensor(loss_s).cpu()
                    loss_s.requires_grad_(True)
                    loss_c = loss_function(y_pre, y_src)
                    loss_mmd = dist_loss(x_src_mmd, x_tar_mmd)
                    #print(loss_s,loss_c,loss_mmd)
    
                    # loss = loss_c + weight * loss_mmd +loss_s
                    loss = loss_c + weight * loss_mmd
    
    
                    # zero the parameter (weight) gradients
                    optimizer.zero_grad()
    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward(retain_graph=True)
                        # update the weights
                        optimizer.step()
    
                    # print loss statistics
                    running_loss += loss.item()
                    running_mmd += loss_mmd.item()
                    running_sc += loss_s.item()
                    # Iterate over batch
                    batch_j += 1
                    if batch_j >= len(list_tar):
                        batch_j = 0

            # Average epoch loss
            epoch_loss = running_loss / n_iters
            epoch_mmd = running_mmd/n_iters
            epoch_sc = running_sc/n_iters
            # Step schedular
            if phase == 'train':
                scheduler.step(epoch_loss)
            
            # Savle loss
            last_lr = scheduler.optimizer.param_groups[0]['lr']
            loss_train[epoch,phase] = epoch_loss
            mmd_train[epoch,phase] = epoch_mmd
            sc_train[epoch,phase] = epoch_sc
            
            logging.info('{} Loss: {:.8f}. Learning rate = {}'.format(phase, epoch_loss,last_lr))
            print('{} Loss: {:.8f}. Learning rate = {}'.format(phase, epoch_loss,last_lr))
            if (phase == 'val') and (epoch_loss < best_loss) and (epoch >(n_epochs*(1-epoch_tail))) :
                best_loss = epoch_loss
                #best_model_wts = copy.deepcopy(net.state_dict())
                # Save model if acheive better validation score
                if best_model_cache == "memory":
                    best_model_wts = copy.deepcopy(net.state_dict())
                else:
                    torch.save(net.state_dict(), save_path[:-4]+"_bestcahce.pkl")
 
    #     # Select best model wts
    #     torch.save(best_model_wts, save_path)
        
    # net.load_state_dict(best_model_wts)           
        # Select best model wts if use memory to cahce models
    if best_model_cache == "memory":
        torch.save(best_model_wts, save_path)
        net.load_state_dict(best_model_wts)  
    else:
        net.load_state_dict((torch.load(save_path[:-4] +"_bestcahce.pkl")))
        torch.save(net.state_dict(), save_path)

    return net, loss_train,mmd_train,sc_train    