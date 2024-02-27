import os

import scanpy
import scanpy as sc
import numpy as np
import pandas as pd
from matplotlib import pyplot as plot


def read_sc_file(file_path,header=0,index_col=0,sep=None):
    '''
    This is a fucntion to load data having multiple formats.
    Params:
    -------
    file_path: str,
        The path of the input file
    header: int, (default: `0`)
        Only used if loading txt or csv files. Set the row number of header.
    index_col: int, (default: `0`)
        Only used if loading txt or csv files. Set the row name number of the expression file.
    sep: str, (default: `"\t"`)
        Only used if loading txt or csv files. Set the seperator of the input file.
    Return:
    -------
    gene_expression: AnnData,
        The load data. 
    '''
    filename = file_path
    separators = ["\t","\n"," ",","] 

    # Read first line to select a right seperator
    def try_seperators(filename, header, index_col, seps):
        for s in seps:
            # first_row = pd.read_csv(filename, header=header, index_col=index_col, sep = s, nrows=1)
            first_row = pd.read_csv(filename, header=header, index_col=index_col, sep=s)
            if(first_row.shape[1]>0):
                return s

        print("cannot find correct seperators, return tab as seperator")
        return '\t'

    # deal with csv file 
    if ((filename.find(".csv")>=0) or (filename.find(".txt")>=0)):

        # If a seperator is defined
        if(sep!=None):
            counts_drop = pd.read_csv(filename, header=header, index_col=index_col, sep = sep)

        else:
            seperator = try_seperators(filename, header, index_col, separators)
            counts_drop = pd.read_csv(filename, header=header, index_col=index_col, sep = seperator)

        gene_expression = sc.AnnData(counts_drop)

    # deal with 10x h5 file
    elif filename.find(".h5")>=0:
        if filename.find(".h5ad")<0:
            gene_expression = sc.read_10x_h5(filename, genome=None, gex_only=True)
        else:
            gene_expression = sc.read_h5ad(filename)
    
    # Deal with 10x mtx files
    else:
        gene_expression = sc.read_10x_mtx(filename,  # the directory with the `.mtx` file 
        var_names='gene_symbols',                # use gene symbols for the variable names (variables-axis index)
        cache=True)            

    return gene_expression

def cal_ncount_ngenes(adata,sparse=False,remove_keys=[]):
    
    mito_genes = (adata.var_names.str.lower().str.rfind('mt-'))!=-1
    rps_genes = (adata.var_names.str.lower().str.rfind('rps'))!=-1
    rpl_genes = (adata.var_names.str.lower().str.rfind('rpl'))!=-1

    adata.var['mt-'] = mito_genes
    adata.var['rps'] = rps_genes
    adata.var['rpl'] = rpl_genes

    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt-'], percent_top=None, log1p=False, inplace=True)
    sc.pp.calculate_qc_metrics(adata, qc_vars=['rps'], percent_top=None, log1p=False, inplace=True)
    sc.pp.calculate_qc_metrics(adata, qc_vars=['rpl'], percent_top=None, log1p=False, inplace=True)

    if len(remove_keys)>0:
        mask = np.ones(adata.shape[1])
        if 'mt-' in remove_keys:
            mask = np.logical_and(mask,mito_genes == False)
        if 'rps' in remove_keys:
            mask = np.logical_and(mask,rps_genes == False)
        if 'rpl' in remove_keys:
            mask = np.logical_and(mask,rpl_genes == False)

        adata = adata[:, mask]

    return adata

def receipe_my(adata,l_n_genes = 500, r_n_genes= 5000, filter_mincells=3,filter_mingenes=200, percent_mito = 5, normalize = False,log = False,sparse = False,plotinfo= False,
                remove_genes=[]):

    sc.pp.filter_cells(adata, min_genes=filter_mingenes)
    sc.pp.filter_genes(adata, min_cells=filter_mincells)
    
    adata = cal_ncount_ngenes(adata,remove_keys=remove_genes)

    adata = adata[
        np.logical_and(
        (adata.obs['n_genes_by_counts'] > l_n_genes), 
        (adata.obs['n_genes_by_counts'] < r_n_genes)),:]
    adata = adata[adata.obs['pct_counts_mt-'] < percent_mito, :]

    if(plotinfo!=False):
        sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt-'],
             jitter=0.4, multi_panel=True, save=True)

    print(adata.shape)
    
    if normalize == True:
        sc.pp.normalize_total(adata)
    adata.raw = adata

    if log == True:
        sc.pp.log1p(adata)

    return adata

def select_common_gene(sc_data_name,top_counts):

    sc_data_name = sc_data_name
    sc_data_path = '../data/' + sc_data_name + '/' + sc_data_name+ '.cluster.h5ad'

    # Read bulk data
    data_bulk=pd.read_csv('../data/CCLE.csv', index_col=0)

    # Load sc_data and preprocessing
    adata_single = scanpy.read_h5ad(sc_data_path)
    #crs_matrix
    if(sc_data_name == 'GSE134836'):
        adata_single.X = adata_single.X.toarray()
    # Retrieve data with the same column headers from two DataFrames.
    common_columns = data_bulk.columns.intersection(adata_single.var_names)
    # Select data with the same column headers.
    selected_data_bulk = data_bulk[common_columns]
    selected_data_single = adata_single[:, adata_single.var.index.isin(common_columns)]

    selected_data_single.raw = selected_data_single
    num_genes_to_keep = top_counts
    #Calculate highly variable genes
    sc.pp.highly_variable_genes(selected_data_single, n_top_genes=num_genes_to_keep)
    #Get the names of the highly variable genes
    highly_variable_genes = selected_data_single.var_names[selected_data_single.var['highly_variable']]
    #Create a new AnnData object with only the highly variable genes
    selected_data_single_top_genes = selected_data_single[:, highly_variable_genes].copy()
    selected_data_bulk_top_genes = selected_data_bulk.loc[:, selected_data_single.var['highly_variable']]
    #write to h5ad
    selected_data_single_top_genes.write_h5ad(sc_data_path[:-13] +'_' + str(top_counts) +'.h5ad')
    # # save
    csv_file_path = '../data/CCLE_' + sc_data_name + '_' + str(top_counts) + '.csv'
    selected_data_bulk_top_genes.to_csv(csv_file_path)
    return True

def sort_bulk(sc_data_name,top_counts):

    sc_data_path = '../data/' + sc_data_name + '/' + sc_data_name+ '_4000.h5ad'
    # Read bulk data
    data_bulk=pd.read_csv('../data/CCLE_' + sc_data_name + '_4000.csv', index_col=0)
    # Load sc_data and preprocessing
    adata_single = scanpy.read_h5ad(sc_data_path)
    selected_data_bulk = data_bulk[adata_single.var_names]

    csv_file_path = '../data/CCLE_' + sc_data_name + '_' + str(top_counts) + '_sort.csv'
    selected_data_bulk.to_csv(csv_file_path)
    return True
#analyze the occurrence of zero values in the gene expression matrix.
def statistics_zero(sc_data_name):
    sc_data_name = sc_data_name
    sc_data_path = '../data/' + sc_data_name + '/' + sc_data_name + '.h5ad'

    adata_single = scanpy.read_h5ad(sc_data_path)
    gene_expression_matrix = adata_single.X

    zero_counts_per_gene = np.sum(gene_expression_matrix == 0, axis=0)/24427

    average_value = np.mean(zero_counts_per_gene)
    print("Average Value:", average_value)
    print(sc_data_name)
    print(zero_counts_per_gene)

    return True
def statistics_gene_cell(sc_data_name):
    sc_data_path = '../data/' + sc_data_name + '/' + sc_data_name + '.cluster.h5ad'
    adata_single = scanpy.read_h5ad(sc_data_path)
    print(sc_data_name)
    print(adata_single.X.shape)
    return True

def impute_dropout(sc_data_name, seed=2024, drop_rate=0.1):
    """
    sc_data_name: original single cell set
    """
    sc_data_path = '../data/' + sc_data_name + '/' + sc_data_name + '_4000.h5ad'
    adata_single = scanpy.read_h5ad(sc_data_path)

    X = adata_single.X

    X_zero = np.copy(X)
    i, j = np.nonzero(X_zero)
    if seed is not None:
        np.random.seed(seed)

    ix = np.random.choice(range(len(i)), int(
        np.floor(drop_rate * len(i))), replace=False)
    X_zero[i[ix], j[ix]] = 0.0

    adata_single.X = X_zero

    adata_single.write_h5ad(sc_data_path[:-5] + '_drop_' + str(drop_rate) + '.h5ad')

    return True
def h5ad_to_csv(sc_data_name,drop_rate):
    if(drop_rate != 0.0):
        adata = scanpy.read_h5ad('../data/' + sc_data_name + '/' + sc_data_name + '_4000_drop_' + str(drop_rate) + '.h5ad')
        adata.to_df().to_csv('../data/' + sc_data_name + '/' + sc_data_name + '_4000_drop_' + str(drop_rate) + '.csv')
    else:
        adata = scanpy.read_h5ad('../data/' + sc_data_name + '/' + sc_data_name + '_4000.h5ad')
        adata.to_df().to_csv('../data/' + sc_data_name + '/' + sc_data_name + '_4000.csv')
########################################################Main######################################
def get_folders_in_directory(directory_path):
    try:
        entries = os.scandir(directory_path)

        folders = [entry.name for entry in entries if entry.is_dir()]

        return folders

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def main_code():
    #, 'GSE134836'
    sc_data_names = ['GSE112274', 'GSE117872', 'GSE134836', 'GSE134838', 'GSE134839', 'GSE134841', 'GSE140440',
                     'GSE149214']

    for folder_name in sc_data_names:
        # select_common_gene(folder_name,4000)
        # impute_dropout(sc_data_name = folder_name,drop_rate=0.4)
        # sort_bulk(folder_name,4000)
        # h5ad_to_csv(folder_name)
        # statistics_zero(folder_name)
        statistics_gene_cell(folder_name)
    # select_common_gene('GSE134836',4000)
    # for folder_name in sc_data_names:
    #         statistics_zero(folder_name)

    # h5ad_to_csv("GSE134836",0.0)
    # h5ad_to_csv("GSE134836",0.1)
    # h5ad_to_csv("GSE134836",0.2)
    # h5ad_to_csv("GSE134836",0.4)
    # statistics_zero('GSE134836')
# main_code()