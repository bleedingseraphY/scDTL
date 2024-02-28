# scDTL documentation
scDTL: single-cell RNA-seq imputation based on deep transfer learning using bulk RNA-seq data

## Installation 

### Retrieve code from GitHub
The software is a stand-alone python script package. The home directory of scDTL can be cloned from the GitHub repository:

```
# Clone from Github
git clone https://github.com/bleedingseraphY/scDTL.git
# Go into the directory
cd scDTL
```

## Data Preparation
### Data download
After setting up the home directory, you need to download other data required for the run. Please create and download the zip format dataset from the website link inside:

[Click here to download gene data](https://www.ncbi.nlm.nih.gov/geo/) 

All resources in the home directory of scDTL should look as follows:

```
scDTL
|   ...
│   README.md
│   bulkmodel.py  
│   scmodel.py
│   cbammodel.py
|   ...
└───data
│   │   CCLE.csv
│   │   CCLE_GSE112274_4000_sort.csv
│   │   CCLE_GSE117872_4000_sort.csv
│   │   CCLE_GSE134836_4000_sort.csv
│   │   CCLE_GSE134838_4000_sort.csv
│   │   CCLE_GSE134839_4000_sort.csv
│   │   CCLE_GSExxxxxx_4000_sort.csv
│   └───GSE112274
│   │      GSE134838_4000.h5ad
│   └───GSE117872
│   │      GSE117872_4000.h5ad
│   └───GSE134836
│   │      GSE134836_4000.h5ad
│   └───GSE134838
│   │      GSE134838_4000.h5ad
│   └───GSE134839
│   │      GSE134839_4000.h5ad
│   └───GSE134841
│   │      GSE134841_4000.h5ad
│   └───GSE140440
│   │      GSE140440_4000.h5ad
│   └───GSE149214
│   │      GSE149214_4000.h5ad
│   |   ...
└───save
|   └───logs
|   └───result
|   └───bulk_encoder
|   └───bulk_pre
|   └───sc_encoder
|   └───sc_pre
|   └───cbam_pre
│   │    ...   
```

### Directory contents
Folders in our package will store the corresponding contents:

- root: python scripts to run the program and README.md
- data: datasets required for the learning
- save/logs: log and error files that record running status. 
- save/result: model generate result. 
- save/xx_pre: models trained through the run. 
- DTL: transfer learning loss.
- scanpypip: python scripts of preprocessing.

## Demo
### Pretrained checkpoints
For the scRNA-Seq data impute task, we provide pre-trained checkpoints for the models stored in save.

An example can be:

Usage:
For resuming training, you can use the --checkpoint option of bulkmodel.py, scmodel.py and cbammodel.py.
For example, run scmodel.py and cbammodel.py with checkpoints to get the single-cell level impute results:

```
source scDTL/bin/activate
python scmodel.py --sc_data_name "GSE134838" --dimreduce "DAE" --bulk_h_dims "4096,2048" --imputor_h_dims "2048,4096" --dropout 0.0 --lr 0.01 --checkpoint "data_GSE134838_4000_bottle_2048_edim_4096,2048_idim_2048,4096_model_DAE_DaNN.pkl"
python cbammodel.py --sc_data "GSE134838" --dimreduce "DAE" --con_kernel "16,32,64,128" --dropout 0.0 --lr 0.01 --checkpoint "data_GSE134838_con_kernel_16,32,64,128_dropRate_0.0_model_DAE.pkl"
```

We can also train bulkmode.py, scmodel.py and cbammodel.py from scratch with user-defined parameters by setting --checkpoint "False":
```
source scDTL/bin/activate
python bulkmodel.py --sc_data_name "GSE134838" --dimreduce "DAE" --encoder_h_dims "4096,2048" --imputor_h_dims "2048,4096" --dropout 0.0 --lr 0.01 --checkpoint "False"
python scmodel.py --sc_data_name "GSE134838" --dimreduce "DAE" --bulk_h_dims "4096,2048" --imputor_h_dims "2048,4096" --dropout 0.0 --lr 0.01 --checkpoint "False"
python cbammodel.py --sc_data "GSE134838" --dimreduce "DAE" --con_kernel "16,32,64,128" --dropout 0.0 --lr 0.01 --checkpoint "False"
```

### Expected output
The expected output format of scDTL is the cbam_impute_all_leiden_GSE134838.csv. The file will be stored in the directory "scDTL/save/result/GSE134838".  

The expected output format of a successful run show includes:

```
scDTL
|   ...
└───save
│   └───result
│   |    │
│   |    └───GSE134838
│   │    │      cbam_impute_zero_leiden_GSE134838.csv   
│   │    ...   
|   └───models
│   │    save/bulk_encoder
│   │    save/bulk_pre
│   │    save/sc_encoder
│   │    save/sc_pre
│   │    save/cbam_pre
│   │    ...
```

For more detailed parameter settings of the two scripts, please refer to the documentation section.

