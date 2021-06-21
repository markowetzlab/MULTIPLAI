#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 16:15:29 2020

@author: martin09
"""
import os
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import dgl
from scipy.sparse import coo_matrix
from scipy import sparse
import dgl
import torch
from dgl import DGLGraph
from scipy.sparse import rand
import numpy as np 
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import NNConv, Set2Set
import dgl
import torch
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from numpy import savetxt
from sklearn.metrics import roc_curve,roc_auc_score,balanced_accuracy_score,precision_score,recall_score
from helper_functions import binary_acc,prepare_graphs_label_v2,collate,accontestset_v2,prepare_graphs_val_Zuri,prepare_graphs_label_forMB
from allmodels import WeavePredictor,MPNNPredictor,GINPredictor,AttentiveFPPredictor,Classifier_GAT_GAP,Classifier_GCN_GAP


def runmymodel2():
    #LOAD AND PREPARE DATA

    patience_cutoff = 40
    input_path = '/data/inputs/inputs/'
    

    DB1 = pd.read_csv(input_path + 'train.csv')

    DB1_clin = pd.read_csv(input_path + 'train_clin.csv')
    DB2 = pd.read_csv(input_path + 'test.csv')
    DB2_clin = pd.read_csv(input_path + 'test_clin.csv')
    celltype = ['CK19','CK8_18','CD68','CK14','SMA','Vimentin', 'CD3','p53','CD44','CD45',
    'CD20','Ki67','EGFR','pS6','vWF_CD31','CK7','panCK','CK5','Fibronectin']

  

    labeltouse = 'ER.Status'
    db_val_clin = pd.read_csv(input_path + 'external_val_clin.csv')
    nfeats_val = pd.read_csv(input_path + 'single_cell_graphs_distances/' + 'nfeats_test_B_scaled_v3.csv')
   
    db_val_clinZ = pd.read_csv(input_path + 'external_val_Zurich_clin.csv')
    nfeats_valZ = pd.read_csv(input_path + 'single_cell_graphs_distances/' + 'nfeats_test_Z_scaled_v3.csv')

    db_val_clinMB = pd.read_csv(input_path + 'val_metabric_clin.csv')
    nfeats_valMB = pd.read_csv(input_path + 'single_cell_graphs_distances/' + 'nfeats_test_internal_scaled_v3.csv')

    
  
    valset = prepare_graphs_val_Zuri(db_val_clin,nfeats_val,celltype,labeltouse,input_path)
    print('test set prepared')
    valsetZ = prepare_graphs_val_Zuri(db_val_clinZ,nfeats_valZ,celltype,labeltouse,input_path)
    print('test set prepared')
    valsetMB = prepare_graphs_label_forMB(db_val_clinMB,nfeats_valMB,celltype,labeltouse,input_path)
   
 
    model = torch.load('/data/outputs/ERonly/sc_dist/GCN_lr0.001_ER.Status_nlayers_2_pooling_att_distance40_graphfeatsize_150_dim1_50_dropout_0.5_bs_30_weightdecay_0.01/20patience_model.pt')
    num_layers = '2'
    pooling = 'att'
    model.eval()


    print('val acc MB')
    test_X,test_Y = map(list, zip(*valsetMB))
    test_bg = dgl.batch(test_X)
    pred_Y = model(test_bg,num_layers,pooling)
    label = torch.tensor(test_Y)
    label=label.unsqueeze(1)
    acc = balanced_accuracy_score(label.detach().numpy(),np.round(pred_Y[0].detach().numpy()))
    print(acc)
    print(recall_score(label.detach().numpy(),np.round(pred_Y[0].detach().numpy())))
    print(precision_score(label.detach().numpy(),np.round(pred_Y[0].detach().numpy())))
    cm_val = confusion_matrix(label.detach().numpy(),np.round(pred_Y[0].detach().numpy()))
    print('confusion matrix')
    print(cm_val)

    print('val acc B')
    test_X,test_Y = map(list, zip(*valset))
    test_bg = dgl.batch(test_X)
    pred_Y = model(test_bg,num_layers,pooling)
    label = torch.tensor(test_Y)
    label=label.unsqueeze(1)
    acc = balanced_accuracy_score(label.detach().numpy(),np.round(pred_Y[0].detach().numpy()))
    print(acc)
    print(recall_score(label.detach().numpy(),np.round(pred_Y[0].detach().numpy())))
    print(precision_score(label.detach().numpy(),np.round(pred_Y[0].detach().numpy())))
    
   
    print('val acc Z')
    test_X,test_Y = map(list, zip(*valsetZ))
    test_bg = dgl.batch(test_X)
    pred_Y = model(test_bg,num_layers,pooling)
    label = torch.tensor(test_Y)
    label=label.unsqueeze(1)
    acc = balanced_accuracy_score(label.detach().numpy(),np.round(pred_Y[0].detach().numpy()))
    print(acc)
    print(recall_score(label.detach().numpy(),np.round(pred_Y[0].detach().numpy())))
    print(precision_score(label.detach().numpy(),np.round(pred_Y[0].detach().numpy())))
    
   

def main():
    runmymodel2()

if __name__ == "__main__":
    import sys
    main()


