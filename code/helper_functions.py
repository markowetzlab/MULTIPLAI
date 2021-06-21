from sklearn.neighbors import kneighbors_graph
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
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from sklearn.metrics import roc_auc_score,balanced_accuracy_score

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc

def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)



def accontestset_v2(model,testset,num_layers,pooling):
  model.eval()
  loss_func = nn.BCELoss()
  test_X,test_Y = map(list, zip(*testset))
  test_bg = dgl.batch(test_X)
  pred_Y = model(test_bg,num_layers,pooling)
  label = torch.tensor(test_Y)
  label=label.unsqueeze(1)
  acc = balanced_accuracy_score(label.detach().numpy(),np.round(pred_Y[0].detach().numpy()))
  return acc,loss_func(pred_Y[0], label.float()),pred_Y[1].detach().numpy(),testset,pred_Y[2].detach().numpy()

def prepare_graphs_label_v2(DB1,DB1_clin,nfeats_scaled,celltype,mylabel,input_path,dist):

    celltypes = celltype
    stains_final = []
    for j in range(len(celltypes)):
        stains_final.append(celltypes[j]+'_scaled')

    allgraphs_train=[]
    list_IDs = DB1['metabricId']
    unique = list_IDs.unique()
    IDS=[]
    overallncells=[]
    for m in range(len(unique)):
    # For each patient
        k = unique[m]
        try:
            print(k)
            kNN_adj = pd.read_csv(input_path + 'single_cell_graphs_distances/' + str(dist)+'_'+  k  + '_AdjM.csv',header = None)
            nfeats = nfeats_scaled[nfeats_scaled.metabricId == k]
            nfeats_arr = nfeats[stains_final].values
            g = dgl.DGLGraph(kNN_adj)
            g.ndata['h_n'] = nfeats_arr
            A = torch.tensor(sparse.csr_matrix(kNN_adj.values).data).reshape(-1, 1)
            g.edata['h_e'] = A.float()
            row=DB1_clin[DB1_clin['METABRIC.ID']==k]
            if not row.empty:
                label = row[mylabel].values[0]
                allgraphs_train.append((g,label))
        except:
            print(input_path+single_cell_graphs_distances)
            print('not in')
    return allgraphs_train


def prepare_graphs_label_forMB(DB1_clin,nfeats_scaled,celltype,mylabel,input_path):

    celltypes = celltype
    stains_final = []
    for j in range(len(celltypes)):
        stains_final.append(celltypes[j]+'_scaled')

    allgraphs_train=[]
    list_IDs = DB1_clin['metabricId']
    unique = list_IDs.unique()
    IDS=[]
    overallncells=[]
    for m in range(len(unique)):
    # For each patient
        k = unique[m]
       
        print(k)
        kNN_adj = pd.read_csv(input_path + 'single_cell_graphs_distances/' + str(40)+'_'+  k  + '_AdjM.csv',header = None)
        nfeats = nfeats_scaled[nfeats_scaled.metabricId == k]
        nfeats_arr = nfeats[stains_final].values
        g = dgl.DGLGraph(kNN_adj)
        g.ndata['h_n'] = nfeats_arr
        A = torch.tensor(sparse.csr_matrix(kNN_adj.values).data).reshape(-1, 1)
        g.edata['h_e'] = A.float()
        row=DB1_clin[DB1_clin['metabricId']==k]
        if not row.empty:
            label = row[mylabel].values[0]
            allgraphs_train.append((g,label))
        
    return allgraphs_train

def prepare_graphs_val_Zuri(DB1_clin,nfeats_scaled,celltype,mylabel,input_path):
    allgraphs_train=[]
    celltypes = celltype
    stains_final = []
    for j in range(len(celltypes)):
        stains_final.append(celltypes[j]+'_scaled')

    list_IDs = DB1_clin['core']
    unique = list_IDs.unique()
    IDS=[]
    overallncells=[]
    for m in range(len(unique)):
        try:
    # For each patient
            k = unique[m]
            kNN_adj = pd.read_csv(input_path + 'single_cell_graphs_distances/' + str(40)+'_'+  k  + '_AdjM.csv',header = None)
            nfeats = nfeats_scaled[nfeats_scaled.core == k]
            nfeats_arr = nfeats[stains_final].values
            g = dgl.DGLGraph(kNN_adj)
            g.ndata['h_n'] = nfeats_arr
            A = torch.tensor(sparse.csr_matrix(kNN_adj).data).reshape(-1, 1)
            g.edata['h_e'] = A.float()
            row=DB1_clin[DB1_clin['core']==k]
            if not row.empty:
                label = row[mylabel].values[0]
                allgraphs_train.append((g,label))
                print(k)
        except:
            print('not in')
    return allgraphs_train



