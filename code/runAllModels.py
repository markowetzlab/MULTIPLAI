#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 16:15:29 2020

@author: martin09
"""
import os
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
from sklearn.metrics import roc_auc_score,balanced_accuracy_score
from helper_functions import binary_acc,prepare_graphs_label_v2,collate,accontestset_v2
from allmodels import WeavePredictor,MPNNPredictor,GINPredictor,AttentiveFPPredictor,Classifier_GAT_GAP,Classifier_GCN_GAP,Classifier_gen


def runmymodel2(mylr,labeltouse,node_feat_size,pooling, num_layers,
    hid_dim1,graph_feat_size,n_tasks,dropout,bsize,weight_decay,dist):
    #LOAD AND PREPARE DATA
    patience_cutoff = 40
    input_path = '/data/inputs/inputs/'
    output_path = '/data/outputs/ERonly/sc_dist/GCN_lr'+str(mylr)+'_'+labeltouse+'_nlayers_'+str(num_layers)+'_pooling_'+pooling+'_distance'+str(dist)+'_graphfeatsize_'+str(graph_feat_size)+'_dim1_'+str(hid_dim1)+'_dropout_'+str(dropout)+'_bs_'+str(bsize)+"_weightdecay_"+str(weight_decay)+'/'
    try:
        os.mkdir(output_path)
    except:
        print('Directory already exists!')

    DB1 = pd.read_csv(input_path + 'train.csv')

    DB1_clin = pd.read_csv(input_path + 'train_clin.csv')
    DB2 = pd.read_csv(input_path + 'test.csv')
    DB2_clin = pd.read_csv(input_path + 'test_clin.csv')
    celltype = ['CK19','CK8_18','CD68','CK14','SMA','Vimentin', 'CD3','p53','CD44','CD45',
    'CD20','Ki67','EGFR','pS6','vWF_CD31','CK7','panCK','CK5','Fibronectin']

    nfeats_train = pd.read_csv(input_path + 'single_cell_graphs_distances/' + 'nfeats_train_scaled_v3.csv')
    nfeats_test = pd.read_csv(input_path + 'single_cell_graphs_distances/' + 'nfeats_dev_scaled_v3.csv')
   
    trainset = prepare_graphs_label_v2(DB1,DB1_clin, nfeats_train,celltype,labeltouse,input_path,dist)
    print('training set prepared')
    testset = prepare_graphs_label_v2(DB2,DB2_clin,nfeats_test,celltype,labeltouse,input_path,dist)
    print('test set prepared')
    # Use PyTorch's DataLoader and the collate function

    data_loader = DataLoader(trainset, batch_size=int(bsize), shuffle=True,
                            collate_fn=collate)

    model = Classifier_gen(int(node_feat_size), int(graph_feat_size),int(hid_dim1), int(n_tasks),float(dropout),int(num_layers),str(pooling))
    print('model initialized')
    # SET LOSS FUNCTION AND OPTIMIZER
    loss_func = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=float(mylr),weight_decay=float(weight_decay))
    model.train()

    #TRAIN
    epoch_losses = []
    epoch_accs =[]
    epoch_test_accs = []
    epoch_test_losses = []
    patience = 0
    for epoch in range(200):
        epoch_loss = 0
        epoch_acc = 0
        epoch_acc_test = 0
        epoch_loss_test=0
        for iter, (bg, label) in enumerate(data_loader):
            prediction = model(bg,num_layers,pooling)
            label=label.unsqueeze(1)
            loss = loss_func(prediction[0], label.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
            acc = balanced_accuracy_score(label.float().detach().numpy(),np.round(prediction[0].detach().numpy()))
            epoch_acc +=acc.item()
            acc_test,loss_test,att_test,test_bg,gfeats= accontestset_v2(model,testset,num_layers,pooling)
            epoch_acc_test +=acc_test.item()
            epoch_loss_test +=loss_test.item() 

        epoch_loss /= (iter + 1)
        epoch_acc /= (iter+1)
        epoch_acc_test /= (iter+1)
        epoch_loss_test /= (iter + 1)

        if epoch == 0:        
            print('Epoch {} | loss {:.4f} | acc {:.4f} | loss test {:.4f}| acc test {:.4f}'.format(epoch, epoch_loss,epoch_acc,epoch_loss_test,epoch_acc_test))
            epoch_losses.append(epoch_loss)
            epoch_accs.append(epoch_acc)
            epoch_test_accs.append(epoch_acc_test)
            epoch_test_losses.append(epoch_loss_test)
        elif epoch_loss_test<np.min(epoch_test_losses):
            patience = 0
            print('Epoch {} | loss {:.4f} | acc {:.4f} | loss test {:.4f}| acc test {:.4f}'.format(epoch, epoch_loss,epoch_acc,epoch_loss_test,epoch_acc_test))
            epoch_losses.append(epoch_loss)
            epoch_accs.append(epoch_acc)
            epoch_test_accs.append(epoch_acc_test)
            epoch_test_losses.append(epoch_loss_test)
            torch.save(model, output_path+'beforepatience_model.pt')
        
        elif epoch_loss_test>np.min(epoch_test_losses):
            patience = patience + 1          
            if patience <= patience_cutoff:
                print('Epoch {} | loss {:.4f} | acc {:.4f} | loss test {:.4f}| acc test {:.4f}'.format(epoch, epoch_loss,epoch_acc,epoch_loss_test,epoch_acc_test))
                print(patience)
                epoch_losses.append(epoch_loss)
                epoch_accs.append(epoch_acc)
                epoch_test_accs.append(epoch_acc_test)
                epoch_test_losses.append(epoch_loss_test)
                #torch.save(model, output_path+'model.pt')
                if patience == patience_cutoff:
                    torch.save(model, output_path+'model.pt')
                    break
            if patience ==patience_cutoff/2:
                print('Epoch {} | loss {:.4f} | acc {:.4f} | loss test {:.4f}| acc test {:.4f}'.format(epoch, epoch_loss,epoch_acc,epoch_loss_test,epoch_acc_test))
                epoch_losses.append(epoch_loss)
                epoch_accs.append(epoch_acc)
                epoch_test_accs.append(epoch_acc_test)
                epoch_test_losses.append(epoch_loss_test)
                torch.save(model, output_path+'20patience_model.pt')
        
    import matplotlib.pyplot as plt 
    plt.title('Binary cross entropy averaged over minibatches')
    plt.plot(epoch_losses, label = "train")
    plt.plot(epoch_test_losses, label = "test")
    plt.legend()
    plt.show()
    plt.savefig(output_path+'loss_plot.png')
    plt.close('all')

    plt.title('Class balanced accuracy averaged over minibatches')
    plt.xlabel("epochs")
    plt.ylabel("Class balanced accuracy")
    plt.plot(epoch_accs, label = "train")
    plt.plot(epoch_test_accs, label = "test")
    plt.legend()
    plt.show()
    plt.savefig(output_path+'acc_plot.png')


    


def main(mylr,labeltouse,node_feat_size,pooling, num_layers,
    hid_dim1,graph_feat_size,n_tasks,dropout,bsize,weight_decay,dist):
    runmymodel2(mylr,labeltouse,node_feat_size,pooling, num_layers,
    hid_dim1,graph_feat_size,n_tasks,dropout,bsize,weight_decay,dist)

if __name__ == "__main__":
    import sys
    main(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],sys.argv[7],sys.argv[8],sys.argv[9],sys.argv[10],sys.argv[11],sys.argv[12])


