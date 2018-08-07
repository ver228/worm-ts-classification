#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 13:59:17 2018

@author: avelinojaver
"""
import sys
from pathlib import Path 
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
#%%
import pandas as pd
import tqdm
from sklearn.metrics import f1_score

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
#%%
class SimpleNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
    
    def forward(self, x):
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
#%%
if __name__ == '__main__':
    save_dir = '/Users/avelinojaver/workspace/WormData/experiments/classify_strains/results/ow_comparison'
    save_dir = Path(save_dir)
    df_info = pd.read_csv(save_dir / 'SWDB_Y_info.csv')
    df_feats_z = pd.read_csv(save_dir / 'SWDB_X_features.csv')
    #df_info = pd.read_csv(save_dir / 'CeNDR_Y_info.csv')
    #df_feats_z = pd.read_csv(save_dir / 'CeNDR_X_features.csv')
    
    
    fold_n_test = 0
    
    is_test = df_info['fold']==0
    
    
    y_test = df_info.loc[is_test, 'strain_id'].values
    x_test = df_feats_z[is_test].values
    
    y_train = df_info.loc[~is_test, 'strain_id'].values
    x_train = df_feats_z[~is_test].values
    
    
    n_classes = int(df_info['strain_id'].max()) +1 
    n_features = x_test.shape[1]
    
    #%%
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).long()
    
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long() 
    #%%
    
    batch_size = 64  
    lr = 0.001
    momentum = 0.9
    n_epochs = 1000
    
    model = SimpleNet(n_features, n_classes)
    #model = model.cuda(cuda_id)
        
    optimizer = optim.SGD(model.parameters(), lr = lr, momentum = momentum)
    
    
    criterion = F.nll_loss
    
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size = batch_size, shuffle=True)
    
    
    
    pbar = tqdm.trange(n_epochs)
    train_losses = []
    for i in pbar:
        #Train model
        model.train()
        train_loss = 0.
        for k, (xx, yy) in enumerate(loader):
            # Reset gradient
            optimizer.zero_grad()
        
            # Forward
            output = model(xx)
            loss = criterion(output, yy)
        
            # Backward
            loss.backward()
        
            # Update parameters
            optimizer.step()
        
            train_loss += loss.item()
        train_loss /= len(loader)
        
        d_str = "train loss = %f" % (train_loss)
        pbar.set_description(d_str)
    
        train_losses.append(train_loss)
    #%%
    model.eval()
    pred = model(x_test)
    
    topk = (1,5)
    maxk = max(topk)
    _, top_pred = pred.topk(maxk)
    top_pred = top_pred.t()
    correct = top_pred.eq(y_test.view(1, -1).expand_as(top_pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].sum(0)
        correct_k = correct_k.detach().cpu().numpy().tolist()
        res.append(correct_k)
        
    ytrue = y_test.detach().cpu().numpy()
    ypred = top_pred[0].detach().cpu().numpy()
    f1 = f1_score(ytrue, ypred, average='macro')
    acc_top1, acc_top5 = map(np.mean, res)    
    
    
    print(acc_top1, acc_top5, f1)