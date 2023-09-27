import os
import torch

from torch.nn import  ModuleList
from torch_geometric.nn import EdgeConv, GCNConv, GATConv, GraphConv, SAGEConv, ChebConv, LINKX

import pytorch_lightning as pl
from argparse import Namespace
import torchmetrics
import sklearn

import sys
sys.path.append("../source/")
from utils import graph_metrics
from DGM.layers import *
if (not os.environ.get("USE_KEOPS")) or os.environ.get("USE_KEOPS")=="False":
    from DGM.layers_dense import *

    
class DGM_Model(pl.LightningModule):
    def __init__(self, hparams):
        super(DGM_Model,self).__init__()
        
        if type(hparams) is not Namespace:
            hparams = Namespace(**hparams)
        
#         self.hparams=hparams
        self.save_hyperparameters(hparams)
        conv_layers = hparams.conv_layers
        fc_layers = hparams.fc_layers
        dgm_layers = hparams.dgm_layers
        k = hparams.k
        self.classification = hparams.classification
            
        self.graph_f = ModuleList() 
        self.node_g = ModuleList() 
        for i,(dgm_l,conv_l) in enumerate(zip(dgm_layers,conv_layers)):
            if len(dgm_l)>0:
                if 'ffun' not in hparams or hparams.ffun == 'gcn':
                    self.graph_f.append(DGM_d(GCNConv(dgm_l[0],dgm_l[-1]),k=hparams.k,distance=hparams.distance))
                if hparams.ffun == 'gat':
                    self.graph_f.append(DGM_d(GATConv(dgm_l[0],dgm_l[-1]),k=hparams.k,distance=hparams.distance))
                if hparams.ffun == 'sage':
                    self.graph_f.append(DGM_d(SAGEConv(dgm_l[0],dgm_l[-1]),k=hparams.k,distance=hparams.distance))
                if hparams.ffun == 'graphconv':
                    self.graph_f.append(DGM_d(GraphConv(dgm_l[0],dgm_l[-1]),k=hparams.k,distance=hparams.distance))
                if hparams.ffun == 'mlp':
                    self.graph_f.append(DGM_d(MLP(dgm_l),k=hparams.k,distance=hparams.distance))
                if hparams.ffun == 'knn':
                    self.graph_f.append(DGM_d(Identity(retparam=0),k=hparams.k,distance=hparams.distance))
                if hparams.ffun == 'hanconv':
                    self.graph_f.append(DGM_d(LINKX(dgm_l[0],dgm_l[-1]),k=hparams.k,distance=hparams.distance))
            else:
                self.graph_f.append(Identity())
            
            if hparams.gfun == 'edgeconv':
                conv_l=conv_l.copy()
                conv_l[0]=conv_l[0]*2
                print("using EdgeConv")
                self.node_g.append(EdgeConv(MLP(conv_l), hparams.pooling))
            if hparams.gfun == 'gcn':
                print("using GCN")
                self.node_g.append(GCNConv(conv_l[0],conv_l[1]))
            if hparams.gfun == 'gat':
                print("using GAT")
                self.node_g.append(GATConv(conv_l[0],conv_l[1]))
            if hparams.gfun == 'sage':
                print("using SAGE")
                self.node_g.append(SAGEConv(conv_l[0],conv_l[1]))
            if hparams.gfun == 'graphconv':
                print("using GraphConv")
                self.node_g.append(GraphConv(conv_l[0],conv_l[1]))
            if hparams.gfun == 'cheb':
                print("using cheb")
                self.node_g.append(ChebConv(conv_l[0],conv_l[1],K=3))
            if hparams.gfun == "hanconv":
                print("using HANConv")
                self.node_g.append(LINKX(conv_l[0],conv_l[1]))
        
        self.fc = MLP(fc_layers, final_activation=False)
        if hparams.pre_fc is not None and len(hparams.pre_fc)>0:
            self.pre_fc = MLP(hparams.pre_fc, final_activation=True)
        self.avg_accuracy = None
        self.avg_error=None
        self.avg_mae = None
        
        #torch lightning specific
        self.automatic_optimization = False
        self.debug=False
        
    def forward(self, x, edges=None):
        if self.hparams.pre_fc is not None and len(self.hparams.pre_fc)>0:
            x = self.pre_fc(x)
            
        graph_x = x.detach()
        lprobslist = []
        for f,g in zip(self.graph_f, self.node_g):
            graph_x,edges,lprobs = f(graph_x,edges,None) # with cDGM, edges is the adjacency matrix
            b,n,d = x.shape
            self.edges = edges

            x = torch.nn.functional.relu(g(torch.dropout(x.view(-1,d), self.hparams.dropout, train=self.training), edges)).view(b,n,-1)
            graph_x = torch.cat([graph_x,x.detach()],-1)
            if lprobs is not None:
                lprobslist.append(lprobs)
                
        return self.fc(x),torch.stack(lprobslist,-1) if len(lprobslist)>0 else None, edges
   
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        
        optimizer = self.optimizers(use_pl_optimizer=True)
        optimizer.zero_grad()
        
        X, y, mask, edges = train_batch
        edges = edges[0]
        
        assert(X.shape[0]==1) #only works in transductive setting
        mask=mask[0]
        
        pred, logprobs, edges_hat = self(X,edges)

        self.log("nr_edges", edges_hat.shape[-1], on_epoch=True, prog_bar=True)

        if self.classification == 1:
            train_pred = pred[:,mask.to(torch.bool),:]
            train_lab = y[:,mask.to(torch.bool),:]
        else:
            pred = pred.squeeze_(-1) 
            train_pred = pred[:,mask.to(torch.bool)]
            train_lab = y[:,mask.to(torch.bool)]

        if self.classification ==1:
            correct_t = (train_pred.argmax(-1) == train_lab.argmax(-1)).float().mean().item()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(train_pred,train_lab)

        else:
            loss = torch.nn.HuberLoss()(train_pred.squeeze_(), train_lab.squeeze_())

        loss.backward()

        #GRAPH LOSS
        if self.classification ==1:
            if logprobs is not None: 
                corr_pred = (train_pred.argmax(-1)==train_lab.argmax(-1)).float().detach()
                # wron_pred = (1-corr_pred)

                if self.avg_accuracy is None:
                    self.avg_accuracy = torch.ones_like(corr_pred)*0.5
                point_w = (self.avg_accuracy-corr_pred)#*(1*corr_pred + self.k*(1-corr_pred))
                graph_loss = point_w * logprobs[:,mask.to(torch.bool),:].exp().mean([-1,-2])
                graph_loss = graph_loss.mean()# + self.graph_f[0].Pr.abs().sum()*1e-3
                graph_loss.backward()
                
                if(self.debug):
                    self.point_w = point_w.detach().cpu()
                    
                self.avg_accuracy = self.avg_accuracy.to(corr_pred.device)*0.95 +  0.05*corr_pred
            optimizer.step()
        else:
            abs_error = abs(train_pred.squeeze_() - train_lab.squeeze_()).mean().item()
        
            # GRAPH LOSS
            if logprobs is not None: 
                mae = abs(train_pred.squeeze_() - train_lab.squeeze_()).detach()

                if self.avg_mae is None:
                    self.avg_mae = torch.ones_like(mae)*6

                point_w = (self.avg_mae - mae)#*(1*corr_pred + self.k*(1-corr_pred))
                graph_loss = 1 - (point_w * logprobs[:,mask.to(torch.bool)].exp().mean([-1,-2]))

                graph_loss = graph_loss.mean()# + self.graph_f[0].Pr.abs().sum()*1e-3
                graph_loss.backward()
                
                self.log('train_graph_loss', graph_loss.detach().cpu())
            
                if(self.debug):
                    self.point_w = point_w.detach().cpu()
                    
                self.avg_mae = self.avg_mae.to(mae.device)*0.95 +  0.05*mae
                
            optimizer.step()

        if self.hparams.assess_graph==1:
            
            if self.hparams.classification ==1:
                ratio, nr_same_labelled_neighbours, nr_diff_labelled_neighbours = graph_metrics.graph_neighbourhood_assessment_static(edges_hat, X.shape[1], y.squeeze(0).argmax(dim=-1), train_mask=mask)
                ccns = graph_metrics.get_cross_class_neighbourhood_similarity_edges(edges_hat, y, one_hot_input=True, train_mask=mask)
                self.log("homophily_mean_train", ratio.mean().item(), on_epoch=True)
                self.log("homophily_std_train", ratio.std().item(), on_epoch=True)
                self.log("nr_same_labelled_neighbours_train", nr_same_labelled_neighbours.mean().item(), on_epoch=True)
                self.log("nr_diff_labelled_neighbours_train", nr_diff_labelled_neighbours.mean().item(), on_epoch=True)
                l1_ccns = torch.nn.functional.l1_loss(ccns, torch.eye(ccns.shape[0]).to(ccns.device))
                self.log("l1_ccns_train", l1_ccns.item(), on_epoch=True, prog_bar=True)
            else:
                homophily_mean, homophily_std = graph_metrics.graph_neighbourhood_assessment_regression_static(edges_hat, X.shape[1], y.squeeze(0), mask)
                self.log("homophily_mean_train", homophily_mean, on_epoch=True)
                self.log("homophily_std_train", homophily_std, on_epoch=True)

        # log accuracy only in case of classification task
        if self.classification == 1:
            self.log('train_acc', correct_t, on_epoch=True)
            self.log("train_mcc", sklearn.metrics.matthews_corrcoef(train_lab.argmax(-1).squeeze().cpu(), train_pred.argmax(-1).squeeze().cpu()))
        else:
            mae = torchmetrics.functional.mean_absolute_error(train_pred.squeeze_(), train_lab.squeeze_())
            self.log('train_mae', mae, on_epoch=True)

        # always log loss 
        self.log('train_loss', loss.detach().cpu(), on_epoch=True)
        self.log("graph_loss", graph_loss.detach().cpu(), on_epoch=True)
        
    
    def test_step(self, test_batch, batch_idx):
        X, y, mask, edges = test_batch
        edges = edges[0]
        
        assert(X.shape[0]==1) 
        mask=mask[0]
        pred,logprobs,edges_hat = self(X,edges)
        if self.classification ==1:
            pred = pred.softmax(-1)

        test_pred = pred[:,mask.to(torch.bool),:]
        test_lab = y[:,mask.to(torch.bool)]

        # define loss function depending on classification or regression task
        if self.classification ==1:
            correct_t = (test_pred.argmax(-1) == test_lab.argmax(-1)).float().mean().item()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(test_pred,test_lab)
        else:
            # loss = torch.nn.functional.l1_loss(test_pred, test_lab.float())
            loss = torch.nn.HuberLoss()(test_pred.squeeze_(), test_lab.squeeze_())


        if self.hparams.assess_graph==1:
            if self.hparams.classification ==1:
                ratio, nr_same_labelled_neighbours, nr_diff_labelled_neighbours = graph_metrics.graph_neighbourhood_assessment_static(edges_hat, X.shape[1], y.squeeze(0).argmax(dim=-1), train_mask=mask)
                ccns = graph_metrics.get_cross_class_neighbourhood_similarity_edges(edges_hat, y, one_hot_input=True, train_mask=mask)
                self.log("homophily_mean_test", ratio.mean().item(), on_epoch=True)
                self.log("homophily_std_test", ratio.std().item(), on_epoch=True)

                self.log("nr_same_labelled_neighbours_test", nr_same_labelled_neighbours.mean().item(), on_epoch=True)
                self.log("nr_diff_labelled_neighbours_test", nr_diff_labelled_neighbours.mean().item(), on_epoch=True)
                l1_ccns = torch.nn.functional.l1_loss(ccns, torch.eye(ccns.shape[0]).to(ccns.device))
                self.log("l1_ccns_test", l1_ccns.item(), on_epoch=True)
            else:
                homophily_mean, homophily_std = graph_metrics.graph_neighbourhood_assessment_regression_static(edges_hat, X.shape[-2], y.squeeze(0), mask)
                self.log("homophily_mean_test", homophily_mean.item(), on_epoch=True)
                self.log("homophily_std_test", homophily_std.item(), on_epoch=True)

        self.log('test_loss', loss.detach().cpu(), on_epoch=True)
        if self.classification ==1:
            self.log('test_acc', correct_t, on_epoch=True)
        else:
            mae_test = torchmetrics.functional.mean_absolute_error(test_pred.squeeze_(), test_lab.squeeze_())
            self.log('test_mae', mae_test, on_epoch=True)

    
    def validation_step(self, val_batch, batch_idx):
        X, y, mask, edges = val_batch
        edges = edges[0]
        
        assert(X.shape[0]==1) 
        mask=mask[0]

        pred,logprobs,edges_hat = self(X,edges)
        if self.classification ==1:
            pred = pred.softmax(-1)

        test_pred = pred[:,mask.to(torch.bool),:]
        test_lab = y[:,mask.to(torch.bool)]

        # define loss function depending on classification or regression task
        if self.classification ==1:
            correct_t_val = (test_pred.argmax(-1) == test_lab.argmax(-1)).float().mean().item()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(test_pred, test_lab)
        else:
            loss = torch.nn.HuberLoss()(test_pred.squeeze_(), test_lab.squeeze_())

        self.log('val_loss', loss.detach().cpu(), on_epoch=True)
        if self.classification ==1:
            self.log('val_acc', correct_t_val, on_epoch=True)
        else:
            mae_test = torchmetrics.functional.mean_absolute_error(test_pred.squeeze_(), test_lab.squeeze_())
            self.log('val_mae', mae_test, on_epoch=True)