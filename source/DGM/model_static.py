import os
import torch

from torch.nn import ModuleList
from torch_geometric.nn import GCNConv, GATConv, GraphConv, SAGEConv, ChebConv

import pytorch_lightning as pl
from argparse import Namespace
import torchmetrics

import sys
sys.path.append("../source/")
from utils import graph_metrics
from DGM.layers import *
if (not os.environ.get("USE_KEOPS")) or os.environ.get("USE_KEOPS")=="False":
    from DGM.layers_dense import *


class PatientPopulationModel(pl.LightningModule):
    def __init__(self,  hparams, nr_node_feautres, nr_classes):
        super(PatientPopulationModel,self).__init__()        
        if type(hparams) is not Namespace:
            hparams = Namespace(**hparams)
        
#         self.hparams=hparams
        self.save_hyperparameters(hparams)
        model_type = hparams.model_type
        self.nr_node_feautres = nr_node_feautres
        k = hparams.k
        self.classification = hparams.classification
            
        self.graph_f = ModuleList() 
        self.node_g = ModuleList() 

        self.hidden_channels = hparams.hidden_channels
        self.dropout = hparams.dropout
        self.conv_layers = torch.nn.ModuleList()
        self.nr_layers = hparams.num_layers
        self.activation = torch.nn.LeakyReLU()
        self.num_tasks = nr_classes

        self.logged_homophily = False

        first_conv_dim = self.nr_node_feautres if hparams.pre_fc is None else hparams.pre_fc[-1]
        
        if model_type == "GCN":
            print("using GCN model")
            self.model_func = GCNConv
        elif model_type == "GraphSAGE":
            print("using SAGE model")
            self.model_func = SAGEConv
        elif model_type == "GAT": 
            print("using GAT model")
            self.model_func = GATConv
        elif model_type == "GraphConv":
            print("using GraphConv")
            self.model_func = GraphConv
        elif model_type == "Cheb":
            print("using Cheb convolution")
            self.model_func = ChebConv
        else:
            print("model type not implemented. Please use one of the following model types: GCN, GAT, GraphSAGE, GIN, Cheb")
            exit(-1)

        if model_type=="GAT":
            heads=3
            if self.nr_layers == 1:
                self.conv_layers.append(self.model_func(first_conv_dim, self.hidden_channels, heads=heads))
                self.lin = torch.nn.Sequential(torch.nn.Linear(self.hidden_channels*heads, self.hidden_channels*2), self.activation, torch.nn.Linear(self.hidden_channels*2, 125), self.activation, torch.nn.Linear(125, self.num_tasks))
            else:
                for i in range(self.nr_layers-1):
                    if i == 0:
                        self.conv_layers.append(self.model_func(first_conv_dim, self.hidden_channels, heads=heads))
                    else:
                        self.conv_layers.append(self.model_func(self.hidden_channels*(i)*heads, self.hidden_channels*(i+1), heads=heads))
                    
                self.conv_layers.append(self.model_func(self.hidden_channels*(self.nr_layers-1)*heads, self.hidden_channels*4, heads=heads))  
                self.lin = torch.nn.Sequential(torch.nn.Linear(self.hidden_channels*4*heads, self.hidden_channels*2), self.activation, torch.nn.Linear(self.hidden_channels*2, 125), self.activation, torch.nn.Linear(125, self.num_tasks))
        
        elif model_type=="Cheb":
            k = 3
            if self.nr_layers == 1:
                self.conv_layers.append(self.model_func(first_conv_dim, self.hidden_channels, k))
                self.lin = torch.nn.Sequential(torch.nn.Linear(self.hidden_channels, self.hidden_channels*2), self.activation, torch.nn.Linear(self.hidden_channels*2, 125), self.activation, torch.nn.Linear(125, self.num_tasks))
            else:
                for i in range(self.nr_layers-1):
                    if i == 0:
                        self.conv_layers.append(self.model_func(first_conv_dim, self.hidden_channels, k))
                    else:
                        self.conv_layers.append(self.model_func(self.hidden_channels*(i), self.hidden_channels*(i+1), k))
                    
                self.conv_layers.append(self.model_func(self.hidden_channels*(self.nr_layers-1), self.hidden_channels*4, k))  
                self.lin = torch.nn.Sequential(torch.nn.Linear(self.hidden_channels*4, self.hidden_channels*2), self.activation, torch.nn.Linear(self.hidden_channels*2, 125), self.activation, torch.nn.Linear(125, self.num_tasks))
        
        else:
            if self.nr_layers == 1:
                self.conv_layers.append(self.model_func(first_conv_dim, self.hidden_channels))
                self.lin = torch.nn.Sequential(torch.nn.Linear(self.hidden_channels, self.hidden_channels*2), self.activation, torch.nn.Linear(self.hidden_channels*2, 125), self.activation, torch.nn.Linear(125, self.num_tasks))
            else:
                for i in range(self.nr_layers-1):
                    if i == 0:
                        self.conv_layers.append(self.model_func(first_conv_dim, self.hidden_channels))
                    else:
                        self.conv_layers.append(self.model_func(self.hidden_channels*(i), self.hidden_channels*(i+1)))
                    
                self.conv_layers.append(self.model_func(self.hidden_channels*(self.nr_layers-1), self.hidden_channels*4))  
                self.lin = torch.nn.Sequential(torch.nn.Linear(self.hidden_channels*4, self.hidden_channels*2), torch.nn.LeakyReLU(), torch.nn.Linear(self.hidden_channels*2, 125), torch.nn.LeakyReLU(), torch.nn.Linear(125, self.num_tasks))
        
        if hparams.pre_fc is not None and len(hparams.pre_fc)>0:
            self.pre_fc = MLP(hparams.pre_fc, final_activation=True)
        self.avg_accuracy = None
        self.avg_error=None
        self.avg_mae = None
        
        self.automatic_optimization = False
        self.debug=False


    def forward(self,x, edges=None):
        if self.hparams.pre_fc is not None and len(self.hparams.pre_fc)>0:
            x = self.pre_fc(x)
            
        for conv_layer in self.conv_layers:
            if x.dim() == 3:
                x = x.squeeze(0)
            x = conv_layer(x, edges)
            if x.dim() == 2:
                x = x.unsqueeze(0)
            x = self.activation(x)
            if self.dropout > 0:
                x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        return x,None, edges

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        
        optimizer = self.optimizers(use_pl_optimizer=True)
        optimizer.zero_grad()
        
        X, y, mask, edges = train_batch
        edges = edges[0]

        if self.logged_homophily == False:
            hom,_,_ = graph_metrics.graph_neighbourhood_assessment_static(edges, X.shape[1], y.argmax(dim=-1), train_mask=None)
            self.log("homophily", hom.mean().item(), on_epoch=True)
            ccns = graph_metrics.get_cross_class_neighbourhood_similarity_edges(edges, y, one_hot_input=True, train_mask=None)
            self.log("ccns", ccns.mean().item(), on_epoch=True)
            print(hom.mean().item())
            self.logged_homophily = True
        
        assert(X.shape[0]==1) #only works in transductive setting
        mask=mask[0]
        
        pred, logprobs, _ = self(X,edges)
        self.log("nr_edges", edges.shape[-1], on_epoch=True, prog_bar=True)

        if self.classification == 1:
            train_pred = pred[:,mask.to(torch.bool),:]
            train_lab = y[:,mask.to(torch.bool),:]
        else:
            pred = pred.squeeze_(-1) 
            train_pred = pred[:,mask.to(torch.bool)]
            train_lab = y[:,mask.to(torch.bool)]

        # define loss function depending on classification or regression task
        if self.classification ==1:
            correct_t = (train_pred.argmax(-1) == train_lab.argmax(-1)).float().mean().item()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(train_pred,train_lab)
        else:
            loss = torch.nn.HuberLoss()(train_pred.squeeze_(), train_lab.squeeze_())

        if self.hparams.assess_graph==1:
            if self.hparams.classification ==1:
                ratio, nr_same_labelled_neighbours, nr_diff_labelled_neighbours = graph_metrics.graph_neighbourhood_assessment_static(edges, X.shape[1], y.squeeze(0).argmax(dim=-1), train_mask=mask)
                ccns = graph_metrics.get_cross_class_neighbourhood_similarity_edges(edges, y, one_hot_input=True, train_mask=mask)
                self.log("homophily_mean_train", ratio.mean().item(), on_epoch=True)
                self.log("homophily_std_train", ratio.std().item(), on_epoch=True)
                self.log("nr_same_labelled_neighbours_train", nr_same_labelled_neighbours.mean().item(), on_epoch=True)
                self.log("nr_diff_labelled_neighbours_train", nr_diff_labelled_neighbours.mean().item(), on_epoch=True)
                l1_ccns = torch.nn.functional.l1_loss(ccns, torch.eye(ccns.shape[0]).to(ccns.device))
                self.log("l1_ccns_train", l1_ccns.item(), on_epoch=True, prog_bar=True)
            else:
                homophily_mean, homophily_std = graph_metrics.graph_neighbourhood_assessment_regression_static(edges, X.shape[-2], y.squeeze(0), mask)
                self.log("homophily_mean_train", homophily_mean, on_epoch=True)
                self.log("homophily_std_train", homophily_std, on_epoch=True)

        loss.backward()
        optimizer.step()

        # log accuracy only in case of classification task
        if self.classification ==1:
            self.log('train_acc', correct_t, on_epoch=True)
        else:
            mae = torchmetrics.functional.mean_absolute_error(train_pred.squeeze_(), train_lab.squeeze_())
            self.log('train_mae', mae, on_epoch=True)

        # always log loss 
        self.log('train_loss', loss.detach().cpu(), on_epoch=True)
        
    
    def test_step(self, test_batch, batch_idx):
        X, y, mask, edges = test_batch
        edges = edges[0]
        
        assert(X.shape[0]==1) # only works in transductive setting
        mask=mask[0]
        pred,_,_ = self(X,edges)
        if self.classification ==1:
            pred = pred.softmax(-1)

        test_pred = pred[:,mask.to(torch.bool),:]
        test_lab = y[:,mask.to(torch.bool),:]

        # define loss function depending on classification or regression task
        if self.classification ==1:
            correct_t = (test_pred.argmax(-1) == test_lab.argmax(-1)).float().mean().item()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(test_pred,test_lab)
        else:
            loss = torch.nn.HuberLoss()(test_pred.squeeze_(), test_lab.squeeze_())


        if self.hparams.assess_graph==1:
            if self.hparams.classification ==1:
                ratio, nr_same_labelled_neighbours, nr_diff_labelled_neighbours = graph_metrics.graph_neighbourhood_assessment_static(edges, X.shape[1], y.squeeze(0).argmax(dim=-1), train_mask=mask)
                ccns = graph_metrics.get_cross_class_neighbourhood_similarity_edges(edges, y, one_hot_input=True, train_mask=mask)
                self.log("homophily_mean_test", ratio.mean().item(), on_epoch=True)
                self.log("homophily_std_test", ratio.std().item(), on_epoch=True)

                self.log("nr_same_labelled_neighbours_test", nr_same_labelled_neighbours.mean().item(), on_epoch=True)
                self.log("nr_diff_labelled_neighbours_test", nr_diff_labelled_neighbours.mean().item(), on_epoch=True)
                l1_ccns = torch.nn.functional.l1_loss(ccns, torch.eye(ccns.shape[0]).to(ccns.device))
                self.log("l1_ccns_test", l1_ccns.item(), on_epoch=True)
            else:
                # print("___________________________________--")
                # print(y.shape, edges_hat.shape, mask.shape)
                homophily_mean, homophily_std = graph_metrics.graph_neighbourhood_assessment_regression_static(edges, X.shape[-2], y.squeeze(0), mask)

                self.log("homophily_mean_test", homophily_mean.item(), on_epoch=True)
                self.log("homophily_std_test", homophily_std.item(), on_epoch=True)

        self.log('test_loss', loss.detach().cpu(), on_epoch=True)
#         self.log('test_graph_loss', loss.detach().cpu())
        if self.classification ==1:
            self.log('test_acc', correct_t, on_epoch=True)
        else:
            mae_test = torchmetrics.functional.mean_absolute_error(test_pred.squeeze_(), test_lab.squeeze_())
            self.log('test_mae', mae_test, on_epoch=True)

    
    def validation_step(self, val_batch, batch_idx):
        X, y, mask, edges = val_batch
        edges = edges[0]
        
        assert(X.shape[0]==1) # only works in transductive setting
        mask=mask[0]

        pred,logprobs,_ = self(X,edges)
        if self.classification ==1:
            pred = pred.softmax(-1)
        
        test_pred = pred[:,mask.to(torch.bool),:]
        test_lab = y[:,mask.to(torch.bool)]

        if self.classification ==1:
            correct_t_val = (test_pred.argmax(-1) == test_lab.argmax(-1)).float().mean().item()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(test_pred,test_lab)
        else:
            loss = torch.nn.HuberLoss()(test_pred.squeeze_(), test_lab.squeeze_())

        self.log('val_loss', loss.detach(), on_epoch=True)
        if self.classification ==1:
            self.log('val_acc', correct_t_val)
        else:
            mae_val = torchmetrics.functional.mean_absolute_error(test_pred.squeeze_(), test_lab.squeeze_())
            self.log('val_mae', mae_val, on_epoch=True)
        


class GATPopulationAttention(pl.LightningModule):
    def __init__(self,  hparams, nr_node_feautres, nr_classes):
        super(GATPopulationAttention,self).__init__()        
        if type(hparams) is not Namespace:
            hparams = Namespace(**hparams)
        
#         self.hparams=hparams
        self.save_hyperparameters(hparams)
        self.nr_node_feautres = nr_node_feautres
        self.classification = hparams.classification
            
        self.graph_f = ModuleList() 
        self.node_g = ModuleList() 

        self.hidden_channels = hparams.hidden_channels
        self.dropout = hparams.dropout
        self.conv_layers = torch.nn.ModuleList()
        self.nr_layers = hparams.num_layers
        self.activation = torch.nn.LeakyReLU()
        self.num_tasks = nr_classes
        self.model_func = GATConv
        first_conv_dim = self.nr_node_feautres if hparams.pre_fc is None else hparams.pre_fc[-1]
        
        heads=3
        if self.nr_layers == 1:
            self.conv_layers.append(self.model_func(first_conv_dim, self.hidden_channels, heads=heads))
            self.lin = torch.nn.Sequential(torch.nn.Linear(self.hidden_channels*heads, self.hidden_channels*2), self.activation, torch.nn.Linear(self.hidden_channels*2, 125), self.activation, torch.nn.Linear(125, self.num_tasks))
        else:
            for i in range(self.nr_layers-1):
                if i == 0:
                    self.conv_layers.append(self.model_func(first_conv_dim, self.hidden_channels, heads=heads))
                else:
                    self.conv_layers.append(self.model_func(self.hidden_channels*(i)*heads, self.hidden_channels*(i+1), heads=heads))
                
            self.conv_layers.append(self.model_func(self.hidden_channels*(self.nr_layers-1)*heads, self.hidden_channels*4, heads=heads))  
            self.lin = torch.nn.Sequential(torch.nn.Linear(self.hidden_channels*4*heads, self.hidden_channels*2), self.activation, torch.nn.Linear(self.hidden_channels*2, 125), self.activation, torch.nn.Linear(125, self.num_tasks))
        
        if hparams.pre_fc is not None and len(hparams.pre_fc)>0:
            self.pre_fc = MLP(hparams.pre_fc, final_activation=True)
        self.avg_accuracy = None
        self.avg_error=None
        self.avg_mae = None
        
        #torch lightning specific
        self.automatic_optimization = False
        self.debug=False

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
    
    def forward(self,x, edges=None):
        if self.hparams.pre_fc is not None and len(self.hparams.pre_fc)>0:
            x = self.pre_fc(x)
            
        for conv_layer in self.conv_layers:
            if x.dim() == 3:
                x = x.squeeze(0)
            x, (edge_index, attention_weights) = conv_layer(x, edges, return_attention_weights=True)
            if x.dim() == 2:
                x = x.unsqueeze(0)
            x = self.activation(x)
            if self.dropout > 0:
                x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)

        return x, attention_weights, edges

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
        
        pred, attention_weights, _ = self(X,edges)
        self.log("nr_edges", edges.shape[-1], on_epoch=True, prog_bar=True)

        if self.classification == 1:
            train_pred = pred[:,mask.to(torch.bool),:]
            train_lab = y[:,mask.to(torch.bool),:]
        else:
            pred = pred.squeeze_(-1) 
            train_pred = pred[:,mask.to(torch.bool)]
            train_lab = y[:,mask.to(torch.bool)]

        # define loss function depending on classification or regression task
        if self.classification ==1:
            correct_t = (train_pred.argmax(-1) == train_lab.argmax(-1)).float().mean().item()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(train_pred,train_lab)
        else:
            loss = torch.nn.HuberLoss()(train_pred.squeeze_(), train_lab.squeeze_())

        if self.hparams.assess_graph==1:
            if self.hparams.classification ==1:
                ratio, nr_same_labelled_neighbours, nr_diff_labelled_neighbours = graph_metrics.graph_neighbourhood_assessment_static(edges, X.shape[1], y.squeeze(0).argmax(dim=-1), train_mask=mask)
                ccns = graph_metrics.get_cross_class_neighbourhood_similarity_edges(edges, y, one_hot_input=True, train_mask=mask)
                self.log("homophily_mean_train", ratio.mean().item(), on_epoch=True)
                self.log("homophily_std_train", ratio.std().item(), on_epoch=True)
                self.log("nr_same_labelled_neighbours_train", nr_same_labelled_neighbours.mean().item(), on_epoch=True)
                self.log("nr_diff_labelled_neighbours_train", nr_diff_labelled_neighbours.mean().item(), on_epoch=True)
                l1_ccns = torch.nn.functional.l1_loss(ccns, torch.eye(ccns.shape[0]).to(ccns.device))
                self.log("l1_ccns_train", l1_ccns.item(), on_epoch=True, prog_bar=True)
            else:
                homophily_mean, homophily_std = graph_metrics.graph_neighbourhood_assessment_regression_static(edges, X.shape[-2], y.squeeze(0), mask)
                self.log("homophily_mean_train", homophily_mean, on_epoch=True)
                self.log("homophily_std_train", homophily_std, on_epoch=True)

        loss.backward()
        optimizer.step()

        # log accuracy only in case of classification task
        if self.classification ==1:
            self.log('train_acc', correct_t, on_epoch=True)
        else:
            mae = torchmetrics.functional.mean_absolute_error(train_pred.squeeze_(), train_lab.squeeze_())
            self.log('train_mae', mae, on_epoch=True)

        # always log loss 
        self.log('train_loss', loss.detach().cpu(), on_epoch=True)
        
    
    def test_step(self, test_batch, batch_idx):
        X, y, mask, edges = test_batch
        edges = edges[0]
        
        assert(X.shape[0]==1) # only works in transductive setting
        mask=mask[0]
        pred, attention_weights,_ = self(X,edges)
        if self.classification ==1:
            pred = pred.softmax(-1)

        test_pred = pred[:,mask.to(torch.bool),:]
        test_lab = y[:,mask.to(torch.bool),:]

        # define loss function depending on classification or regression task
        if self.classification ==1:
            correct_t = (test_pred.argmax(-1) == test_lab.argmax(-1)).float().mean().item()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(test_pred,test_lab)
        else:
            loss = torch.nn.HuberLoss()(test_pred.squeeze_(), test_lab.squeeze_())

        if self.hparams.assess_graph==1:
            if self.hparams.classification ==1:
                ratio, nr_same_labelled_neighbours, nr_diff_labelled_neighbours = graph_metrics.graph_neighbourhood_assessment_static(edges, X.shape[1], y.squeeze(0).argmax(dim=-1), train_mask=mask)
                ccns = graph_metrics.get_cross_class_neighbourhood_similarity_edges(edges, y, one_hot_input=True, train_mask=mask)
                self.log("homophily_mean_test", ratio.mean().item(), on_epoch=True)
                self.log("homophily_std_test", ratio.std().item(), on_epoch=True)

                self.log("nr_same_labelled_neighbours_test", nr_same_labelled_neighbours.mean().item(), on_epoch=True)
                self.log("nr_diff_labelled_neighbours_test", nr_diff_labelled_neighbours.mean().item(), on_epoch=True)
                l1_ccns = torch.nn.functional.l1_loss(ccns, torch.eye(ccns.shape[0]).to(ccns.device))
                self.log("l1_ccns_test", l1_ccns.item(), on_epoch=True)
            else:
                homophily_mean, homophily_std = graph_metrics.graph_neighbourhood_assessment_regression_static(edges, X.shape[-2], y.squeeze(0), mask)

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
        
        assert(X.shape[0]==1) # only works in transductive setting
        mask=mask[0]

        pred,logprobs,_ = self(X,edges)
        if self.classification ==1:
            pred = pred.softmax(-1)
        
        test_pred = pred[:,mask.to(torch.bool),:]
        test_lab = y[:,mask.to(torch.bool)]

        if self.classification ==1:
            correct_t_val = (test_pred.argmax(-1) == test_lab.argmax(-1)).float().mean().item()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(test_pred,test_lab)
        else:
            loss = torch.nn.HuberLoss()(test_pred.squeeze_(), test_lab.squeeze_())

        self.log('val_loss', loss.detach(), on_epoch=True)
        if self.classification ==1:
            self.log('val_acc', correct_t_val)
        else:
            mae_val = torchmetrics.functional.mean_absolute_error(test_pred.squeeze_(), test_lab.squeeze_())
            self.log('test_mae', mae_val, on_epoch=True)
