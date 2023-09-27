import torch

from torch.nn import ModuleList
from torch_geometric.nn import EdgeConv, DenseGCNConv, DenseGATConv

import pytorch_lightning as pl
from argparse import Namespace
import torchmetrics

import sys
sys.path.append("../source/")
from utils import utils
from utils import graph_metrics
from DGM.layers import *


class cDGM_Model(pl.LightningModule):
    def __init__(self, hparams):
        super(cDGM_Model,self).__init__()
        
        if type(hparams) is not Namespace:
            hparams = Namespace(**hparams)
            
        # self.hparams=hparams
        self.save_hyperparameters(hparams)
        conv_layers = hparams.conv_layers
        fc_layers = hparams.fc_layers
        dgm_layers = hparams.dgm_layers
        self.k = hparams.k
        self.classification = hparams.classification
            
        self.graph_f = ModuleList() 
        self.node_g = ModuleList() 
        for i,(dgm_l,conv_l) in enumerate(zip(dgm_layers,conv_layers)):
            if len(dgm_l)>0:
                self.graph_f.append(DGM_c(MLP(dgm_l),distance=hparams.distance))
            else:
                self.graph_f.append(Identity())
            
            if hparams.gfun == 'edgeconv':
                conv_l=conv_l.copy()
                conv_l[0]=conv_l[0]*2
                self.node_g.append(EdgeConv(MLP(conv_l), hparams.pooling))
            if hparams.gfun == 'gcn':
                self.node_g.append(DenseGCNConv(conv_l[0],conv_l[1]))
            if hparams.gfun == 'gat':
                self.node_g.append(DenseGATConv(conv_l[0],conv_l[1]))
        
        self.fc = MLP(fc_layers, final_activation=False)
        if hparams.pre_fc is not None and len(hparams.pre_fc)>0:
            self.pre_fc = MLP(hparams.pre_fc, final_activation=True)
        self.avg_accuracy = None
        
        #torch lightning specific
        self.automatic_optimization = False
        self.debug=False
        
    def forward(self,x, edges=None):
        if self.hparams.pre_fc is not None and len(self.hparams.pre_fc)>0:
            x = self.pre_fc(x)

        graph_x = x.detach()
        lprobslist = []
        for f,g in zip(self.graph_f, self.node_g):
            graph_x,edges,lprobs = f(graph_x,edges,None) # edges is the continous weighted adjacency matrix
            b,n,d = x.shape

            self.edges=edges
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
        
        assert(X.shape[0]==1) #only works in transductive setting
        mask=mask[0]
        
        pred,logprobs, edges_hat = self(X)
        
        if self.classification==1:
            train_pred = pred[:,mask.to(torch.bool),:]
            train_lab = y[:,mask.to(torch.bool),:]
        else:
            pred = pred.squeeze_(-1) 
            train_pred = pred[:,mask.to(torch.bool)]
            train_lab = y[:,mask.to(torch.bool)]
#         train_w = weight[None,mask.to(torch.bool)]    

        if self.hparams.assess_graph==1:
            if self.hparams.classification ==1:
                ratio, nr_same_labelled_neighbours, nr_diff_labelled_neighbours = graph_metrics.graph_neighbourhood_assessment_lgl(edges_hat, y.squeeze(0).argmax(dim=-1), train_mask=mask)
                ccns = graph_metrics.get_cross_class_neighbourhood_similarity_adj(edges_hat, y, one_hot=True, train_mask=mask)
                self.log("homophily_mean_train", ratio.mean().item(), on_epoch=True)
                self.log("homophily_std_train", ratio.std().item(), on_epoch=True)
                self.log("nr_same_labelled_neighbours_train", nr_same_labelled_neighbours.mean().item(), on_epoch=True)
                self.log("nr_diff_labelled_neighbours_train", nr_diff_labelled_neighbours.mean().item(), on_epoch=True)
                l1_ccns = torch.nn.functional.l1_loss(ccns, torch.eye(ccns.shape[0]).to(ccns.device))
                self.log("l1_ccns_train", l1_ccns.item(), on_epoch=True, prog_bar=True)
            else:
                homophily_mean, homophily_std = graph_metrics.graph_neighbourhood_assessment_regression_discrete(edges_hat, y, mask)
                self.log("homophily_mean_train", homophily_mean, on_epoch=True)
                self.log("homophily_std_train", homophily_std, on_epoch=True)

        #loss = torch.nn.functional.cross_entropy(train_pred.view(-1,train_pred.shape[-1]),train_lab.argmax(-1).flatten())
        # loss = torch.nn.functional.binary_cross_entropy_with_logits(train_pred,train_lab)
        if self.classification ==1:
            correct_t = (train_pred.argmax(-1) == train_lab.argmax(-1)).float().mean().item()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(train_pred,train_lab)
        else:
            # loss = torch.nn.functional.l1_loss(train_pred, train_lab.float())
            loss = torch.nn.HuberLoss()(train_pred.squeeze_(), train_lab.squeeze_())
            mae = torchmetrics.functional.mean_absolute_error(train_pred.squeeze_(), train_lab.squeeze_())
            self.log("mae_train", mae.item(), on_epoch=True)

        loss.backward()

        # correct_t = (train_pred.argmax(-1) == train_lab.argmax(-1)).float().mean().item()

# #         #GRAPH LOSS
#         corr_pred = (train_pred.argmax(-1)==train_lab.argmax(-1)).float().detach()
#         wron_pred = (1-corr_pred)
    
#         if self.avg_accuracy is None:
#             self.avg_accuracy = torch.ones_like(corr_pred)*0.5
        
#         point_w = (self.avg_accuracy-corr_pred)#*(1*corr_pred + self.k*(1-corr_pred))
#         graph_loss = point_w * logprobs[:,mask.to(torch.bool),:].exp().mean([-1,-2])
    
#         graph_loss = graph_loss.mean()# + self.graph_f[0].Pr.abs().sum()*1e-3
#         graph_loss.backward()
        
        optimizer.step()
#         self.point_w = point_w.detach()
#         self.avg_accuracy = self.avg_accuracy.to(corr_pred.device)*0.95 +  0.05*corr_pred

        if self.classification ==1:
            self.log('train_acc', correct_t)
        self.log('train_loss', loss.detach().cpu())
#         self.log('train_graph_loss', graph_loss.detach().cpu())
#         return loss.detach()#, graph_loss.item()
        # if(self.debug):
        #     self.point_w = point_w.detach().cpu()

    
    def test_step(self, train_batch, batch_idx):
        X, y, mask, edges = train_batch
        assert(X.shape[0]==1) #only works in transductive setting
        mask=mask[0]
        pred,logprobs, edges = self(X)
        pred = pred.softmax(-1)
        for i in range(1,self.hparams.test_eval):
            pred_,logprobs, edges_hat = self(X)
            pred+=pred_.softmax(-1)
            
        if self.hparams.assess_graph==1:
            if self.hparams.classification ==1:
                ratio, nr_same_labelled_neighbours, nr_diff_labelled_neighbours = graph_metrics.graph_neighbourhood_assessment_lgl(edges_hat, y.squeeze(0).argmax(dim=-1), train_mask=mask)
                ccns = graph_metrics.get_cross_class_neighbourhood_similarity_adj(edges_hat, y, one_hot=True, train_mask=mask)
                self.log("homophily_mean_test", ratio.mean().item(), on_epoch=True)
                self.log("homophily_std_test", ratio.std().item(), on_epoch=True)
                self.log("nr_same_labelled_neighbours_test", nr_same_labelled_neighbours.mean().item(), on_epoch=True)
                self.log("nr_diff_labelled_neighbours_test", nr_diff_labelled_neighbours.mean().item(), on_epoch=True)
                l1_ccns = torch.nn.functional.l1_loss(ccns, torch.eye(ccns.shape[0]).to(ccns.device))
                self.log("l1_ccns_test", l1_ccns.item(), on_epoch=True, prog_bar=True)
            else:
                homophily_mean, homophily_std = graph_metrics.graph_neighbourhood_assessment_regression_discrete(edges_hat, y, mask)
                self.log("homophily_mean_test", homophily_mean, on_epoch=True)
                self.log("homophily_std_test", homophily_std, on_epoch=True)
            
        test_pred = pred[:,mask.to(torch.bool),:]
        test_lab = y[:,mask.to(torch.bool),:]
        correct_t = (test_pred.argmax(-1) == test_lab.argmax(-1)).float().mean().item()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(test_pred,test_lab)
        self.log('test_loss', loss.detach().cpu())
#         self.log('test_graph_loss', loss.detach().cpu())
        if self.classification == 1:
            self.log('test_acc', correct_t)
    
    def validation_step(self, train_batch, batch_idx):
        X, y, mask, edges = train_batch
        
        assert(X.shape[0]==1) #only works in transductive setting
        mask=mask[0]
        
        pred,logprobs, edges = self(X)
        pred = pred.softmax(-1)
        for i in range(1,self.hparams.test_eval):
            pred_,logprobs, edges_hat = self(X)
            pred = pred + pred_.softmax(-1)
        
        test_pred = pred[:,mask.to(torch.bool),:]
        test_lab = y[:,mask.to(torch.bool),:]
        # correct_t = (test_pred.argmax(-1) == test_lab.argmax(-1)).float().mean().item()
        # loss = torch.nn.functional.binary_cross_entropy_with_logits(test_pred,test_lab)
        if self.classification ==1:
            correct_t = (test_pred.argmax(-1) == test_lab.argmax(-1)).float().mean().item()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(test_pred,test_lab)
            self.log('val_acc', correct_t)
        else:
            loss = torch.nn.functional.l1_loss(test_pred, test_lab.float())
        
        self.log('val_loss', loss.detach().cpu())

        if self.hparams.assess_graph==1:
            if self.hparams.classification ==1:
                ratio, nr_same_labelled_neighbours, nr_diff_labelled_neighbours = graph_metrics.graph_neighbourhood_assessment_lgl(edges_hat, y.squeeze(0).argmax(dim=-1), train_mask=mask)
                ccns = graph_metrics.get_cross_class_neighbourhood_similarity_adj(edges_hat, y, one_hot=True, train_mask=mask)
                self.log("homophily_mean_val", ratio.mean().item(), on_epoch=True)
                self.log("homophily_std_val", ratio.std().item(), on_epoch=True)
                self.log("nr_same_labelled_neighbours_val", nr_same_labelled_neighbours.mean().item(), on_epoch=True)
                self.log("nr_diff_labelled_neighbours_val", nr_diff_labelled_neighbours.mean().item(), on_epoch=True)
                l1_ccns = torch.nn.functional.l1_loss(ccns, torch.eye(ccns.shape[0]).to(ccns.device))
                self.log("l1_ccns_val", l1_ccns.item(), on_epoch=True, prog_bar=True)
            else:
                homophily_mean, homophily_std = graph_metrics.graph_neighbourhood_assessment_regression_discrete(edges_hat, y, mask)
                self.log("homophily_mean_val", homophily_mean, on_epoch=True)
                self.log("homophily_std_val", homophily_std, on_epoch=True)
#         self.log('val_graph_loss', loss)

        
#         ####### visualizations ###########
#         try:
#             self.graph_f[0].debug=True
#             pred,logprobs = self(X)
#             self.graph_f[0].debug=False

#             x = self.graph_f[0].x[0].detach()
#             c = torch.argmax(y,-1)
#             D = self.graph_f[0].distance(x)[0]
#             D.diagonal().fill_(0)

#             sidx = torch.argsort( (c[0]+1)*10 + (mask+1)*1)
#             P = torch.exp(-D[sidx,:][:,sidx]*torch.clamp(self.graph_f[0].temperature.detach().cpu(),-5,5).exp())#>0.001

#             img = PIL.Image.fromarray((P*255).byte().detach().cpu().numpy())
#             img = img.resize((512,512), PIL.Image.ANTIALIAS)

#             I = wandb.Image(img, caption="adj")
#             self.logger.experiment.log({'adj': [I]})
#         except:
#             pass
      

