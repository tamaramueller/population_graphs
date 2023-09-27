import sys
sys.path.insert(0,'./keops')

import os
os.environ["USE_KEOPS"] = "False"

from torch.utils.data import DataLoader
from datasets import PlanetoidDataset, TadpoleDataset, SyntheticData, MyData
import pytorch_lightning as pl
from datasets import *
from DGM.model_dDGM import DGM_Model
from DGM.model_cDGM import cDGM_Model
from DGM.model_static import PatientPopulationModel

from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch_geometric

sys.path.append("../source/")
from utils import utils


def run_training_process(run_params):
    
    train_data = None
    test_data = None

    device="cuda"
    if run_params.dataset in ['Cora', 'CiteSeer', 'PubMed']:
        train_data = PlanetoidDataset(split='train', name=run_params.dataset, device=device, args=run_params)
        val_data = PlanetoidDataset(split='val', name=run_params.dataset, samples_per_epoch=1, args=run_params, device=device)
        test_data = PlanetoidDataset(split='test', name=run_params.dataset, samples_per_epoch=1, args=run_params, device=device)
    elif run_params.dataset == 'tadpole':
        train_data = TadpoleDataset(args=run_params,mask="train", device=device)
        test_data = TadpoleDataset(args=run_params, mask="test",samples_per_epoch=1, device=device)
        val_data = TadpoleDataset(args=run_params, mask="val",samples_per_epoch=1, device=device)
    elif run_params.dataset == "synthetic":
        train_data = SyntheticData(run_params, device=device, samples_per_epoch=1, seed=run_params.seed, split="train")
        val_data = SyntheticData(run_params, device=device, samples_per_epoch=1, seed=run_params.seed, split="val")
        test_data = SyntheticData(run_params, device=device, samples_per_epoch=1, seed=run_params.seed, split="test")
    elif run_params.dataset == "abide":
        train_data = ABIDEData(split='train', device=device, args=run_params)
        val_data = ABIDEData(split='val', samples_per_epoch=1, args=run_params, device=device)
        test_data = ABIDEData(split='test', samples_per_epoch=1, args=run_params, device=device)
    else:
        raise Exception("Dataset %s not supported" % run_params.dataset)
        
    train_loader = DataLoader(train_data, batch_size=1,num_workers=0)
    val_loader = DataLoader(val_data, batch_size=1)
    test_loader = DataLoader(test_data, batch_size=1)

    class MyDataModule(pl.LightningDataModule):
        def setup(self,stage=None):
            pass
        def train_dataloader(self):
            return train_loader
        def val_dataloader(self):
            return val_loader
        def test_dataloader(self):
            return test_loader
    
    # configure input feature size
    if run_params.pre_fc is None or len(run_params.pre_fc)==0: 
        if len(run_params.dgm_layers[0])>0:
            run_params.dgm_layers[0][0]=train_data.n_features
        run_params.conv_layers[0][0]=train_data.n_features
    else:
        run_params.pre_fc[0]=train_data.n_features
    run_params.fc_layers[-1] = train_data.num_classes
    
    # specify model
    if run_params.model == "DGM":
        model = DGM_Model(run_params)
    elif run_params.model == "cDGM":
        model = cDGM_Model(run_params)
    elif run_params.model == "static":
        model = PatientPopulationModel(run_params, train_data.X.shape[-1], train_data.num_classes)
    else:
        raise Exception("Model %s not supported" % run_params.model)

    # define callbacks
    checkpoint_callback = ModelCheckpoint(
            save_last=False,
            save_top_k=1,
            verbose=False,
            monitor='val_loss',
            mode='min'
        )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=20,
        verbose=False,
        mode='min')
    
    if run_params.dataset == "ukbbbrain":
        callbacks = [checkpoint_callback]
    else:
        callbacks = [checkpoint_callback , early_stop_callback]

    if val_data==test_data:
        callbacks = None
    
    # defain logging and trainer
    wandb_logger = WandbLogger(project="dgm_experiments", entity="ukbbgnns", name="dgm_%s_pg%s_%s" % (run_params.dataset, run_params.provided_graph, run_params.ffun))
    trainer = pl.Trainer.from_argparse_args(run_params,logger=wandb_logger, callbacks=callbacks)
    
    # train and test
    trainer.fit(model, datamodule=MyDataModule())
    trainer.test(ckpt_path=checkpoint_callback.best_model_path)

    
if __name__ == "__main__":

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    # parameters for DGM training
    params = parser.parse_args(['--gpus','1',                         
                              '--log_every_n_steps','100',                          
                              '--max_epochs','200',
                              '--progress_bar_refresh_rate','10',                         
                              '--check_val_every_n_epoch','1'])
    parser.add_argument("--num_gpus", default=1, type=int)
    parser.add_argument("--fold", default='0', type=int)
    parser.add_argument("--conv_layers", default=[[32,32],[32,16],[16,8]], type=lambda x :eval(x))
    parser.add_argument("--dgm_layers", default= [[32,16,4],[],[]], type=lambda x :eval(x))
    parser.add_argument("--fc_layers", default=[8,8,3], type=lambda x :eval(x))
    parser.add_argument("--pre_fc", default=[-1,32], type=lambda x :eval(x))
    parser.add_argument("--gfun", default='gcn')
    parser.add_argument("--ffun", default='gcn')
    parser.add_argument("--pooling", default='add')
    parser.add_argument("--distance", default='euclidean')
    
    # general parameters specifying the training process, dataset, model, task, etc.
    parser.add_argument("--dataset", default='synthetic', type=str)
    parser.add_argument("--model", default="DGM", type=str)
    parser.add_argument("--provided_graph", default=0, type=int, help="set to 0 if for Cora, the provided graph shall NOT be used")
    parser.add_argument("--seed", default=42, type=int, help="set seed for reproducibility")
    parser.add_argument("--classification", default=1, type=int, help="set to 1 if it's a classification task, otherwise regression")
    parser.add_argument("--assess_graph", default=0, type=int, help="set 1 to assess graph")
    parser.add_argument("--device", default=0, type=int, help="define the device to use")
    parser.add_argument("--train_set_size", type=float, default=0.8)
    parser.add_argument("--self_loops", default=0, type=int, help="define whether the created graph should get added self loops or not")
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--lr", default=1e-2, type=float)

    # parameters for random graph construction
    parser.add_argument("--edge_prob", default=0.001, type=float, help="set edge probability for Erdos Renyi graph, if required")
    parser.add_argument("--random_adj_matrix", default=3, type=int)
    # parameter for synthetic graph construction
    parser.add_argument("--percentage_same_label_neighbours", default=0.9, type=float, help="define whether the node attributes should be used or not")
    # parameters for static graph construction
    parser.add_argument("--hidden_channels", default=32, type=int, help="define the number of hidden channels")
    parser.add_argument("--num_layers", default=3, type=int, help="define the number of layers")
    parser.add_argument("--model_type", default="GCN", type=str, help="define the model type for static graph construction")
    # parameters for graph construction
    parser.add_argument("--node_features", default='all', type=str, help="define which features to use as node features: all, imaging, phenotypes")
    parser.add_argument("--edges", default='all', type=str, help="define which features to use for edge construction: all, imaging, phenotypes")
    parser.add_argument("--k", default=5, type=int) 
    parser.add_argument("--method_sim_matrix", type=str, default="knn", help="use one of: cdist, cosine_sim")

    # parameters for synthetic dataset
    parser.add_argument("--nr_nodes", type=int, default=100)
    parser.add_argument("--nr_features", type=int, default=50)
    parser.add_argument("--nr_informative", type=int, default=10)
    parser.add_argument("--nr_classes", type=int, default=2)

    parser.set_defaults(default_root_path='./log')
    params = parser.parse_args(namespace=params)

    cpu_generator, gpu_generator = utils.make_deterministic(params.seed)
    torch_geometric.seed_everything(params.seed)
    
    run_training_process(params)
