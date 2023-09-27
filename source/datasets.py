import torch
import pickle
import numpy as np
import os.path as osp
import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch_geometric
from torch_cluster import knn_graph
from torch_geometric.data import Data
from torch_geometric.transforms import KNNGraph
from typing import Tuple, List, Union, Dict
import os
import sys
sys.path.append("../source/")
from utils import utils


class ABIDEData(torch.utils.data.Dataset):
    def __init__(self, args, split, samples_per_epoch=100, device="cpu", seed=42) -> None:
        abide_path = 'data/Abide/parisot_features.csv'

        x_data, y = self.read_nodes(abide_path)
        x_data = torch.tensor(x_data.values)
        y = torch.tensor(y)
        
        train_mask, test_mask, val_mask = utils.create_train_test_val_masks_levelled(y, test_size=((1-0.7)/2), validation_size=((1-0.7)/2), random_state=42)
        self.y = torch.eye(2, device=y.device)[y.long()]
        self.X = torch.tensor(x_data, dtype=torch.float32).to(device)

        if split=="train":
            self.mask = torch.from_numpy(train_mask).to(device)
        elif split=="test":
            self.mask = torch.from_numpy(test_mask).to(device)
        else:
            self.mask = torch.from_numpy(val_mask).to(device)

        self.samples_per_epoch = samples_per_epoch
        self.num_classes = 2
        self.n_features = self.X.shape[-1]

        if args.provided_graph == 0:
            print("Starting without a graph")
            self.edge_index = torch.tensor([[0],[0]]).to(device)
        elif args.provided_graph == 1:
            print("Using self loops only")
            self.edge_index = torch_geometric.utils.add_self_loops(torch.tensor([[0],[0]]), num_nodes=self.X.shape[0])[0].to(device)
        elif args.provided_graph == 2:
            print("using random graph to start")
            self.edge_index = torch_geometric.utils.erdos_renyi_graph(self.X.shape[0], edge_prob=args.edge_prob).to(device)
        elif args.provided_graph == 3:
            print("using graph with random n neighbours")
            edges0 = torch.randint(self.X.shape[0], (self.X.shape[0]*args.k,)).tolist()
            edges1_listoflists =[[i]*args.k for i in range(self.X.shape[0])]
            edges1 = [item for sublist in edges1_listoflists for item in sublist]
            self.edge_index = torch.tensor([edges0,edges1]).to(device)
        elif args.provided_graph == 5:
            print("use Euclidean kNN graph")
            self.edge_index = torch_geometric.nn.knn_graph(self.X, k=args.k, loop=True, flow='source_to_target').to(device)
        elif args.provided_graph == 6:
            print("using cosine sim KNN graph")
            self.edge_index = torch_geometric.nn.knn_graph(self.X.to(device), k=args.k, loop=True, flow='source_to_target', cosine=True).to(device)
        elif args.provided_graph == 10:
            print(f"Caution, synthetically generating graph with {args.percentage_same_label_neighbours} same label neighbours")
            nr_same_labelled_neighbours = int(args.percentage_same_label_neighbours*args.k)
            nr_diff_labelled_neighbours = int(args.k - nr_same_labelled_neighbours)
            self.edge_index = utils.get_edges_based_on_ground_truth(
                self.X.shape[-2], self.y.argmax(dim=1), nr_same_label_neighbours=nr_same_labelled_neighbours,
                nr_different_label_neighbours=nr_diff_labelled_neighbours)

    def __len__(self):
        return self.samples_per_epoch

    def read_nodes(self, node_dataset_path: str) -> Tuple[pd.DataFrame, np.ndarray]:
        def read_fn(path: str) -> pd.DataFrame:
            return pd.read_csv(path, header=None, sep='\t')

        def columns_fn(df: pd.DataFrame) -> Dict[str, Union[str, List[str]]]:
            categorical_columns = None
            continuous_columns = range(len(df.columns) - 1)
            node_feature_columns = continuous_columns
            label_column = len(df.columns) - 1

            columns = {
                'categorical': categorical_columns,
                'continuous': continuous_columns,
                'features': node_feature_columns,
                'label': label_column
            }
            return columns

        def processing_fn(
                    df: pd.DataFrame,
                    columns: Dict[str, Union[str, List[str]]]
                ) -> pd.DataFrame:
            # Scale the data
            scaler = StandardScaler()
            df[ list(columns['features']) ] = scaler.fit_transform( df[ list(columns['features']) ] )
            return df

        df = read_fn(node_dataset_path)
        columns = columns_fn(df)
        df = processing_fn(df, columns)

        node_features_df = df[columns['features']]
        labels = df[columns['label']].to_numpy(dtype=np.int8)
        return node_features_df, labels
    
    def __getitem__(self, idx):
        return self.X, self.y, self.mask, self.edge_index
    

class TadpoleDataset(torch.utils.data.Dataset):
    """Dataest class adapted from Kazi et al. 2022."""

    def __init__(self, args, mask="train", samples_per_epoch=100, device='cpu',full=False):
        with open('data/Tadpole/tadpole_data.pickle', 'rb') as f:
            X_,y_,train_mask_,test_mask_, weight_ = pickle.load(f) # Load the data

        if not full:
            X_ = X_[...,:30,:] # For DGM we use modality 1 (M1) for both node representation and graph learning.
        
        self.n_features = X_.shape[-2]
        self.num_classes = y_.shape[-2]
        
        self.X = torch.from_numpy(X_[:,:,args.fold]).float().to(device)
        self.y = torch.from_numpy(y_[:,:,args.fold]).float().to(device)
        self.weight = torch.from_numpy(np.squeeze(weight_[:1,args.fold])).float().to(device)
        if mask=="train":
            self.mask = torch.from_numpy(np.concatenate((train_mask_[:,args.fold][:-50], np.array([0]*50)))).to(device)
        elif mask=="test":
            self.mask = torch.from_numpy(test_mask_[:,args.fold]).to(device)
        else:
            self.mask = torch.from_numpy(np.concatenate((np.array([0]*514), train_mask_[:,args.fold][-50:]))).to(device)

        if args.provided_graph == 0:
            print("Not using provided graph")
            self.edge_index = torch.tensor([[0],[0]]).to(device)
        elif args.provided_graph == 1:
            print("using self loops only")
            self.edge_index = torch_geometric.utils.add_self_loops(torch.tensor([[0],[0]]), num_nodes=self.X.shape[0])[0].to(device)
        elif args.provided_graph == 2:
            print("using random graph to start")
            self.edge_index = torch_geometric.utils.erdos_renyi_graph(self.X.shape[0], edge_prob=args.edge_prob).to(device)
        elif args.provided_graph == 3:
            print("using graph with random n neighbours")
            edges0 = torch.randint(self.X.shape[0], (self.X.shape[0]*args.k,)).tolist()
            edges1_listoflists =[[i]*args.k for i in range(self.X.shape[0])]
            edges1 = [item for sublist in edges1_listoflists for item in sublist]
            self.edge_index = torch.tensor([edges0,edges1]).to(device)
        elif args.provided_graph == 5:
            print("use Euclidean kNN graph")
            self.edge_index = torch_geometric.nn.knn_graph(self.X, k=args.k, loop=True, flow='source_to_target').to(device)
        elif args.provided_graph == 6:
            print("using cosine sim KNN graph")
            self.edge_index = torch_geometric.nn.knn_graph(self.X.to(device), k=args.k, loop=True, flow='source_to_target', cosine=True).to(device)
        elif args.provided_graph == 10:
            print(f"Caution, synthetically generating graph with {args.percentage_same_label_neighbours} same label neighbours")
            nr_same_labelled_neighbours = int(args.percentage_same_label_neighbours*args.k)
            nr_diff_labelled_neighbours = int(args.k - nr_same_labelled_neighbours)
            self.edge_index = utils.get_edges_based_on_ground_truth(
                self.X.shape[-2], self.y.argmax(dim=1), nr_same_label_neighbours=nr_same_labelled_neighbours,
                nr_different_label_neighbours=nr_diff_labelled_neighbours)
        else:
            raise ValueError("provided graph must be 0,1,2,3, or 5")
            
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        return self.X, self.y, self.mask, self.edge_index


def get_planetoid_dataset(name, normalize_features=True, transform=None, split="complete"):
    path = osp.join('.', 'data', name)
    if split == 'complete':
        dataset = Planetoid(path, name)
        dataset[0].train_mask.fill_(False)
        dataset[0].train_mask[:dataset[0].num_nodes - 1000] = 1
        dataset[0].val_mask.fill_(False)
        dataset[0].val_mask[dataset[0].num_nodes - 1000:dataset[0].num_nodes - 500] = 1
        dataset[0].test_mask.fill_(False)
        dataset[0].test_mask[dataset[0].num_nodes - 500:] = 1
    else:
        dataset = Planetoid(path, name, split=split)
    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform
    return dataset

def one_hot_embedding(labels, num_classes):
    y = torch.eye(num_classes) 
    return y[labels] 


class PlanetoidDataset(torch.utils.data.Dataset):
    def __init__(self, args, split='train', samples_per_epoch=100, name='Cora', device='cpu'):
        dataset = get_planetoid_dataset(name)
        self.X = dataset[0].x.float().to(device)
        self.y = one_hot_embedding(dataset[0].y,dataset.num_classes).float().to(device)
        if args.provided_graph == 0:
            print("Not using provided graph")
            self.edge_index = torch.tensor([[0],[0]]).to(device)
        elif args.provided_graph == 1:
            print("using self loops only")
            self.edge_index = torch_geometric.utils.add_self_loops(torch.tensor([[0],[0]]),num_nodes=self.X.shape[0])[0].to(device)
        elif args.provided_graph == 2:
            print("using random graph to start")
            self.edge_index = torch_geometric.utils.erdos_renyi_graph(self.X.shape[0], edge_prob=args.edge_prob).to(device)
        elif args.provided_graph == 3:
            print("using graph with random n neighbours")
            edges0 = torch.randint(self.X.shape[0], (self.X.shape[0]*args.k,)).tolist()
            edges1_listoflists =[[i]*args.k for i in range(self.X.shape[0])]
            edges1 = [item for sublist in edges1_listoflists for item in sublist]
            self.edge_index = torch.tensor([edges0,edges1]).to(device)
        elif args.provided_graph == 4:
            print("using provided graph")
            self.edge_index = dataset[0].edge_index.to(device)
        elif args.provided_graph == 5:
            print("use Euclidean kNN graph")
            self.edge_index = torch_geometric.nn.knn_graph(self.X, k=args.k, loop=True, flow='source_to_target').to(device)
        elif args.provided_graph == 6:
            print("using cosine sim KNN graph")
            self.edge_index = torch_geometric.nn.knn_graph(self.X.to(device), k=args.k, loop=True, flow='source_to_target', cosine=True).to(device)
        elif args.provided_graph == 10:
            print(f"Caution, synthetically generating graph with {args.percentage_same_label_neighbours} same label neighbours")
            nr_same_labelled_neighbours = int(args.percentage_same_label_neighbours*args.k)
            nr_diff_labelled_neighbours = int(args.k - nr_same_labelled_neighbours)
            self.edge_index = utils.get_edges_based_on_ground_truth(
                self.X.shape[-2], self.y.argmax(dim=1), nr_same_label_neighbours=nr_same_labelled_neighbours,
                nr_different_label_neighbours=nr_diff_labelled_neighbours)
        else:
            raise ValueError("provided graph must be 0,1,2,3,4, or 5")
        
        self.n_features = dataset[0].num_node_features
        self.num_classes = dataset.num_classes
        
        if split=='train':
            self.mask = dataset[0].train_mask.to(device)
        if split=='val':
            self.mask = dataset[0].val_mask.to(device)
        if split=='test':
            self.mask = dataset[0].test_mask.to(device)
         
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        return self.X, self.y, self.mask, self.edge_index


class MyData(torch.utils.data.Dataset):
    def __init__(self, args, device, samples_per_epoch=1, seed=42,split="train") -> None:
        self.data_str = args.dataset
        self.args = args
        if self.data_str == "synthetic":
            self.data = SyntheticData(args, device, samples_per_epoch, seed, split)
        
    def get_train_test_val_masks(self):
        train_ind, test_ind, val_ind, train_mask, test_mask, val_mask = utils.get_train_test_val_split(
            self.y, train_size=self.args.train_set_size, seed=self.args.seed)
        return train_ind, test_ind, val_ind, train_mask, test_mask, val_mask

    def get_node_features_transductive(self):
        node_features = self.X
        imputed_df = self.X
        node_features = torch.tensor(node_features.values).float()
        node_features = torch.nn.functional.normalize(node_features, p=2.0, dim=0)

        return node_features, imputed_df

    def create_adj_matrix_transductive(self):
        if self.args.random_adj_matrix == 1:
            # create a random adjacency matrix
            print("using random adj matrix")
            edges = torch_geometric.utils.erdos_renyi_graph(self.nr_nodes, edge_prob=self.args.edge_prob)
            if self.args.self_loops == 1:
                edges, _ = torch_geometric.utils.add_remaining_self_loops(edges)
        elif self.args.random_adj_matrix == 4:
            # starting from only self loops
            edges = torch.tensor([[],[]])
            edges = torch_geometric.utils.add_remaining_self_loops(edges, num_nodes=self.data_x.shape[0])[0].long()
        elif self.args.random_adj_matrix == 2:
            # construct adj matrix with specific parameters
            print("setting up adj matrix")
            nr_same_labelled_neighbours = int(self.args.percentage_same_label_neighbours*self.args.k)
            nr_diff_labelled_neighbours = int(self.args.k - nr_same_labelled_neighbours)
            edges = utils.get_edges_based_on_ground_truth(
                self.node_features.shape[0], self.data_y, nr_same_label_neighbours=nr_same_labelled_neighbours, 
                nr_different_label_neighbours=nr_diff_labelled_neighbours)
            if self.args.self_loops == 1:
                edges, _ = torch_geometric.utils.add_remaining_self_loops(edges)
        else:
            print("using graph creation methods")
            if self.args.method_sim_matrix == "cdist" or self.args.method_sim_matrix == "knn":
                knn_self_loop = True if self.args.self_loops else False
                # get similarity matrix with imputed values
                if self.args.method_sim_matrix == "cdist":
                    edges = knn_graph(torch.tensor(self.imputed_df[self.cols_edges].values), self.args.k, loop=knn_self_loop, cosine=True)
                else:
                    edges = knn_graph(torch.tensor(self.imputed_df[self.cols_edges].values), self.args.k, loop=knn_self_loop, cosine=False)
                if self.args.self_loops == 1:
                    edges, _ = torch_geometric.utils.add_remaining_self_loops(edges)    

        return edges

    def __len__(self):
        return self.samples_per_epoch
    
    def __getitem__(self, idx):
        return self.X, self.y, self.mask, self.edges


class SyntheticData(MyData):
    def __init__(self, args, device, samples_per_epoch, seed, split) -> None:
        self.n_features = args.nr_features
        self.nr_nodes = args.nr_nodes
        if args.classification == 1:
            self.num_classes = args.nr_classes
        else:
            self.num_classes = 1
        self.nr_informative = args.nr_informative
        self.seed = seed
        self.args = args
        self.device = device
        self.samples_per_epoch = samples_per_epoch
        self.classification = args.classification

        if args.classification ==1 :
            self.X, self.y, self.full_data, self.useable_columns = self.get_classification_data()
        else:
            self.X, self.y, self.full_data, self.useable_columns = self.get_regression_data()
        self.train_ind, self.test_ind, self.val_ind, self.train_mask, self.test_mask, self.val_mask = self.get_train_test_val_masks()
        
        if split == "train":
            self.mask = torch.tensor(self.train_mask)
        elif split == "test":
            self.mask = torch.tensor(self.test_mask)
        elif split == "val":
            self.mask = torch.tensor(self.val_mask)

        self.informative_features = utils.get_correlation_of_dataframe(self.full_data, "label")[:self.nr_informative+1].index.tolist()[1:]
        self.non_informative_features = list(set(self.useable_columns) - set(self.informative_features))
        self.cols_edges, self.cols_nodes = self.useable_columns, self.useable_columns
        self.node_features, self.imputed_df = self.get_node_features_transductive()

        self.edges = self.create_adj_matrix_transductive()
        self.X = self.node_features
        if args.classification == 1:
            self.y = torch.eye(self.num_classes, device=device)[self.y] # shape: (nr_nodes, nr_classes)
        else:
            self.y = torch.nn.functional.normalize(self.y, p=2.0, dim = 0).unsqueeze(-1)

    def get_classification_data(self):
        useable_columns = [str(i) for i in range(self.n_features)]
        full_data, _, data_y = utils.generate_synthetic_population_dataset(
            self.nr_nodes, self.num_classes, self.n_features, self.nr_informative, self.seed)
        data_x = full_data[useable_columns]

        return data_x, data_y, full_data, useable_columns

    def get_regression_data(self):
        useable_columns = [str(i) for i in range(self.n_features)]
        full_data, _, data_y = utils.generate_synthetic_population_dataset_regression(
            self.nr_nodes, self.num_classes, self.n_features, self.nr_informative, self.seed)
        data_x = full_data[useable_columns]

        return data_x, data_y, full_data, useable_columns
    

class PopulationGraphUKBB:
    def __init__(self, data_dir, filename_train, filename_val, filename_test, phenotype_columns, columns_kept, num_node_features, task, num_classes, k, edges):
        self.data_dir = data_dir
        self.filename_train = filename_train
        self.filename_val = filename_val
        self.filename_test = filename_test
        self.phenotype_columns = phenotype_columns
        self.columns_kept = columns_kept
        self.num_node_features = num_node_features
        self.num_classes = num_classes
        self.k = k
        self.edges = edges
        self.task = task

    def load_data(self):
        """
        Loads the dataframes for the train, val, and test, and returns 1 dataframe for all.
        """

        # Read csvs for tran, val, test
        data_df_train = pd.read_csv(self.data_dir + self.filename_train)
        data_df_val = pd.read_csv(self.data_dir+self.filename_val)
        data_df_test = pd.read_csv(self.data_dir+self.filename_test)
        
        # Give labels for classification 
        if self.task == 'classification':        
                frames = [data_df_train, data_df_val, data_df_test]
                df = pd.concat(frames)

                labels = list(range(0,self.num_classes))
                df['Age'] = pd.qcut(df['Age'], q=self.num_classes, labels=labels).astype('int') #Balanced classes
                # df['Age'] = pd.cut(df['Age'], bins=self.num_classes, labels=labels).astype('int') #Not balanced classes
                
                a = data_df_train.shape[0]
                b = data_df_val.shape[0]

                data_df_train = df.iloc[:a, :]
                data_df_val = df.iloc[a:a+b, :]
                data_df_test = df.iloc[a+b:, :]
        
        a = data_df_train.shape[0] 
        b = data_df_train.shape[0]+data_df_val.shape[0] 
        num_nodes = b + data_df_test.shape[0] 

        train_idx = np.arange(0, a, dtype=int)
        val_idx = np.arange(a, b, dtype=int)
        test_idx = np.arange(b, num_nodes, dtype=int)
        frames = [data_df_train, data_df_val, data_df_test] 

        data_df = pd.concat(frames, ignore_index=True)

        return data_df, train_idx, val_idx, test_idx, num_nodes

    def get_phenotypes(self, data_df):
        """
        Takes the dataframe for the train, val, and test, and returns 1 dataframe with only the phenotypes.
        """
        phenotypes_df = data_df[self.phenotype_columns]      

        return phenotypes_df  

    def get_features_demographics(self, phenotypes_df):
        """
        Returns the phenotypes of every node, meaning for every subject. 
        The node features are defined by the non-imaging information
        """
        phenotypes = phenotypes_df.to_numpy()
        phenotypes = torch.from_numpy(phenotypes).float()
        return phenotypes

    def get_node_features(self, data_df):
        """
        Returns the features of every node, meaning for every subject.
        """
        # df_node_features = data_df.iloc[:, 2:]
        df_node_features = data_df.iloc[:, 22:]
        node_features = df_node_features.to_numpy()
        node_features = torch.from_numpy(node_features).float()
        return node_features

    def get_subject_masks(self, train_index, validate_index, test_index):
        """Returns the boolean masks for the arrays of integer indices.

        inputs:
        train_index: indices of subjects in the train set.
        validate_index: indices of subjects in the validation set.
        test_index: indices of subjects in the test set.

        returns:
        a tuple of boolean masks corresponding to the train/validate/test set indices.
        """

        num_subjects = len(train_index) + len(validate_index) + len(test_index)

        train_mask = np.zeros(num_subjects, dtype=bool)
        train_mask[train_index] = True
        train_mask = torch.from_numpy(train_mask)

        validate_mask = np.zeros(num_subjects, dtype=bool)
        validate_mask[validate_index] = True
        validate_mask = torch.from_numpy(validate_mask)

        test_mask = np.zeros(num_subjects, dtype=bool)
        test_mask[test_index] = True
        test_mask = torch.from_numpy(test_mask)

        return train_mask, validate_mask, test_mask

    def get_labels(self, data_df):
        """
        Returns the labels for every node, in our case, age.

        """
        if self.task == 'regression':
            labels = data_df['Age'].values   
            labels = torch.from_numpy(labels).float()
        elif self.task == 'classification':
            labels = data_df['Age'].values 
            print(np.unique(labels, return_counts=True))
            labels = torch.from_numpy(labels)
        else:
            raise ValueError('Task should be either regression or classification.')
        return labels
                        
    def get_edges_using_KNNgraph(self, dataset, k):
        """
        Extracts edge index based on the cosine similarity of the node features.

        Inputs:
        dataset: the population graph (without edge_index).
        k: number of edges that will be kept for every node.

        Returns: 
        dataset: graph dataset with the acquired edges.
        """
        
        if self.edges == 'phenotypes':
            # Edges extracted based on the similarity of the selected phenotypes (imaging+non imaging)
            dataset.pos = dataset.phenotypes   
        elif self.edges == 'imaging':
            # Edges extracted based on the similarity of the node features
            dataset.pos = dataset.x  
        elif self.edges == 'all':
            dataset.pos = torch.cat((dataset.x, dataset.phenotypes), dim=1)
        else:
            raise ValueError('Choose appropriate edge connection.')

        dataset.cuda()
        dataset = KNNGraph(k=k, force_undirected=True)(dataset)
        dataset.to('cpu')
        dataset = Data(x = dataset.x, y = dataset.y, phenotypes = dataset.phenotypes, train_mask=dataset.train_mask, 
                        val_mask= dataset.val_mask, test_mask=dataset.test_mask, edge_index=dataset.edge_index, 
                        num_nodes=dataset.num_nodes)
        return dataset
    
    def get_population_graph(self):
        """
        Creates the population graph.
        """
        # Load data
        data_df, train_idx, val_idx, test_idx, num_nodes = self.load_data()

        # Take phenotypes and node_features dataframes
        phenotypes_df = self.get_phenotypes(data_df)
        phenotypes = self.get_features_demographics(phenotypes_df)
        node_features = self.get_node_features(data_df) 

        # Mask val & test subjects
        train_mask, val_mask, test_mask = self.get_subject_masks(train_idx, val_idx, test_idx)
        # Get the labels
        labels = self.get_labels(data_df) 

        if  self.task == 'classification':
            labels= one_hot_embedding(labels,abs(self.num_classes)) 

        population_graph = Data(x = node_features, y= labels, phenotypes= phenotypes, train_mask= train_mask, val_mask=val_mask, test_mask=test_mask, num_nodes=num_nodes, k=self.k)        
        # Get edges using existing pyg KNNGraph class
        population_graph = self.get_edges_using_KNNgraph(population_graph, k=self.k)
        return population_graph


def one_hot_embedding(labels, num_classes):
    y = torch.eye(num_classes) 
    return y[labels] 


