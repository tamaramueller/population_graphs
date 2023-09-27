# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
import os.path as osp
import torch_geometric.transforms as T

from typing import Optional, Callable, List, Union
from torch_sparse import SparseTensor, coalesce
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.utils.undirected import to_undirected
from torch_geometric.utils import remove_self_loops
from nsd_utils.classic import Planetoid
from definitions import ROOT_DIR
import pickle
import torch_geometric
import sys
from typing import Callable, Tuple, List, Union, Optional, Dict
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

sys.path.append(osp.join(ROOT_DIR, "../"))
from utils import utils


def create_train_test_val_masks_levelled(labels, test_size=0.2, validation_size=0.1, random_state=None):
    """
    Create training, test, and validation masks for equally distributed label classes.

    Parameters:
    - labels: List of binary labels (0 or 1).
    - test_size: Fraction of the data to include in the test split (default is 0.2).
    - validation_size: Fraction of the data to include in the validation split (default is 0.1).
    - random_state: Seed for random number generation (optional).

    Returns:
    - train_mask: Boolean mask for the training set.
    - test_mask: Boolean mask for the test set.
    - validation_mask: Boolean mask for the validation set.
    """
    # Ensure labels contain only 0s and 1s
    if not all(label in [0, 1] for label in labels):
        raise ValueError("Labels must be binary (0 or 1).")

    # Split the data into positive and negative classes
    positive_indices = np.where(np.array(labels) == 1)[0]
    negative_indices = np.where(np.array(labels) == 0)[0]

    # Split positive and negative examples into train, test, and validation sets
    pos_train, pos_test = train_test_split(positive_indices, test_size=(test_size+validation_size), random_state=random_state)
    neg_train, neg_test = train_test_split(negative_indices, test_size=(test_size+validation_size), random_state=random_state)
    
    # Further split the test sets into test and validation
    pos_test, pos_val = train_test_split(pos_test, test_size=validation_size, random_state=random_state)
    neg_test, neg_val = train_test_split(neg_test, test_size=validation_size, random_state=random_state)

    # Create masks
    train_mask = np.zeros(len(labels), dtype=bool)
    test_mask = np.zeros(len(labels), dtype=bool)
    val_mask = np.zeros(len(labels), dtype=bool)

    train_mask[pos_train] = 1
    train_mask[neg_train] = 1

    test_mask[pos_test] = 1
    test_mask[neg_test] = 1

    val_mask[pos_val] = 1
    val_mask[neg_val] = 1

    return train_mask, test_mask, val_mask


class Actor(InMemoryDataset):
    r"""
    Code adapted from https://github.com/pyg-team/pytorch_geometric/blob/2.0.4/torch_geometric/datasets/actor.py

    The actor-only induced subgraph of the film-director-actor-writer
    network used in the
    `"Geom-GCN: Geometric Graph Convolutional Networks"
    <https://openreview.net/forum?id=S1e2agrFvS>`_ paper.
    Each node corresponds to an actor, and the edge between two nodes denotes
    co-occurrence on the same Wikipedia page.
    Node features correspond to some keywords in the Wikipedia pages.
    The task is to classify the nodes into five categories in term of words of
    actor's Wikipedia.

    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = 'https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/master'

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt'
                ] + [f'film_split_0.6_0.2_{i}.npz' for i in range(10)]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        for f in self.raw_file_names[:2]:
            download_url(f'{self.url}/new_data/film/{f}', self.raw_dir)
        for f in self.raw_file_names[2:]:
            download_url(f'{self.url}/splits/{f}', self.raw_dir)

    def process(self):

        with open(self.raw_paths[0], 'r') as f:
            data = [x.split('\t') for x in f.read().split('\n')[1:-1]]

            rows, cols = [], []
            for n_id, col, _ in data:
                col = [int(x) for x in col.split(',')]
                rows += [int(n_id)] * len(col)
                cols += col
            x = SparseTensor(row=torch.tensor(rows), col=torch.tensor(cols))
            x = x.to_dense()

            y = torch.empty(len(data), dtype=torch.long)
            for n_id, _, label in data:
                y[int(n_id)] = int(label)

        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            # Remove self-loops
            edge_index, _ = remove_self_loops(edge_index)
            # Make the graph undirected
            edge_index = to_undirected(edge_index)
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        train_masks, val_masks, test_masks = [], [], []
        for f in self.raw_paths[2:]:
            tmp = np.load(f)
            train_masks += [torch.from_numpy(tmp['train_mask']).to(torch.bool)]
            val_masks += [torch.from_numpy(tmp['val_mask']).to(torch.bool)]
            test_masks += [torch.from_numpy(tmp['test_mask']).to(torch.bool)]
        train_mask = torch.stack(train_masks, dim=1)
        val_mask = torch.stack(val_masks, dim=1)
        test_mask = torch.stack(test_masks, dim=1)

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])


class WikipediaNetwork(InMemoryDataset):
    r"""
    Code adapted from https://github.com/pyg-team/pytorch_geometric/blob/2.0.4/torch_geometric/datasets/wikipedia_network.py

    The Wikipedia networks introduced in the
    `"Multi-scale Attributed Node Embedding"
    <https://arxiv.org/abs/1909.13021>`_ paper.
    Nodes represent web pages and edges represent hyperlinks between them.
    Node features represent several informative nouns in the Wikipedia pages.
    The task is to predict the average daily traffic of the web page.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"chameleon"`,
            :obj:`"crocodile"`, :obj:`"squirrel"`).
        geom_gcn_preprocess (bool): If set to :obj:`True`, will load the
            pre-processed data as introduced in the `"Geom-GCN: Geometric
            Graph Convolutional Networks" <https://arxiv.org/abs/2002.05287>_`,
            in which the average monthly traffic of the web page is converted
            into five categories to predict.
            If set to :obj:`True`, the dataset :obj:`"crocodile"` is not
            available.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)

    """

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name.lower()
        assert self.name in ['chameleon', 'squirrel']
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> Union[str, List[str]]:
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')[1:-1]
        x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
        x = torch.tensor(x, dtype=torch.float)
        y = [int(r.split('\t')[2]) for r in data]
        y = torch.tensor(y, dtype=torch.long)

        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
        edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
        # Remove self-loops
        edge_index, _ = remove_self_loops(edge_index)
        # Make the graph undirected
        edge_index = to_undirected(edge_index)
        edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        data = Data(x=x, edge_index=edge_index, y=y)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])


class WebKB(InMemoryDataset):
    r"""
    Code adapted from https://github.com/pyg-team/pytorch_geometric/blob/2.0.4/torch_geometric/datasets/webkb.py

    The WebKB datasets used in the
    `"Geom-GCN: Geometric Graph Convolutional Networks"
    <https://openreview.net/forum?id=S1e2agrFvS>`_ paper.
    Nodes represent web pages and edges represent hyperlinks between them.
    Node features are the bag-of-words representation of web pages.
    The task is to classify the nodes into one of the five categories, student,
    project, course, staff, and faculty.
    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Cornell"`,
            :obj:`"Texas"` :obj:`"Washington"`, :obj:`"Wisconsin"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/'
           '1c4c04f93fa6ada91976cda8d7577eec0e3e5cce/new_data')

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in ['cornell', 'texas', 'washington', 'wisconsin']

        super(WebKB, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        for name in self.raw_file_names:
            download_url(f'{self.url}/{self.name}/{name}', self.raw_dir)

    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')[1:-1]
            x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
            x = torch.tensor(x, dtype=torch.float32)

            y = [int(r.split('\t')[2]) for r in data]
            y = torch.tensor(y, dtype=torch.long)

        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            edge_index = to_undirected(edge_index)
            # We also remove self-loops in these datasets in order not to mess up with the Laplacian.
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


class Tadpole(InMemoryDataset):
    r"""
        loading the TADPOLE dataset
    """

    def __init__(self, root, name, transform=None, pre_transform=None, provided_graph=5):
        self.name = name.lower()
        self.provided_graph = provided_graph
        super(Tadpole, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return 'tadpole_data.pt'

    # def download(self):
    #     for name in self.raw_file_names:
    #         download_url(f'{self.url}/{self.name}/{name}', self.raw_dir)

    def process(self):
        with open('/vol/aimspace/users/muel/DGM_pytorch/data/tadpole_data.pickle', 'rb') as f:
            X_,y_,train_mask_,test_mask_, weight_ = pickle.load(f) # Load the data
        X_ = X_[...,:30,:] # Only use the first 30 features
        # n_features = .
        
        fold = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        k = 10
        edge_prob = 0.1

        X = torch.from_numpy(X_[:,:,fold]).float().to(device)
        y = torch.from_numpy(y_[:,:,fold]).float().argmax(dim=1).to(device)

        train_mask = torch.tensor(np.concatenate((train_mask_[:,fold][:-50], np.array([0]*50))), dtype=torch.bool).to(device)
        test_mask = torch.tensor(test_mask_[:,fold], dtype=torch.bool).to(device)
        val_mask = torch.tensor(np.concatenate((np.array([0]*514), train_mask_[:,fold][-50:])), dtype=torch.bool).to(device)

        if self.provided_graph == 0:
            print("Not using provided graph")
            edge_index = torch.tensor([[0],[0]]).to(device)
        elif self.provided_graph == 1:
            print("using self loops only")
            edge_index = torch_geometric.utils.add_self_loops(torch.tensor([[0],[0]]), num_nodes=X.shape[0])[0].to(device)
        elif self.provided_graph == 2:
            print("using random graph to start")
            edge_index = torch_geometric.utils.erdos_renyi_graph(X.shape[0], edge_prob=edge_prob).to(device)
        elif self.provided_graph == 3:
            print("using graph with random n neighbours")
            edges0 = torch.randint(X.shape[0], (X.shape[0]*k,)).tolist()
            edges1_listoflists =[[i]*k for i in range(X.shape[0])]
            edges1 = [item for sublist in edges1_listoflists for item in sublist]
            edge_index = torch.tensor([edges0,edges1]).to(device)
        elif self.provided_graph == 5:
            print("use Euclidean kNN graph")
            edge_index = torch_geometric.nn.knn_graph(X, k=k, loop=True, flow='source_to_target').to(device)
            edge_index = to_undirected(edge_index)
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = coalesce(edge_index, None, X.size(0), X.size(0))
        elif self.provided_graph == 6:
            print("using cosine sim KNN graph")
            edge_index = torch_geometric.nn.knn_graph(X.to(device), k=k, loop=True, flow='source_to_target', cosine=True).to(device)
        elif self.provided_graph == 10:
            percentage_same_label_neighbours = 0.9
            print(f"Caution, synthetically generating graph with {percentage_same_label_neighbours} same label neighbours")
            nr_same_labelled_neighbours = int(percentage_same_label_neighbours*k)
            nr_diff_labelled_neighbours = int(k - nr_same_labelled_neighbours)
            edge_index = utils.get_edges_based_on_ground_truth(
                X.shape[-2], y.argmax(dim=1), nr_same_label_neighbours=nr_same_labelled_neighbours,
                nr_different_label_neighbours=nr_diff_labelled_neighbours)

        data = Data(x=X, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        # data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


class Abide(InMemoryDataset):
    r"""
        loading the ABIDE dataset
    """

    def __init__(self, root, name, transform=None, pre_transform=None, provided_graph=5):
        self.name = name.lower()
        self.provided_graph = provided_graph
        super(Abide, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return f'abide_data_{self.provided_graph}.pt'

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


    def process(self):
        abide_path = '/vol/aimspace/users/muel/abide1/parisot_features.csv'
        
        x_data, y = self.read_nodes(abide_path)
        x_data = torch.tensor(x_data.values)
        y = torch.LongTensor(y)
        # print(y)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        k = 10
        edge_prob = 0.1
        percentage_same_label_neighbours = 0.9
        train_mask, test_mask, val_mask = create_train_test_val_masks_levelled(y, test_size=((1-0.7)/2), validation_size=((1-0.7)/2), random_state=42)
        # y = torch.eye(2, device=y.device)[y.long()]
        X = torch.tensor(x_data, dtype=torch.float32).to(device)

        train_mask = torch.from_numpy(train_mask).to(device)
        test_mask = torch.from_numpy(test_mask).to(device)
        val_mask = torch.from_numpy(val_mask).to(device)

        if self.provided_graph == 0:
            print("Not using provided graph")
            edge_index = torch.tensor([[0],[0]]).to(device)
        elif self.provided_graph == 1:
            print("using self loops only")
            edge_index = torch_geometric.utils.add_self_loops(torch.tensor([[0],[0]]), num_nodes=X.shape[0])[0].to(device)
        elif self.provided_graph == 2:
            print("using random graph to start")
            edge_index = torch_geometric.utils.erdos_renyi_graph(X.shape[0], edge_prob=edge_prob).to(device)
        elif self.provided_graph == 3:
            print("using graph with random n neighbours")
            edges0 = torch.randint(X.shape[0], (X.shape[0]*k,)).tolist()
            edges1_listoflists =[[i]*k for i in range(X.shape[0])]
            edges1 = [item for sublist in edges1_listoflists for item in sublist]
            edge_index = torch.tensor([edges0,edges1]).to(device)
        elif self.provided_graph == 5:
            print("use Euclidean kNN graph")
            edge_index = torch_geometric.nn.knn_graph(X, k=k, loop=True, flow='source_to_target').to(device)
            edge_index = to_undirected(edge_index)
            # We also remove self-loops in these datasets in order not to mess up with the Laplacian.
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = coalesce(edge_index, None, X.size(0), X.size(0))
        elif self.provided_graph == 6:
            print("using cosine sim KNN graph")
            edge_index = torch_geometric.nn.knn_graph(X.to(device), k=k, loop=True, flow='source_to_target', cosine=True).to(device)
            edge_index = to_undirected(edge_index)
            # We also remove self-loops in these datasets in order not to mess up with the Laplacian.
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))
        elif self.provided_graph == 10:
            print(f"Caution, synthetically generating graph with {percentage_same_label_neighbours} same label neighbours")
            nr_same_labelled_neighbours = int(percentage_same_label_neighbours*k)
            nr_diff_labelled_neighbours = int(k - nr_same_labelled_neighbours)
            edge_index = utils.get_edges_based_on_ground_truth(
                X.shape[-2], y.argmax(dim=1), nr_same_label_neighbours=nr_same_labelled_neighbours,
                nr_different_label_neighbours=nr_diff_labelled_neighbours)

        data = Data(x=X, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        # data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)
    

class UKBBCardiac(InMemoryDataset):
    r"""
        loading the UKBB Cardiac dataset
    """

    def __init__(self, root, name, transform=None, pre_transform=None, provided_graph=5):
        self.name = name.lower()
        self.provided_graph = provided_graph
        super(UKBBCardiac, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return 'ukbb_cardiac.pt'

    def process(self):
        cardiac_features = pd.read_csv("/vol/aimspace/projects/ukbb/668815/cardiac_features_wenjia_merged.csv")
        clinical_features = pd.read_csv("/vol/aimspace/projects/GNNs/partial_csvs/clinical_features_cardiac_graph_all.csv")
        sex = clinical_features[["sex", "eid"]]
        additional_features = pd.read_csv("/vol/aimspace/projects/ukbb/cardiac/cardiac_segmentations/projects/SelfSuperBio/668815/tabular/cardiac_feature_668815_vector_labeled_noOH.csv")
        additional_features_filtered = additional_features[["eid", "CAD_broad", "Body fat percentage-2.0", "Smoking status-2.0", "Body mass index (BMI)-2.0",  "Frequency of other exercises in last 4 weeks-2.0"]]
        clinical_features = sex.merge(additional_features_filtered, on="eid")

        clinical_and_cardiac = clinical_features.merge(cardiac_features, right_on="eid_87802", left_on="eid")
        clinical_and_cardiac = clinical_and_cardiac.dropna()

        X = clinical_and_cardiac.drop(["eid_x", "eid_y", "eid_60520", "eid_87802", "CAD_broad"], axis=1)
        X = torch.tensor(X.values, dtype=torch.float32)

        y = torch.tensor(clinical_and_cardiac["CAD_broad"].values, dtype=int)
        cad_subjects = X[y==1]
        non_cad_subjects = X[y==0]
        k = 10
        edge_prob = 0.001

        non_cad_subjects = non_cad_subjects[:cad_subjects.shape[0]]

        X = torch.cat((cad_subjects, non_cad_subjects), dim=0)
        y = torch.cat((torch.ones(cad_subjects.shape[0]), torch.zeros(non_cad_subjects.shape[0])), dim=0).long()

        train_mask, test_mask, val_mask = utils.create_train_test_val_masks_levelled(y, test_size=((1-0.8)/2), validation_size=((1-0.8)/2), random_state=42)

        min_max_scaler = preprocessing.MinMaxScaler()
        min_max_scaler.fit(X[train_mask])
        X = torch.tensor(min_max_scaler.transform(X), dtype=torch.float32)

        if self.provided_graph == 0:
            print("Not using provided graph")
            edge_index = torch.tensor([[0],[0]])
        elif self.provided_graph == 1:
            print("using self loops only")
            edge_index = torch_geometric.utils.add_self_loops(torch.tensor([[0],[0]]), num_nodes=X.shape[0])[0]
        elif self.provided_graph == 2:
            print("using random graph to start")
            edge_index = torch_geometric.utils.erdos_renyi_graph(X.shape[0], edge_prob=edge_prob)
        elif self.provided_graph == 3:
            print("using graph with random n neighbours")
            edges0 = torch.randint(X.shape[0], (X.shape[0]*k,)).tolist()
            edges1_listoflists =[[i]*k for i in range(X.shape[0])]
            edges1 = [item for sublist in edges1_listoflists for item in sublist]
            edge_index = torch.tensor([edges0,edges1])
        elif self.provided_graph == 5:
            print("use Euclidean kNN graph")
            edge_index = torch_geometric.nn.knn_graph(X, k=k, loop=True, flow='source_to_target')
            edge_index = to_undirected(edge_index)
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = coalesce(edge_index, None, X.size(0), X.size(0))
        elif self.provided_graph == 6:
            print("using cosine sim KNN graph")
            edge_index = torch_geometric.nn.knn_graph(X, k=k, loop=True, flow='source_to_target', cosine=True)
        elif self.provided_graph == 10:
            percentage_same_label_neighbours = 0.9
            print(f"Caution, synthetically generating graph with {percentage_same_label_neighbours} same label neighbours")
            nr_same_labelled_neighbours = int(percentage_same_label_neighbours*k)
            nr_diff_labelled_neighbours = int(k - nr_same_labelled_neighbours)
            edge_index = utils.get_edges_based_on_ground_truth(
                X.shape[-2], y.argmax(dim=1), nr_same_label_neighbours=nr_same_labelled_neighbours,
                nr_different_label_neighbours=nr_diff_labelled_neighbours)

        data = Data(x=X, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        # data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)


class Covid(InMemoryDataset):
    r"""
        loading the TADPOLE dataset
    """

    def __init__(self, root, name, transform=None, pre_transform=None, provided_graph=5):
        self.name = name.lower()
        self.provided_graph = provided_graph
        super(Covid, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return 'covid_data.pt'

    def process(self):
        covid_data = pd.read_csv("/vol/aimspace/users/muel/DGM_pytorch/data/Covid/COVID19_04-02-pos-Python-normalisiert.csv")

        x_data = torch.tensor(covid_data.drop(["ICU", "Unnamed: 0"], axis=1).values)
        y = torch.tensor(covid_data["ICU"].values)
        k = 10
        edge_prob = 0.001
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_mask, test_mask, val_mask = utils.create_train_test_val_masks_levelled(y, test_size=((1-0.7)/2), validation_size=((1-0.7)/2), random_state=42)

        min_max_scaler = preprocessing.MinMaxScaler()
        min_max_scaler.fit(x_data[train_mask])
        X = torch.tensor(min_max_scaler.transform(x_data), dtype=torch.float32).to(device)

        train_mask = torch.from_numpy(train_mask).to(device)
        test_mask = torch.from_numpy(test_mask).to(device)
        val_mask = torch.from_numpy(val_mask).to(device)

        if self.provided_graph == 0:
            print("Not using provided graph")
            edge_index = torch.tensor([[0],[0]]).to(device)
        elif self.provided_graph == 1:
            print("using self loops only")
            edge_index = torch_geometric.utils.add_self_loops(torch.tensor([[0],[0]]), num_nodes=X.shape[0])[0].to(device)
        elif self.provided_graph == 2:
            print("using random graph to start")
            edge_index = torch_geometric.utils.erdos_renyi_graph(X.shape[0], edge_prob=edge_prob).to(device)
        elif self.provided_graph == 3:
            print("using graph with random n neighbours")
            edges0 = torch.randint(X.shape[0], (X.shape[0]*k,)).tolist()
            edges1_listoflists =[[i]*k for i in range(X.shape[0])]
            edges1 = [item for sublist in edges1_listoflists for item in sublist]
            edge_index = torch.tensor([edges0,edges1]).to(device)
        elif self.provided_graph == 5:
            print("use Euclidean kNN graph")
            edge_index = torch_geometric.nn.knn_graph(X, k=k, loop=True, flow='source_to_target').to(device)
            edge_index = to_undirected(edge_index)
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = coalesce(edge_index, None, X.size(0), X.size(0))
        elif self.provided_graph == 6:
            print("using cosine sim KNN graph")
            edge_index = torch_geometric.nn.knn_graph(X.to(device), k=k, loop=True, flow='source_to_target', cosine=True).to(device)
        elif self.provided_graph == 10:
            percentage_same_label_neighbours = 0.9
            print(f"Caution, synthetically generating graph with {percentage_same_label_neighbours} same label neighbours")
            nr_same_labelled_neighbours = int(percentage_same_label_neighbours*k)
            nr_diff_labelled_neighbours = int(k - nr_same_labelled_neighbours)
            edge_index = utils.get_edges_based_on_ground_truth(
                X.shape[-2], y.argmax(dim=1), nr_same_label_neighbours=nr_same_labelled_neighbours,
                nr_different_label_neighbours=nr_diff_labelled_neighbours)

        data = Data(x=X, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        # data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.name)



def get_fixed_splits(data, dataset_name, seed):
    with np.load(f'/vol/aimspace/users/muel/neural-sheaf-diffusion/splits/{dataset_name}_split_0.6_0.2_{seed}.npz') as splits_file:
        train_mask = splits_file['train_mask']
        val_mask = splits_file['val_mask']
        test_mask = splits_file['test_mask']

    data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
    data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
    data.test_mask = torch.tensor(test_mask, dtype=torch.bool)

    if dataset_name in {'cora', 'citeseer', 'pubmed'}:
        data.train_mask[data.non_valid_samples] = False
        data.test_mask[data.non_valid_samples] = False
        data.val_mask[data.non_valid_samples] = False
        print("Non zero masks", torch.count_nonzero(data.train_mask + data.val_mask + data.test_mask))
        print("Nodes", data.x.size(0))
        print("Non valid", len(data.non_valid_samples))
    else:
        assert torch.count_nonzero(data.train_mask + data.val_mask + data.test_mask) == data.x.size(0)

    return data


def get_dataset(name):
    data_root = osp.join(ROOT_DIR, 'datasets')
    if name in ['cornell', 'texas', 'wisconsin']:
        dataset = WebKB(root=data_root, name=name, transform=T.NormalizeFeatures())
    elif name in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(root=data_root, name=name, transform=T.NormalizeFeatures())
    elif name == 'film':
        dataset = Actor(root=data_root, transform=T.NormalizeFeatures())
    elif name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root=data_root, name=name, transform=T.NormalizeFeatures())
    elif name == 'tadpole':
        dataset = Tadpole(root=data_root, name=name, transform=T.NormalizeFeatures(), provided_graph=5)
    elif name == 'abide':
        dataset = Abide(root=data_root, name=name, transform=T.NormalizeFeatures(), provided_graph=5)
    elif name == 'ukbbcardiac':
        dataset = UKBBCardiac(root=data_root, name=name, transform=T.NormalizeFeatures(), provided_graph=5)
    elif name == 'covid':
        dataset = Covid(root=data_root, name=name, transform=T.NormalizeFeatures(), provided_graph=5)
    else:
        raise ValueError(f'dataset {name} not supported in dataloader')

    return dataset