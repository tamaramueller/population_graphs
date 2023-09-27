import torch
import torch_geometric
from tqdm import tqdm
import random
import pandas as pd
import numpy as np
from torch_geometric.data import Data, DataLoader, InMemoryDataset
from sklearn.datasets import make_classification, make_regression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


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


def make_deterministic(SEED=42):
    torch.backends.cudnn.deterministic = True
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    cpu_generator = torch.Generator(device="cpu").manual_seed(SEED)
    gpu_generator = torch.Generator(device="cuda").manual_seed(SEED)
    
    return cpu_generator, gpu_generator


def get_edges_based_on_ground_truth(nr_patients, labels, nr_same_label_neighbours, nr_different_label_neighbours,
                                   make_undirected=True):
    """
    This function returns 'edges' between nodes such that the node is connected to nr_same_label_neighbours 
    neighbours with the same label and nr_different_label_neighbours neighbours with a different label
    @param nr_patients: int: number of patients
    @param labels: LongTensor with labels
    @param nr_same_label_neighbours: number of neighbours with the same label
    @param nr_different_label_neighbours: number of neighbours with a different label

    return edges: LongTensor([[],[]]) matching shape of PyTorchGeometric edges
    """
    patients_with_label = {}
    for label in set(labels.flatten().detach().cpu().numpy()):
        patients_with_label[label] = list((labels == label).nonzero(as_tuple=False).squeeze(1).detach().cpu().numpy())

    # edges = torch.LongTensor([[],[]])
    target_nodes = []
    source_nodes = []

    for i in range(0, nr_patients):
        # iterating through all patients
        label_i = int(labels[i])

        list_all_diff_label_neighbours = []
        for key in patients_with_label.keys():
            if key != label_i:
                list_all_diff_label_neighbours.append(patients_with_label[key])
        list_all_diff_label_neighbours_flat = [item for sublist in list_all_diff_label_neighbours for item in sublist]

        # sample nr_same_label_neighbours nieghbours with same label
        if len(patients_with_label[label_i]) >= nr_same_label_neighbours:
            same_label_nodes = random.sample(patients_with_label[label_i], nr_same_label_neighbours)
        else:
            same_label_nodes = patients_with_label[label_i] + random.sample(list_all_diff_label_neighbours_flat, min((nr_same_label_neighbours-len(patients_with_label[label_i])), len(list_all_diff_label_neighbours_flat)))
        # sample nr_different_label_neighbours nieghbours with different label
        

        different_label_nodes = random.sample(list_all_diff_label_neighbours_flat, min(nr_different_label_neighbours, len(list_all_diff_label_neighbours_flat)))

        new_neighbours = same_label_nodes + different_label_nodes
        target_nodes.append(new_neighbours)
        source_nodes.append([i]*(nr_same_label_neighbours+nr_different_label_neighbours))

    return torch.LongTensor([[item for sublist in source_nodes for item in sublist], [item for sublist in target_nodes for item in sublist]]).to(labels.device)


def get_train_test_val_split(labels_vector, train_size, seed=42):
    train_ind, test_ind = train_test_split(np.arange(len(labels_vector)), test_size=(1-train_size), random_state=seed)
    test_ind, val_ind = train_test_split(test_ind, test_size=0.5, random_state=seed)

    train_mask = np.array([x in train_ind for x in np.arange(len(labels_vector))])
    val_mask = np.array([x in val_ind for x in np.arange(len(labels_vector))])
    test_mask = np.array([x in test_ind for x in np.arange(len(labels_vector))])

    return train_ind, test_ind, val_ind, train_mask, test_mask, val_mask


def get_correlation_of_dataframe(dataframe, col_of_interest):
    return dataframe.corr()[col_of_interest].sort_values(ascending=False)


def generate_synthetic_population_dataset(nr_nodes:int, nr_classes:int, nr_features:int, nr_informative:int, seed:int):
    """
    @params:
        nr_graphs: int: number of graphs in dataset
        nr_classes: int: number of classes in the dataset
        connectivity: list: specify which probability of connectivity each class should have
            e.g. for a binary classification task the connectivity list could look like this: [0.1, 0.6] to produce 
                 very dense graphs for one class and very sparse ones for the other
        nodes_per_graph: int: specifies the number of nodes a graph should contain
        nr_node_features: int: specifies the nunmber of node features per node
        means: list: list of mean values for Gaussian distributions from which node features of the different classes will be sampled
        std_devs: list: list of standard deviations for Gaussian distribution from which node featuers of the different classes will be sampled
    """

    data_x, data_y = make_classification(n_samples=nr_nodes, n_features=nr_features, n_informative=nr_informative, n_classes=nr_classes, \
                                                          shuffle=True, random_state=seed)
    columns = [str(i) for i in range(nr_features)]
    data = pd.DataFrame(data_x, columns=columns)
    data["label"] = data_y
    data["RID"] = [int(i) for i in range(nr_nodes)]

    return data, torch.tensor(data_x), torch.tensor(data_y)


def generate_synthetic_population_dataset_regression(nr_nodes:int, nr_targets:int, nr_features:int, nr_informative:int, seed:int):
    """
    @params:
        nr_graphs: int: number of graphs in dataset
        nr_classes: int: number of classes in the dataset
        connectivity: list: specify which probability of connectivity each class should have
            e.g. for a binary classification task the connectivity list could look like this: [0.1, 0.6] to produce 
                 very dense graphs for one class and very sparse ones for the other
        nodes_per_graph: int: specifies the number of nodes a graph should contain
        nr_node_features: int: specifies the nunmber of node features per node
        means: list: list of mean values for Gaussian distributions from which node features of the different classes will be sampled
        std_devs: list: list of standard deviations for Gaussian distribution from which node featuers of the different classes will be sampled
    """

    data_x, data_y = make_regression(n_samples=nr_nodes, n_features=nr_features, n_informative=nr_informative, n_targets=nr_targets, \
                                                          shuffle=True, random_state=seed)
    columns = [str(i) for i in range(nr_features)]
    data = pd.DataFrame(data_x, columns=columns)
    data["label"] = data_y
    data["RID"] = [int(i) for i in range(nr_nodes)]

    return data, torch.tensor(data_x), torch.tensor(data_y)


def convert_edges_to_adj_matrix(edges, nr_nodes):
    """
    This function takes PyTorchGeometric edges and converts them into an adjacency matrix of the graph
    @param edges: torch.LongTensor([[],[]]) representing the edges of a graph
    @param nr_nodes: int value number of nodes
    
    return adj_matrix: torch.Tensor of size nr_nodes x nr_nodes with a 1 if there is an edge between the nodes and 0 otherwise
    """
    adj_matrix = torch.zeros((nr_nodes, nr_nodes))
    for i in range(len(edges[0])):
        adj_matrix[edges[0][i]][edges[1][i]] = 1
        
    return adj_matrix


def get_matrix_indicating_same_labels(labels:torch.tensor):
    """
        This function returns a binary matrix indicating which nodes have the same labels
        a_ij = 0: nodes a_i and a_j don't have the same label
        a_ij = 1: ndoes a_i and a_j have the same label
    """
    return (torch.cdist(labels.unsqueeze(-1).float(), labels.unsqueeze(-1).float())==0).int()


def get_matrix_indicating_different_labels(labels:torch.tensor):
    """
        This function returns a binary matrix indicating which nodes have different labels
        a_ij = 0: nodes a_i and a_j have the same label
        a_ij = 1: ndoes a_i and a_j don't have the same label
    """
    return (torch.cdist(labels.unsqueeze(-1).float(), labels.unsqueeze(-1).float())!=0).int()


def get_number_conn_same_label_different_label(adj:torch.tensor, labels:torch.tensor, train_mask:torch.tensor=None, device="cpu"):
    """
        Returns two vectors indicating the number of connections to the same label and to different labels
        for each node in the graph
        adj: adjacency matrix (torch.tensor of shape [x,x])
        labels: label vector (torch.tensor of shape [x])
        train_mask: in case of transductive learning, you can pass a train mask and only the GNA of the training nodes will be evaluated
    """
    device = labels.device.type
    
    if train_mask is None:
        train_mask = torch.ones(adj.shape).to(device)
    else: 
        train_mask = torch.matmul(train_mask.unsqueeze(0).double().T, train_mask.unsqueeze(0).double()).int().to(device)
    matrix_ind_same_labels = get_matrix_indicating_same_labels(labels)
    matrix_ind_diff_labels = get_matrix_indicating_different_labels(labels)
    conn_same_labels = adj * matrix_ind_same_labels * train_mask
    conn_diff_labels = adj * matrix_ind_diff_labels * train_mask

    conn_same_labels = conn_same_labels.sum(dim=0)
    conn_diff_labels = conn_diff_labels.sum(dim=0)

    no_neighbours_in_subset = (conn_same_labels + conn_diff_labels)==0

    return conn_same_labels, conn_diff_labels, no_neighbours_in_subset


def get_all_incoming_neighbours_of_node(node_id, edges, train_ids=None):
    """
    returns a tensor with all neighbours of a node (index)
    @param node_id: ID of the node you wnat to get all neighbours from
    @param edges: torch [[],[]] with edges
    
    returns tensor with neighbour node IDs
    """
    return edges[0][torch.where(edges[1]==node_id)]