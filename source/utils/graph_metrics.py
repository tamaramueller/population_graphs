import torch
import torch.nn.functional as F
import numpy as np
import torch
import sys
import torch_geometric
import socket
import os

# from . import missing_data_utils, tadpole_configs, train_utils

sys.path.append("../source/")
from utils import utils


def get_cosine_similarity(vec1, vec2):
    if isinstance(vec1, np.ndarray):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    elif isinstance(vec1, torch.Tensor):
        return torch.nn.CosineSimilarity(dim=0)(vec1.float(), vec2.float())
    else:
        sys.exit(-1, "type must be numpy array or torch tensor")


def get_consecutive_labels(labels):
    unique_labels = torch.unique(labels.long()).to(labels.device)
    n_classes = len(unique_labels)
    label_to_index = torch.zeros(torch.amax(labels).long().item() + 1, dtype=torch.long, device=labels.device)
    label_to_index[unique_labels] = torch.arange(n_classes).to(labels.device)
    consecutive_labels = label_to_index[labels.long()]

    return consecutive_labels, n_classes


def get_cross_class_neighbourhood_similarity_edges(edges, labels, one_hot_input:bool=False, train_mask=None):
    """
        calculate the cross-class neighbourhood similarity from https://arxiv.org/pdf/2106.06134.pdf
    """
    if train_mask is not None  and not train_mask.dtype == torch.bool:
        train_mask = train_mask.bool()

    if not one_hot_input:
        consecutive_labels, n_classes = get_consecutive_labels(labels=labels)
        one_hot = torch.eye(n_classes, device=labels.device)[consecutive_labels] # shape: (nr_nodes, nr_classes)
    else:
        one_hot = labels.squeeze(0)
        n_classes = one_hot.shape[-1]
        consecutive_labels = torch.argmax(one_hot, dim=1)

    if train_mask is not None:
        train_ids = torch.where(train_mask)[0]
        # edges= torch_geometric.utils.subgraph(train_ids, edges)[0]
        # consecutive_labels = consecutive_labels#[train_mask]
        one_hot = one_hot[train_mask]
        # only use train nodes in case we're interested in the train set only
        neighbours_generator = (utils.get_all_incoming_neighbours_of_node(i, edges) for i in train_ids)
    else:
        # otherwise use full graph
        neighbours_generator = (utils.get_all_incoming_neighbours_of_node(i, edges) for i in range(consecutive_labels.shape[0]))

    histogram = torch.stack([torch.bincount(consecutive_labels[neighbours], minlength=n_classes) for neighbours in neighbours_generator]).float()

    class_count = torch.sum(one_hot, dim=0) # shape: (nr_classes)

    similarities = F.cosine_similarity(histogram.unsqueeze(0) , histogram.unsqueeze(1) , dim=-1, eps=1e-8) # shape: (nr_edges)
    nominator = one_hot.T @ similarities @ one_hot # shape: (nr_classes, nr_classes)

    denominator = torch.outer(class_count, class_count) # shape: (nr_classes, nr_classes)

    return nominator / denominator # shape: (nr_classes, nr_classes)


def get_cross_class_neighbourhood_similarity_adj(adj_matrix, labels, one_hot:bool=False, train_mask=None):
    """
        calculate the cross-class neighbourhood similarity from https://arxiv.org/pdf/2106.06134.pdf
    """
    if train_mask is not None  and not train_mask.dtype == torch.bool:
        train_mask = train_mask.bool()  
    if not one_hot:
        consecutive_labels, n_classes = get_consecutive_labels(labels=labels)

        one_hot = torch.eye(n_classes, device=labels.device)[consecutive_labels] # shape: (nr_nodes, nr_classes)
    else:
        one_hot = labels.squeeze(0)
    adj_matrix = adj_matrix.squeeze(0)
    histogram = adj_matrix.T @ one_hot

    class_count = torch.sum(one_hot, dim=0) # shape: (nr_classes)

    similarities = F.cosine_similarity(histogram.unsqueeze(0) , histogram.unsqueeze(1) , dim=-1, eps=1e-8) # shape: (nr_edges)
    nominator = one_hot.T @ similarities @ one_hot # shape: (nr_classes, nr_classes)

    denominator = torch.outer(class_count, class_count) # shape: (nr_classes, nr_classes)

    return nominator / denominator # shape: (nr_classes, nr_classes)


def graph_neighbourhood_assessment_static(edges, nr_nodes, labels, train_mask=None):
    """
        calculate the 1-hop graph neighbourhood assessment score (GNA)
    """
    if train_mask is not None and not train_mask.dtype == torch.bool:
        train_mask = train_mask.bool()
    adj_matrix = torch_geometric.utils.to_dense_adj(edges, max_num_nodes=nr_nodes).squeeze(0)
    # in case of transductive learning, you can pass a train mask and only the GNA of the training nodes will be evaluated
    nr_same_labelled_neighbours, nr_diff_labelled_neighbours, no_neighbours_in_subset = utils.get_number_conn_same_label_different_label(adj_matrix, labels, None)

    # if train_mask is not None:
    #     if isinstance(train_mask, np.ndarray):
    #         train_mask = torch.from_numpy(train_mask)
    #     ratio = nr_same_labelled_neighbours[train_mask]/(nr_same_labelled_neighbours[train_mask] + nr_diff_labelled_neighbours[train_mask])
    # else:
    ratio = nr_same_labelled_neighbours/(nr_same_labelled_neighbours + nr_diff_labelled_neighbours)

    ratio[torch.isinf(ratio)] = 0 
    ratio[torch.isnan(ratio)] = 0

    if train_mask is not None:
        ratio = ratio[~no_neighbours_in_subset & train_mask]
    else:
        ratio = ratio[~(no_neighbours_in_subset)]
    # if train_mask is not None:
    #     ratio = ratio[train_mask]

    return ratio, nr_same_labelled_neighbours, nr_diff_labelled_neighbours


def graph_neighbourhood_assessment_lgl(weight_matrix, labels, train_mask=None):
    """
        calculate the 1-hop graph neighbourhood assessment score (GNA) for the end-to-end training of the graph structure with the edge weights
    """

    # labels = torch.argmax(labels.squeeze(0), dim=1)

    if train_mask is not None  and not train_mask.dtype == torch.bool:
        train_mask = train_mask.bool()
    same_label_matrix = utils.get_matrix_indicating_same_labels(labels)
    diff_label_matrix = utils.get_matrix_indicating_different_labels(labels)

    weights_same_labels = same_label_matrix * weight_matrix.squeeze(0)
    weights_diff_labels = diff_label_matrix * weight_matrix.squeeze(0)

    # get weight of all incoming edges with same label
    nodewise_weights_same_label = weights_same_labels.mean(axis=0)
    # geit weight of all incoming edges with different label
    nodewise_weights_diff_label = weights_diff_labels.mean(axis=0)

    ratio = nodewise_weights_same_label / (nodewise_weights_diff_label + nodewise_weights_same_label)

    return ratio, nodewise_weights_same_label, nodewise_weights_diff_label


def graph_neighbourhood_assessment_regression_static(edges, nr_nodes, labels, train_mask=None):
    """
        calculate the 1-hop homophily for regression tasks with discrete adjacency matrices (edges)
    """
    if train_mask is not None  and not train_mask.dtype == torch.bool:
        train_mask = train_mask.bool()

    labels = (labels - (labels.min())) / (labels.max() - labels.min())
    
    adj_matrix = torch_geometric.utils.to_dense_adj(edges, max_num_nodes=nr_nodes).squeeze(0)

    distance_matrix = torch.cdist(labels, labels, p=1)
    distance_matrix = distance_matrix * adj_matrix

    summed_distance_per_node = distance_matrix.sum(axis=-2)
    nr_neighbours_per_node = adj_matrix.sum(axis=0)

    no_neighbours = (nr_neighbours_per_node == 0)
    
    homophily_per_node = (summed_distance_per_node / nr_neighbours_per_node).squeeze()
    
    if train_mask is not None:
        homophily_per_node = homophily_per_node[train_mask&(~no_neighbours)]
    else:
        homophily_per_node = homophily_per_node[~no_neighbours]
    homophily_per_node = homophily_per_node[~torch.isnan(homophily_per_node)]
    homophily_per_node = homophily_per_node[~torch.isinf(homophily_per_node)]

    homophily_mean = homophily_per_node.mean()
    homophily_std = homophily_per_node.std()

    return 1-homophily_mean, homophily_std


def graph_neighbourhood_assessment_regression_discrete(adj_matrix, labels, train_mask=None):
    """
        calculate the 1-hop homophily for regression tasks with continuous adjacency matrices 
    """
    if train_mask is not None  and not train_mask.dtype == torch.bool:
        train_mask = train_mask.bool()

    labels = (labels - (labels.min())) / (labels.max() - labels.min())

    distance_matrix = torch.cdist(labels, labels, p=1)
    distance_matrix = distance_matrix * adj_matrix

    summed_distance_per_node = distance_matrix.sum(axis=-2)
    nr_neighbours_per_node = adj_matrix.sum(axis=0)
    
    homophily_per_node = (summed_distance_per_node / nr_neighbours_per_node).squeeze(0)
    if train_mask is not None:
        homophily_per_node = homophily_per_node[train_mask]
    homophily_mean = homophily_per_node.mean()
    homophily_std = homophily_per_node.std()

    return 1-homophily_mean.item(), homophily_std.item()


def get_cross_class_neighbourhood_similarity_edges_khop(edges, labels, k:int, one_hot_input:bool=False, train_mask=None):
    """
        calculate the cross-class neighbourhood similarity from https://arxiv.org/pdf/2106.06134.pdf
    """
    if train_mask is not None  and not train_mask.dtype == torch.bool:
        train_mask = train_mask.bool()

    if not one_hot_input:
        consecutive_labels, n_classes = get_consecutive_labels(labels=labels)
        one_hot = torch.eye(n_classes, device=labels.device)[consecutive_labels] # shape: (nr_nodes, nr_classes)
    else:
        one_hot = labels.squeeze(0)
        n_classes = one_hot.shape[-1]
        consecutive_labels = torch.argmax(one_hot, dim=1)

    if train_mask is not None:
        train_ids = torch.where(train_mask)[0]
        # edges= torch_geometric.utils.subgraph(train_ids, edges)[0]
        # consecutive_labels = consecutive_labels#[train_mask]
        one_hot = one_hot[train_mask]
        # only use train nodes in case we're interested in the train set only
        neighbours_generator = (utils.get_k_hop_neighbours(i, k, edges) for i in train_ids)
        # neighbours_generator = (utils.get_k_hop_neighbours(edges, i, k) for i in train_ids)
    else:
        # otherwise use full graph
        neighbours_generator = (utils.get_k_hop_neighbours(i, k, edges) for i in range(consecutive_labels.shape[0]))

        # neighbours_generator = (utils.get_k_hop_neighbours(edges, i ,k) for i in range(consecutive_labels.shape[0]))

    histogram = torch.stack([torch.bincount(consecutive_labels[neighbours], minlength=n_classes) for neighbours in neighbours_generator]).float()

    class_count = torch.sum(one_hot, dim=0) # shape: (nr_classes)

    similarities = F.cosine_similarity(histogram.unsqueeze(0) , histogram.unsqueeze(1) , dim=-1, eps=1e-8) # shape: (nr_edges)
    nominator = one_hot.T @ similarities @ one_hot # shape: (nr_classes, nr_classes)

    denominator = torch.outer(class_count, class_count) # shape: (nr_classes, nr_classes)

    return nominator / denominator # shape: (nr_classes, nr_classes)


def get_ccns_distance(ccns:torch.tensor):
    return F.l1_loss(ccns, torch.eye(ccns.shape[0]))