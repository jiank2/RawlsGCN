import os
import numpy as np
import scipy.sparse as sp
import torch

from collections import defaultdict

from utils.utils import encode_onehot, row_normalize, symmetric_normalize, matrix2tensor
from utils.sinkhorn_knopp import SinkhornKnopp


class GraphDataset:
    def __init__(self, configs):
        if not os.path.isfile("../data/{name}.pt".format(name=configs["name"])):
            raise FileNotFoundError("Dataset does not exist!")
        # load data
        data = torch.load("../data/{name}.pt".format(name=configs["name"]))

        # read fields
        self.num_nodes = data["num_nodes"]
        self.num_edges = data["num_edges"]
        self.num_node_features = data["num_node_features"]
        self.num_classes = data["num_classes"]
        self.raw_graph = data["adjacency_matrix"]
        self.features = torch.FloatTensor(
            np.array(
                row_normalize(data["node_features"])
            )
        )
        self.labels = data["labels"]

        self.is_ratio = configs["is_ratio"]
        self.split_by_class = configs["split_by_class"]
        self.num_train = configs["num_train"]
        self.num_val = configs["num_val"]
        self.num_test = configs["num_test"]
        self.ratio_train = configs["ratio_train"]
        self.ratio_val = configs["ratio_val"]

        # free memory
        del data

    def random_split(self):
        # initialization
        mask = torch.empty(self.num_nodes, dtype=torch.bool).fill_(False)
        if self.is_ratio:
            self.num_train = int(self.ratio_train * self.num_nodes)
            self.num_val = int(self.ratio_val * self.num_nodes)
            self.num_test = self.num_nodes - self.num_train - self.num_val

        # get indices for training
        if not self.is_ratio and self.split_by_class:
            self.train_idx = self.get_split_by_class(num_train_per_class=self.num_train)
        else:
            self.train_idx = torch.randperm(self.num_nodes)[:self.num_train]

        # get remaining indices
        mask[self.train_idx] = True
        remaining = (~mask).nonzero(as_tuple=False).view(-1)
        remaining = remaining[torch.randperm(remaining.size(0))]

        # get indices for validation and test
        self.val_idx = remaining[:self.num_val]
        self.test_idx = remaining[self.num_val:self.num_val + self.num_test]

        # free memory
        del mask, remaining

    def set_random_split(self, splits):
        self.train_idx = splits["train_idx"]
        self.val_idx = splits["val_idx"]
        self.test_idx = splits["test_idx"]

    def get_split_by_class(self, num_train_per_class):
        res = None
        for c in range(self.num_classes):
            idx = (self.labels == c).nonzero(as_tuple=False).view(-1)
            idx = idx[torch.randperm(idx.size(0))[:num_train_per_class]]
            res = torch.cat((res, idx)) if res is not None else idx
        return res

    @staticmethod
    def get_doubly_stochastic(mat):
        sk = SinkhornKnopp(max_iter=1000, epsilon=1e-2)
        mat = matrix2tensor(
            sk.fit(mat)
        )
        return mat
    
    @staticmethod
    def get_row_normalized(mat):
        mat = matrix2tensor(
            row_normalize(mat)
        )
        return mat
    
    @staticmethod
    def get_column_normalized(mat):
        mat = matrix2tensor(
            row_normalize(mat)
        )
        mat = torch.transpose(mat, 0, 1)
        return mat
    
    @staticmethod
    def get_symmetric_normalized(mat):
        mat = matrix2tensor(
            symmetric_normalize(mat)
        )
        return mat

    def preprocess(self, type="laplacian"):
        if type == "laplacian":
            self.graph = matrix2tensor(
                symmetric_normalize(self.raw_graph + sp.eye(self.raw_graph.shape[0]))
            )
        elif type == "row":
            self.graph = matrix2tensor(
                row_normalize(self.raw_graph + sp.eye(self.raw_graph.shape[0]))
            )
        elif type == "doubly_stochastic_no_laplacian":
            self.graph = self.get_doubly_stochastic(self.raw_graph + sp.eye(self.raw_graph.shape[0]))
        elif type == "doubly_stochastic_laplacian":
            self.graph = symmetric_normalize(self.raw_graph + sp.eye(self.raw_graph.shape[0]))
            self.graph = self.get_doubly_stochastic(self.graph)
        else:
            raise ValueError(
                "type should be laplacian, row, doubly_stochastic_no_laplacian or doubly_stochastic_laplacian"
            )

    def get_degree_splits(self):
        deg = self.raw_graph.sum(axis=0)
        self.degree_splits = defaultdict(list)
        for idx in range(self.num_nodes):
            degree = deg[0, idx]
            self.degree_splits[degree].append(idx)
    
    def encode_degree_splits_to_labels(self):
        label = 0
        encoded_labels = set()
        self.degree_labels = [0] * self.num_nodes
        for degree, nodes in self.degree_splits.items():
            if degree in encoded_labels:
                continue
            for node_id in nodes:
                self.degree_labels[node_id] = label
            label += 1
        self.degree_labels = torch.LongTensor(self.degree_labels)
