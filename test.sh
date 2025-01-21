#!/bin/bash
python - <<EOF
import torch
print(torch.__version__) 
print(torch.version.cuda)
import torch_sparse
import torch_cluster
import torch_scatter
import torch_spline_conv
import torch_geometric
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Node2Vec
from sklearn.neighbors import NearestNeighbors, KDTree, BallTree
from sklearn.metrics import DistanceMetric
EOF