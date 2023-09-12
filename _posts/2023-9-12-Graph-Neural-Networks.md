---
layout: page
title: Graph Neural Networks
---

Many real-world applications require processing and understanding of graph-structured data. For instance, one may want to process World Wide Web data, social networks, citation networks, scene graphs, human bodies, and many more. What unites these kinds of data is the availability of neighbourhood information amongst entities. This information can serve as a very strong end informative inductive bias. Thus, it is worthwhile exploring neural architectures that can embed this information in the processing. This is where **Graph Neural Networks** play their part in the abundance of neural architectures.  

In this blog, we will discuss the basics of graph neural networks, their variations, and some notable extensions that can be useful in certain applications.

# Introduction

Graph Neural Networks (GNN) are a class of Deep Learning architectures that are specifically designed to process graph-structured data.
In other words, they can process data structures that contain neighborhood information between entities. Mainly, this is done by sharing features of connected nodes through a message-passing operation. This way, the structural prior of the graph is embedded in the neural network. 

More formally, GNNs are tailored to process graph data: $G = \left(X, V, E\right)$, where $V$ are the vertices of the graph and $E$ are the edges that connect vertices. Additionally, many applications have access to the vertex attributes $X \in \mathcal{R}^{N \times D}$ which essentially holds features $x_i$ for each vertex $i$ in the graph. Edges are typically represented as the adjacency matrix $A$ where $A_{ij} = 1$ if $(i, j) \in E$. The goal of the GNN is to take the node attributes and edges $X, E$ and produce a node-level representation $H \in \mathcal{R}^{N \times F}$. Thanks to message-passing, features of adjacent nodes influence each other. This results in representations $H$ that are aware of the connectivity and structure of the input graph. (A nice demo here [Graph-Convolutional Networks](https://tkipf.github.io/graph-convolutional-networks/)).

# Types of Graph Neural Networks

There are two major types of GNNs in the literature, Spectral-based and Spatial-Based GNNs. Spectral-Based GNNs perform graph operations in the eigenspace of the Laplacian matrix of the graph, hence the name. Spatial-based GNNs are more straightforward and intuitive as they directly operate on the vertex features while incorporating adjacency information from edges $E$.

**Spectral-Based GNNs**: These types of GNNs are based on the convolution on the graph operator denoted as $\boldsymbol{x} \ast_G \boldsymbol{g}$. Here, $\boldsymbol{U}$ is the matrix composed of eigenvectors of the normalized Laplacian of the graph derived from the adjacency matrix: $\boldsymbol{L}_{norm} = \textbf{I}_n - \boldsymbol{D}^{-1/2}\boldsymbol{A}\boldsymbol{D}^{-1/2}$.

$$\boldsymbol{x} \ast_G \boldsymbol{g} = \boldsymbol{U}\boldsymbol{g}_{\theta}\boldsymbol{U}^\textrm{T}\boldsymbol{x}$$

Spectral-based GNNs differ in the design choice of the learnable filter $\boldsymbol{g}_{\theta}$ in the equation above. The disadvantage of this type of GNNs is the requirement to compute the eigenspace of the Laplacian matrix which is a costly operation. Moreover, this operation is not flexible. When the structure of the input graph is changed, the eigenspace changes as well requiring a different filter.

**Spatial-Based GNNs**: The difference between Spectral-Based GNNs and Spatial-Based GNNs is that message-passing in the vertex space as opposed to the spectral space. So, a typical Spatial-Based GNN can be formulated as follows

$$
\boldsymbol{h}_v^{(k)} = U_k \left( \boldsymbol{h}_v^{(k - 1)}, \sum _{u \in N(v)} M_k(\boldsymbol{h}_v^{(k - 1)}, \boldsymbol{h}_u^{(k - 1)}, e _{vu}) \right)
$$

Here where $\boldsymbol{h}_v^{(0)} = \boldsymbol{x}_v$ and $U _k$ and $M _k$ are neural networks with learnable parameters, such as Multilayer Perceptrons (MLP). $\boldsymbol{e _{vu}}$ denotes the (optional) edge features connecting nodes $v$ and $u$. The equation, in essence, updates the features of nodes using the feature from previous layer: $\boldsymbol{h}_v^{(k - 1)}$ and the aggregated neighbour messages from all neighbours: $\sum _{u \in N(v)} M_k(\boldsymbol{h}_v^{(k - 1)}, \boldsymbol{h}_u^{(k - 1)}, e _{vu})$. This operation optionally takes into account edge features, if there are any.

**Graph-Convolutional Neural Networks (GCN)**: GCN by *Kipf et al.* is a kind of middle ground between spectral and spatial-based GNNs. It can be interpreted through both lenses. GCN forward pass is computed as follows:

$$ \boldsymbol{H} = \sigma(\hat{\boldsymbol{A}}\boldsymbol{X}\boldsymbol{W}) $$

where $\boldsymbol{H} \in R^{N \times H}$ is the hidden state of nodes in a graph after a forward pass, $\boldsymbol{X} \in R^{N \times \textrm{D}}$ is the input graph signal where each of $n$ nodes has $D$ dimensional features and $\boldsymbol{W} \in R^{\textrm{D} \times H}$ is the learnable weight matrix. The result is passed through a non linearity $\sigma$, which can be arbitrarily chosen. A common choice is the $ReLU$ function. Stacking several such layers on top of each other makes the network process broader neighbourhood information. The matrix $\hat{\boldsymbol{A}}$ can be initialized as either the degree or symmetrically normalized adjacency matrix which also includes self-links. 

To see why this definition can be viewed as a spatial-based method, it can be rewritten as follows:

$$ \boldsymbol{h}_v = \sigma \left ( \sum _{u \in \{N(v) \cup v \} } \boldsymbol{\hat{A}} _{v,u} \boldsymbol{W}^{\textrm{T}} \boldsymbol{x}_u \right) $$

Now, we can interpret GCN as a network that weights and aggregates neighbourhood features, akin to more general spatial-based GNNs.

# Extensions of Graph Neural Networks
There are several modifications one can perform on top of standard GNNs that can boost the performance of those models. Several of them are listed and described below. Keep in mind that these are application-specific and may not be suitable for every task.

**Weight-Unsharing in GCN**: One of the limitations of the vanilla GCN described above is the shared weight $W$ for all nodes. That is, all node features are extracted using the same matrix. This can be a limitation is those cases where nodes in the graph clearly behave differently. For instance, when modeling the human body as a graph of joints, it is clear that not all human joints adhere to the same pattern. Hence, it is beneficial to have different weight matrices associated with different joints. This process is typically called weight-unsharing.

One can achieve weight unsharing by modifying the vanilla GCN equation to be:

$$ \boldsymbol{h}_v = \sigma \left ( \sum _{u \in \{N(v) \cup v \} } \boldsymbol{\hat{A}} _{v,u} \boldsymbol{W}^{\textrm{T}} _{g(v, u)} \boldsymbol{x}_u \right) $$

Where $g(v, u)$ corresponds to the group where nodes $v$ and $u$ belong. So, $\boldsymbol{W}_{g(v, u)}$ is the feature transformation specific to the group $g(v, u)$. So, nodes are partitioned into several groups, and each group has its own weight matrix. Admittedly, the capacity of feature extraction of the network grows. However, one should think about this partitioning and how it can be done in regard to the use case at hand.

**Partitioned GCN**: Sometimes, a symmetric adjacency matrix doesn't really capture the nature of relationships in the graph at hand. Again, take the human body as the graph. It is evident that a parent joint (e.g. Knee) influences the child joint (e.g. Foot) in a different manner than the child influences the parent. In this case, it is worth thinking about partitioning the whole graph into directed subgraphs that reflect desired directional relationships. In the GCN, this can be done by splitting the adjacency matrix and modifying the formula just a bit.

The first step is to split the adjacency matrix into multiple groups $A_1, A_2, \dots, A_n$. These groups are not necessarily mutually exclusive and the same joint can be included in multiple subgroups. Then, define a separate weight matrix for each group: $W_1, W_2, \dots, W_n$. The forward pass then becomes:

$$ \boldsymbol{H} = \sigma \left( \sum_{g \in G} \hat{\boldsymbol{A}}_g \boldsymbol{X}\boldsymbol{W}_g  \right) $$

This way, the GCN can extract and aggregate group-specific features. Again, one should think about how the graph at hand should be partitioned into groups. A meaningful split of the adjacency matrix can result in significant performance boosts.

**Relationship to Self-Attention**: Lastly, one can notice that GNNs are very much related to self-attention in Transformers. Self-attention essentially computes dynamic weights that are used to aggregate features from the input set. It is formulated as:

$$ \textrm{Attention}(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V}) = \textrm{SoftMax}(\frac{\boldsymbol{Q}\boldsymbol{K}^T}{\sqrt{d_k}})\boldsymbol{V} $$

If we note $\boldsymbol{\hat{A}} = \textrm{SoftMax}(\frac{\boldsymbol{Q}\boldsymbol{K}^T}{\sqrt{d_k}})$, we can see how similar this operation is to the multiplication with the adjacency matrix in GCN for message-passing. Indeed, self-attention can be thought of as designing a dynamically weighted adjacency matrix conditioned on the input. This is far more expressive than the GCN formulation. However, self-attention operation comes at a significant computational cost. So, it is worth considering simpler GNN-based methods if the task at hand permits it.

# Summary

Graph Neural Networks have seen significant use in many fields of deep learning these past few years. Their ability to extract connectivity features from graphs is remarkable. The main striking force of these networks is the ability to create embeddings while incorporating neighbourhood priors in the graphs. Many variations of GNNs have been developed and I expect more architectures tailored to specific tasks will be researched in the future. 
