# scikitgraph
A graph based machine learning library for python.


## Installation

```bash
python setup.py install
```

## Requirements
* numpy
* pandas
* scickit learn
* networkx (>2.4)

## Simple example

Adding new columns to the dataset.

```bash
import pandas as pd
import networkx as nx
import numpy as np
import scikitgraph as sg

G = nx.karate_club_graph() # Importes the graph
f = pd.DataFrame(data = {'name': range(34),'col1': np.random.rand(34), 'col2': np.random.rand(34)}) # Creates random features for the nodes

f = gfp.betweenness(G,f) #Adds a column to the dataframe with the betweenness centrality of the nodes.
f = gfp.pagerank(G,f) #Adds a column to the dataframe with the PageRank of the nodes.
f = gfp.node_embeddings(G,f,20, walk_length=10, num_walks=50) #Adds columns to the dataframe with the embeddings of the nodes.

```
