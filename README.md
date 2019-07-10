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

```python
>>> import pandas as pd
>>> import networkx as nx
>>> import numpy as np
>>> import scikitgraph as sg

>>> G = nx.karate_club_graph() # Importes the graph
>>> f = pd.DataFrame(data = {'name': range(34),'col1': np.random.rand(34), 'col2': np.random.rand(34)}) # Creates random features for the nodes
>>> f.columns
Index(['name', 'col1', 'col2'], dtype='object')

>>> f = sg.betweenness(G,f) #Adds a column to the dataframe with the betweenness centrality of the nodes.
>>> f = sg.pagerank(G,f) #Adds a column to the dataframe with the PageRank of the nodes.
>>> f = sg.node_embeddings(G,f,20, walk_length=10, num_walks=50) #Adds columns to the dataframe with the embeddings of the nodes.
>>> f.columns
Index(['name', 'col1', 'col2', 'betweenness', 'pagerank', 'node_embeddings_0',
       'node_embeddings_1', 'node_embeddings_2', 'node_embeddings_3',
       'node_embeddings_4', 'node_embeddings_5', 'node_embeddings_6',
       'node_embeddings_7', 'node_embeddings_8', 'node_embeddings_9',
       'node_embeddings_10', 'node_embeddings_11', 'node_embeddings_12',
       'node_embeddings_13', 'node_embeddings_14', 'node_embeddings_15',
       'node_embeddings_16', 'node_embeddings_17', 'node_embeddings_18',
       'node_embeddings_19'],
      dtype='object')
```

## Contributing

Pull requests for new features, bug fixes, and suggestions are welcome!

## License

[MIT](https://github.com/fedealbanese/scikitgraph/blob/master/LICENSE)
