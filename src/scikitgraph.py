import pandas as pd
import networkx as nx
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

#funtions
def degree(G,f):
    """
    Adds a column to the dataframe f with the degree of each node.
    G: a networkx graph.
    f: a pandas dataframe.
    """
    if not(set(f.name) == set(G.nodes()) and len(f.name) == len(G.nodes())):
        raise ValueError('The number of nodes and the length of the datadrame should be the same.')   
    degree_dic =  nx.degree_centrality(G)
    degree_df = pd.DataFrame(data = {'name': list(degree_dic.keys()), 'degree': list(degree_dic.values()) })  
    f = pd.merge(f, degree_df, on='name')
    return f

def centrality(G,f):
    """
    Adds a column to the dataframe f with the centrality of each node.
    G: a networkx graph.
    f: a pandas dataframe.
    """
    if not(set(f.name) == set(G.nodes()) and len(f.name) == len(G.nodes())):
        raise ValueError('The number of nodes and the length of the datadrame should be the same.')   
    centrality_dic = nx.degree_centrality(G)
    centrality_df = pd.DataFrame(data = {'name': list(centrality_dic.keys()), 'centrality': list(centrality_dic.values()) })  
    f = pd.merge(f, centrality_df, on='name')
    return f

def betweenness(G,f):
    """
    Adds a column to the dataframe f with the betweenness of each node.
    G: a networkx graph.
    f: a pandas dataframe.
    """
    if not(set(f.name) == set(G.nodes()) and len(f.name) == len(G.nodes())):
        raise ValueError('The number of nodes and the length of the datadrame should be the same.')   
    betweenness_dic = nx.betweenness_centrality(G)
    betweenness_df = pd.DataFrame(data = {'name': list(betweenness_dic.keys()), 'betweenness': list(betweenness_dic.values()) })  
    f = pd.merge(f, betweenness_df, on='name')
    return f

def pagerank(G,f):
    """
    Adds a column to the dataframe f with the pagerank of each node.
    G: a networkx graph.
    f: a pandas dataframe.
    """
    if not(set(f.name) == set(G.nodes()) and len(f.name) == len(G.nodes())):
        raise ValueError('The number of nodes and the length of the datadrame should be the same.')   
    pagerank_dic = nx.pagerank(G)
    pagerank_df = pd.DataFrame(data = {'name': list(pagerank_dic.keys()), 'pagerank': list(pagerank_dic.values()) })  
    f = pd.merge(f, pagerank_df, on='name')
    return f

def clustering(G,f):
    """
    Adds a column to the dataframe f with the clustering coeficient of each node.
    G: a networkx graph.
    f: a pandas dataframe.
    """
    if not(set(f.name) == set(G.nodes()) and len(f.name) == len(G.nodes())):
        raise ValueError('The number of nodes and the length of the datadrame should be the same.')   
    clustering_dic = nx.clustering(G)
    clustering_df = pd.DataFrame(data = {'name': list(clustering_dic.keys()), 'clustering': list(clustering_dic.values()) })  
    f = pd.merge(f, clustering_df, on='name')
    return f

def communities_greedy_modularity(G,f):
    """
    Adds a column to the dataframe f with the community of each node.
    The communitys are detected using greedy modularity.
    G: a networkx graph.
    f: a pandas dataframe.
    """
    # funciona con la version de '2.4rc1.dev_20190610203526' de netwrokx (no con la 2.1)
    if not(set(f.name) == set(G.nodes()) and len(f.name) == len(G.nodes())):
        raise ValueError('The number of nodes and the length of the datadrame should be the same.')   
    communities_dic = nx.algorithms.community.greedy_modularity_communities(G)
    communities_df = pd.DataFrame(data = {'name': [i for j in range(len(communities_dic)) for i in list(communities_dic[j])], 'communities_greedy_modularity': [j for j in range(len(communities_dic)) for i in list(communities_dic[j])] })  
    f = pd.merge(f, communities_df, on='name')
    return f

def communities_label_propagation(G,f):
    """
    Adds a column to the dataframe f with the community of each node.
    The communitys are detected using glabel propagation.
    G: a networkx graph.
    f: a pandas dataframe.
    """
    # funciona con la version de '2.4rc1.dev_20190610203526' de netwrokx (no con la 2.1)
    if not(set(f.name) == set(G.nodes()) and len(f.name) == len(G.nodes())):
        raise ValueError('The number of nodes and the length of the datadrame should be the same.')   
    communities_gen = nx.algorithms.community.label_propagation_communities(G)
    communities_dic = [community for community in communities_gen]
    communities_df = pd.DataFrame(data = {'name': [i for j in range(len(communities_dic)) for i in list(communities_dic[j])], 'communities_label_propagation': [j for j in range(len(communities_dic)) for i in list(communities_dic[j])] })  
    f = pd.merge(f, communities_df, on='name')
    return f

def mean_neighbors(G,f,column,n=1):
    """
    Adds a column to the dataframe f with the mean value of its neigbors feature.
    G: a networkx graph.
    f: a pandas dataframe.
    column: the column to which the mean is applied.
    n: no se como decirlo: seria a primeros vecinos o segundos vecinos.
    """
    if not(set(f.name) == set(G.nodes()) and len(f.name) == len(G.nodes())):
        raise ValueError('The number of nodes and the length of the datadrame should be the same.')   
    mean_neighbors = np.zeros([f.shape[0]])
    matrix = nx.to_numpy_matrix(G)
    for e in range(1,n):
        matrix += matrix ** e
    for i in f.index:
        neighbors = matrix[i]>0
        mean_neighbors[i] = f[neighbors.tolist()[0]][column].mean()
    f["mean_neighbors"] = mean_neighbors
    return f

def std_neighbors(G,f,column,n=1):
    """
    Adds a column to the dataframe f with the standar desviation value of its neigbors feature.
    G: a networkx graph.
    f: a pandas dataframe.
    column: the column to which the mean is applied.
    n: no se como decirlo: seria a primeros vecinos o segundos vecinos.
    """
    if not(set(f.name) == set(G.nodes()) and len(f.name) == len(G.nodes())):
        raise ValueError('The number of nodes and the length of the datadrame should be the same.')   
    std_neighbors = np.zeros([f.shape[0]])
    matrix = nx.to_numpy_matrix(G)
    for e in range(1,n):
        matrix += matrix ** e
    for i in f.index:
        neighbors = matrix[i]>0
        std_neighbors[i] = f[neighbors.tolist()[0]][column].std()
    f["std_neighbors"] = std_neighbors
    return f

def max_neighbors(G,f,column,n=1):
    """
    Adds a column to the dataframe f with the maximum value of its neigbors feature.
    G: a networkx graph.
    f: a pandas dataframe.
    column: the column to which the mean is applied.
    n: neighbourhood order.
    """
    if not(set(f.name) == set(G.nodes()) and len(f.name) == len(G.nodes())):
        raise ValueError('The number of nodes and the length of the datadrame should be the same.')   
    max_neighbors = np.zeros([f.shape[0]])
    matrix = nx.to_numpy_matrix(G)
    for e in range(1,n):
        matrix += matrix ** e
    for i in f.index:
        neighbors = matrix[i]>0
        max_neighbors[i] = f[neighbors.tolist()[0]][column].max()
    f["max_neighbors"] = max_neighbors
    return f

def min_neighbors(G,f,column,n=1):
    """
    Adds a column to the dataframe f with the minimum value of its neigbors feature.
    G: a networkx graph.
    f: a pandas dataframe.
    column: the column to which the mean is applied.
    n: neighbourhood order.
    """
    if not(set(f.name) == set(G.nodes()) and len(f.name) == len(G.nodes())):
        raise ValueError('The number of nodes and the length of the datadrame should be the same.')   
    min_neighbors = np.zeros([f.shape[0]])
    matrix = nx.to_numpy_matrix(G)
    for e in range(1,n):
        matrix += matrix ** e
    for i in f.index:
        neighbors = matrix[i]>0
        min_neighbors[i] = f[neighbors.tolist()[0]][column].min()
    f["min_neighbors"] = min_neighbors
    return f

def within_module_degree(G,f, column_communities = None, community_method = "label_propagation"):
    """ 
    the within_module_degree calculates: Zi = (ki-ks)/Ss 
    Ki = number of links between the node i and all the nodes of its cluster
    Ks = mean degree of the nodes in cluster s
    Ss = the standar desviation of the nodes in cluster s

    The within-module degree z-score measures how well-connected node i is to other nodes in the module.

    PAPER: Guimera, R., & Amaral, L. A. N. (2005). Functional cartography of complex metabolic networks. nature, 433(7028), 895.
    
    G: a networkx graph.
    f: a pandas dataframe.
    column_communities: a column of the dataframe with the communities for each node. If None, the communities will be estimated using metodo comunidades.
    community_method: method to calculate the communities in the graph G if they are not provided with columna_comunidades. 
    """
    if not(set(f.name) == set(G.nodes()) and len(f.name) == len(G.nodes())):
        raise ValueError('The number of nodes and the length of the datadrame should be the same.')   
    if column_communities == None:
        if community_method == "label_propagation":
            f = communities_label_propagation(G,f)
            column_communities = "communities_label_propagation"
        elif community_method == "greedy_modularity":
            f = communities_greedy_modularity(G,f)
            column_communities = "communities_greedy_modularity"
        else:
            raise ValueError('A clustering method should be provided.')
    
    z_df = pd.DataFrame(data = {'name': [], 'within_module_degree': [] }) 
    for comutnity in set(f[column_communities]):
        G2 = G.subgraph(f[f[column_communities] == comutnity]["name"].values)
        Ks = 2*len(G2.edges) / len(G2.nodes)
        Ss = np.std([i[1] for i in G2.degree()])
        z_df = pd.concat([z_df,pd.DataFrame(data = {'name': list(G2.nodes), 'within_module_degree': [(i[1]-Ks)/Ss for i in G2.degree()] }) ])
    
    f = pd.merge(f, z_df, on='name')
    return f

def participation_coefficient(G,f, column_communities = None, community_method = "label_propagation"):
    """
    the participation_coefficient calculates: Pi = 1- sum_s( (Kis/Kit)^2 ) 
    Kis = number of links between the node i and the nodes of the cluster s
    Kit = degree of the node i

    The participation coefficient of a node is therefore close to 1 if its links are uniformly distributed among all the modules and 0 if all its links are within its own module.
    
    PAPER: Guimera, R., & Amaral, L. A. N. (2005). Functional cartography of complex metabolic networks. nature, 433(7028), 895.
    
    G: a networkx graph.
    f: a pandas dataframe.
    columna_comunidades: a column of the dataframe with the communities for each node. If None, the communities will be estimated using metodo comunidades.
    metodo_comunidades: method to calculate the communities in the graph G if they are not provided with columna_comunidades. 
    """
    if not(set(f.name) == set(G.nodes()) and len(f.name) == len(G.nodes())):
        raise ValueError('The number of nodes and the length of the datadrame should be the same.')   
    if column_communities == None:
        if community_method == "label_propagation":
            f = communities_label_propagation(G,f)
            column_communities = "communities_label_propagation"
        elif community_method == "greedy_modularity":
            f = communities_greedy_modularity(G,f)
            column_communities = "communities_greedy_modularity"
        else:
            raise ValueError('A clustering method should be provided.')
    
    p_df = pd.DataFrame(data = {'name': f['name'], 'participation_coefficient': [1 for _ in f['name']] }) 
    for node in f['name']:
        Kit = len(G.edges(node))
        for comutnity in set(f[column_communities]): 
            Kis = len([edge for edge in G.edges(node) if edge[1] in f[ f[column_communities] == comutnity ]["name"]])
            p_df.loc[ p_df["name"] == node, 'participation_coefficient' ] -= ( Kis / Kit ) ** 2     
    f = pd.merge(f, p_df, on='name')
    return f

def node_embeddings(G,f,dim=20, walk_length=16, num_walks=100, workers=2):
    """
    Adds the embeddings of the nodes to the dataframe f.
    G: a networkx graph.
    f: a pandas dataframe.
    dim: the dimension of the embedding.

    Grover, A., & Leskovec, J. (2016, August). node2vec: Scalable feature learning for networks. In Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 855-864). ACM.
    """
    if not(set(f.name) == set(G.nodes()) and len(f.name) == len(G.nodes())):
        raise ValueError('The number of nodes and the length of the datadrame should be the same.')   
    from node2vec import Node2Vec
    node2vec = Node2Vec(G, dimensions=dim, walk_length=walk_length, num_walks=num_walks, workers=workers)
    model = node2vec.fit(window=10, min_count=1)
    
    embeddings_df = pd.DataFrame(columns = ['name']+['node_embeddings_'+str(i) for i in range(dim)])
    embeddings_df['name'] = f['name']
    for name in embeddings_df['name']:
        embeddings_df[embeddings_df['name'] == name] = [name] + list(model[str(name)])
    f = pd.merge(f, embeddings_df, on='name')
    return f

#Transformers
class Dumb(BaseEstimator, TransformerMixin):
    def __init__(self,m = 8):
        self.m = m
        print('a',self.m)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print('b',self.m)
        return X
    
class Replace(BaseEstimator, TransformerMixin):
    def __init__(self, value1,value2):
        self.value1 = value1
        self.value2 = value2

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.replace(self.value1, self.value2, regex=True)    
    
class DropName(BaseEstimator, TransformerMixin):
    """
    Drops the "name" column.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_prima = X.drop(['name'],axis=1)
        return X_prima    

class Graph_fuction(BaseEstimator, TransformerMixin):
    """
    Adds a column to the dataframe f with the result of the function for each node.
    G: a networkx graph. The names of the nodes should be incuded in the train dataframe.
    function: a python function that takes the graph G as input and outpus a column of the same length that the number of nodes in the graph.
    column_name: a string with the name of the column
    """
    def __init__(self, G, function, column_name = "Graph_fuction"):
        self.G = G
        self.function = function
        self.column_name = column_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        G_train = self.G.subgraph(X['name'].values) 
        column =  self.function(G_train)
        degree_df = pd.DataFrame(data = {'name': list(G_train.nodes()), self.column_name: column })  
        X_prima = pd.merge(X, degree_df, on='name')
        print(X_prima.columns)
        return X_prima

class Graph_features_fuction(BaseEstimator, TransformerMixin):
    """
    Adds a column to the dataframe f with the result of the function for each node.
    G: a networkx graph. The names of the nodes should be incuded in the train dataframe.
    function: a python function that takes the graph G as input and outpus a column of the same length that the number of nodes in the graph.
    column_name: a string with the name of the column
    """
    def __init__(self, G, function, column_name = "Graph_features_fuction"):
        self.G = G
        self.function = function
        self.column_name = column_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        G_train = self.G.subgraph(X['name'].values) 
        column =  self.function(G_train, X)
        degree_df = pd.DataFrame(data = {'name': list(G_train.nodes()), self.column_name: column })  
        X_prima = pd.merge(X, degree_df, on='name')
        print(X_prima.columns)
        return X_prima
    
class Degree(BaseEstimator, TransformerMixin):
    """
    Adds a column to the dataframe f with the degree of each node.
    G: a networkx graph. The names of the nodes should be incuded in the train dataframe.
    """
    def __init__(self, G):
        self.G = G

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        G_train = self.G.subgraph(X['name'].values) 
        degree_dic =  nx.degree_centrality(G_train)
        degree_df = pd.DataFrame(data = {'name': list(degree_dic.keys()), 'degree': list(degree_dic.values()) })  
        X_prima = pd.merge(X, degree_df, on='name')
        return X_prima
    
class Clustering(BaseEstimator, TransformerMixin):
    """
    Adds a column to the dataframe f with the degree of each node.
    G: a networkx graph. The names of the nodes should be incuded in the train dataframe.
    """
    def __init__(self, G):
        self.G = G

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        G_train = self.G.subgraph(X['name'].values) 
        clustering_dic = nx.clustering(G_train)
        clustering_df = pd.DataFrame(data = {'name': list(clustering_dic.keys()), 'clustering': list(clustering_dic.values()) })  
        X_prima = pd.merge(X, clustering_df, on='name')
        return X_prima    
    
class Centrality(BaseEstimator, TransformerMixin):
    """
    Adds a column to the dataframe f with the centrality of each node.
    G: a networkx graph. The names of the nodes should be incuded in the train dataframe.
    """
    def __init__(self, G):
        self.G = G

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        G_train = self.G.subgraph(X['name'].values) 
        centrality_dic = nx.degree_centrality(G_train)
        centrality_df = pd.DataFrame(data = {'name': list(centrality_dic.keys()), 'centrality': list(centrality_dic.values()) })  
        X_prima = pd.merge(X, centrality_df, on='name')
        return X_prima    

class Betweenness(BaseEstimator, TransformerMixin):
    """
    Adds a column to the dataframe f with the betweenness of each node.
    G: a networkx graph. The names of the nodes should be incuded in the train dataframe.
    """
    def __init__(self, G):
        self.G = G

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        G_train = self.G.subgraph(X['name'].values) 
        betweenness_dic = nx.betweenness_centrality(G_train)
        betweenness_df = pd.DataFrame(data = {'name': list(betweenness_dic.keys()), 'betweenness': list(betweenness_dic.values()) })  
        X_prima = pd.merge(X, betweenness_df, on='name')
        return X_prima    

class Pagerank(BaseEstimator, TransformerMixin):
    """
    Adds a column to the dataframe f with the pagerank of each node.
    G: a networkx graph. The names of the nodes should be incuded in the train dataframe.
    """
    def __init__(self, G):
        self.G = G

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        G_train = self.G.subgraph(X['name'].values) 
        pagerank_dic = nx.pagerank(G_train)
        pagerank_df = pd.DataFrame(data = {'name': list(pagerank_dic.keys()), 'pagerank': list(pagerank_dic.values()) })  
        X_prima = pd.merge(X, pagerank_df, on='name')
        return X_prima

    
class Communities_greedy_modularity(BaseEstimator, TransformerMixin):
    """
    Adds a column to the dataframe f with the comunity of each node.
    The comunitys are detected using greedy modularity.
    G: a networkx graph. The names of the nodes should be incuded in the train dataframe.
    
    It works with networkx vesion: '2.4rc1.dev_20190610203526'
    """    
    def __init__(self, G):
        self.G = G

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        G_train = self.G.subgraph(X['name'].values) 
        communities_dic = nx.algorithms.community.greedy_modularity_communities(G_train)
        communities_df = pd.DataFrame(data = {'name': [i for j in range(len(communities_dic)) for i in list(communities_dic[j])], 'communities_greedy_modularity': [j for j in range(len(communities_dic)) for i in list(communities_dic[j])] })  
        X_prima = pd.merge(X,communities_df, on='name')
        return X_prima    
    
    

class Communities_label_propagation(BaseEstimator, TransformerMixin):
    """
    Adds a column to the dataframe f with the comunity of each node.
    The comunitys are detected using glabel propagation.
    G: a networkx graph. The names of the nodes should be incuded in the train dataframe.
    
    It works with networkx vesion: '2.4rc1.dev_20190610203526'
    """    
    def __init__(self, G):
        self.G = G

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        G_train = self.G.subgraph(X['name'].values) 
        communities_gen = nx.algorithms.community.label_propagation_communities(G_train)
        communities_dic = [community for community in communities_gen]
        communities_df = pd.DataFrame(data = {'name': [i for j in range(len(communities_dic)) for i in list(communities_dic[j])], 'communities_label_propagation': [j for j in range(len(communities_dic)) for i in list(communities_dic[j])] })  
        X_prima = pd.merge(X,communities_df, on='name')
        return X_prima    
    

class Mean_neighbors(BaseEstimator, TransformerMixin):
    """
    Adds a column to the dataframe f with the mean value of its neigbors feature.
    G: a networkx graph. The names of the nodes should be incuded in the train dataframe.
    column: the column to which the mean is applied.
    n: neighbourhood order.

    """
    def __init__(self, G, column, n=1):
        self.G = G
        self.column = column
        self.n = n

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        G_train = self.G.subgraph(X['name'].values) 
        mean_neighbors = np.zeros([X.shape[0]])
        matrix = nx.to_numpy_matrix(G_train)
        for e in range(1,self.n):
            matrix += matrix ** e
        for i in range(X.shape[0]):
            neighbors = matrix[i]>0
            mean_neighbors[i] = X[neighbors.tolist()[0]][self.column].mean() 
        X_prima = X
        X_prima["mean_" + str(self.n) + "_neighbors_" + str(self.column)] = mean_neighbors
        return X_prima
    
    
class Std_neighbors(BaseEstimator, TransformerMixin):
    """
    Adds a column to the dataframe f with the standar desviation value of its neigbors feature.
    G: a networkx graph. The names of the nodes should be incuded in the train dataframe.
    column: the column to which the mean is applied.
    n: neighbourhood order.

    """
    def __init__(self, G, column, n=1):
        self.G = G
        self.column = column
        self.n = n

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        G_train = self.G.subgraph(X['name'].values) 
        
        std_neighbors = np.zeros([X.shape[0]])
        matrix = nx.to_numpy_matrix(G_train)
        for e in range(1,self.n):
            matrix += matrix ** e
        for i in range(X.shape[0]):
            neighbors = matrix[i]>0
            std_neighbors[i] = X[neighbors.tolist()[0]][self.column].std()
        X_prima = X
        X_prima["std_" + str(self.n) + "_neighbors_" + str(self.column)] = std_neighbors
        return X_prima    
    
    
class Max_neighbors(BaseEstimator, TransformerMixin):
    """
    Adds a column to the dataframe f with the maximum value of its neigbors feature.
    G: a networkx graph. The names of the nodes should be incuded in the train dataframe.
    column: the column to which the mean is applied.
    n: neighbourhood order.

    """
    def __init__(self, G, column, n=1):
        self.G = G
        self.column = column
        self.n = n

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        G_train = self.G.subgraph(X['name'].values) 
        
        max_neighbors = np.zeros([X.shape[0]])
        matrix = nx.to_numpy_matrix(G_train)
        for e in range(1,self.n):
            matrix += matrix ** e
        for i in range(X.shape[0]):
            neighbors = matrix[i]>0
            max_neighbors[i] = X[neighbors.tolist()[0]][self.column].max()
        X_prima = X
        X_prima["max_" + str(self.n) + "_neighbors_" + str(self.column)] = max_neighbors
        return X_prima       

class Min_neighbors(BaseEstimator, TransformerMixin):
    """
    Adds a column to the dataframe f with the minimum value of its neigbors feature.
    G: a networkx graph. The names of the nodes should be incuded in the train dataframe.
    column: the column to which the mean is applied.
    n: neighbourhood order.

    """
    def __init__(self, G, column, n=1):
        self.G = G
        self.column = column
        self.n = n

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        G_train = self.G.subgraph(X['name'].values) 
        
        min_neighbors = np.zeros([X.shape[0]])
        matrix = nx.to_numpy_matrix(G_train)
        for e in range(1,self.n):
            matrix += matrix ** e
        for i in range(X.shape[0]):
            neighbors = matrix[i]>0
            min_neighbors[i] = X[neighbors.tolist()[0]][self.column].min()
        X_prima = X
        X_prima["min_" + str(self.n) + "_neighbors_" + str(self.column)] = min_neighbors
        return X_prima          
    

    
class Within_module_degree(BaseEstimator, TransformerMixin):
    """
    the within_module_degree calculates: Zi = (ki-ks)/Ss 
    Ki = number of links between the node i and all the nodes of its cluster
    Ks = mean degree of the nodes in cluster s
    Ss = the standar desviation of the nodes in cluster s

    The within-module degree z-score measures how well-connected node i is to other nodes in the module.

    PAPER: Guimera, R., & Amaral, L. A. N. (2005). Functional cartography of complex metabolic networks. nature, 433(7028), 895.
    
    G: a networkx graph.
    column_communities: a column of the dataframe with the comunities for each node. If None, the comunities will be estimated using metodo communityes.
    """
    def __init__(self, G, column_communities):
        self.G = G
        self.column_communities = column_communities

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        G_train = self.G.subgraph(X['name'].values) 
        z_df = pd.DataFrame(data = {'name': [], 'within_module_degree': [] }) 
        for community in set(X[self.column_communities]):
            G2 = G_train.subgraph(X[X[self.column_communities] == community]["name"].values)
            Ks = 2*len(G2.edges) / len(G2.nodes)
            Ss = np.std([i[1] for i in G2.degree()])
            z_df = pd.concat([z_df,pd.DataFrame(data = {'name': list(G2.nodes), 'within_module_degree': [np.divide(i[1]-Ks, Ss) for i in G2.degree()] }) ])
        
        X_prima = pd.merge(X, z_df, on='name')
        return X_prima


 
class Participation_coefficient(BaseEstimator, TransformerMixin):
    """
    The participation_coefficient calculates: Pi = 1- sum_s( (Kis/Kit)^2 ) 
    Kis = number of links between the node i and the nodes of the cluster s
    Kit = degree of the node i

    The participation coefficient of a node is therefore close to 1 if its links are uniformly distributed among all the modules and 0 if all its links are within its own module.
    
    PAPER: Guimera, R., & Amaral, L. A. N. (2005). Functional cartography of complex metabolic networks. nature, 433(7028), 895.
    
    G: a networkx graph.
    column_communities: a column of the dataframe with the comunities for each node. If None, the comunities will be estimated using metodo communityes.
    """
    def __init__(self, G, column_communities):
        self.G = G
        self.column_communities = column_communities

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        G_train = self.G.subgraph(X['name'].values) 
        p_df = pd.DataFrame(data = {'name': X['name'], 'participation_coefficient': [1 for _ in X['name']] }) 
        for node in X['name']:
            Kit = len(G_train.edges(node))
            for community in set(X[self.column_communities]): 
                Kis = len([edge for edge in G_train.edges(node) if edge[1] in X[ X[self.column_communities] == community ]["name"]])
                p_df.loc[ p_df["name"] == node, 'participation_coefficient' ] -= np.divide(Kis, Kit) ** 2     
        X_prima = pd.merge(X, p_df, on='name')  
        return X_prima
 
class Node_embeddings(BaseEstimator, TransformerMixin):
    """
    Adds the embeddings of the nodes to the dataframe f.
    G: a networkx graph.
    dim: the dimension of the embedding.

    Grover, A., & Leskovec, J. (2016, August). node2vec: Scalable feature learning for networks. In Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 855-864). ACM.
    """
    def __init__(self, G,dim=20, walk_length=16, num_walks=100, workers=2):
        self.G = G
        self.dim = dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = workers
        

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        from node2vec import Node2Vec
        node2vec = Node2Vec(self.G, dimensions=self.dim, walk_length=self.walk_length, num_walks=self.num_walks, workers=self.workers)
        model = node2vec.fit(window=10, min_count=1)
    
        embeddings_df = pd.DataFrame(columns = ['name']+['node_embeddings_'+str(i) for i in range(self.dim)])
        embeddings_df['name'] = X['name']
        for name in embeddings_df['name']:
            embeddings_df[embeddings_df['name'] == name] = [name] + list(model[str(name)])
        X_prima = pd.merge(X, embeddings_df, on='name')
        return X_prima        
