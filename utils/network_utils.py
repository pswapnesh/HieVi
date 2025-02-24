import networkx as nx
import numpy as np


def make_network(hdb, df, wt_nan=1e12,min_lambda = -1):
    """
    Constructs a directed graph from HDBSCAN's condensed tree.

    Parameters:
        hdb: HDBSCAN object with condensed_tree_ attribute.
        df (pd.DataFrame): Dataframe containing node attributes.
        wt_nan (float): Weight assigned when lambda_val is NaN.

    Returns:
        nx.DiGraph: A directed graph with nodes and edges from the condensed tree.
    """
    G = nx.DiGraph()

    # Add all nodes from the condensed tree
    all_nodes = set(hdb.condensed_tree_._raw_tree['parent']).union(set(hdb.condensed_tree_._raw_tree['child']))
    G.add_nodes_from(all_nodes)

    # Add edges with weights
    for row in hdb.condensed_tree_._raw_tree:
        parent, child, lambda_val = int(row['parent']), int(row['child']), row['lambda_val']
        weight = lambda_val if np.isfinite(lambda_val) else wt_nan
        if min_lambda > 0:
            if lambda_val > min_lambda:                
                G.add_edge(parent, child, weight=weight, distance=1 / weight if weight != 0 else wt_nan)
        else:
            G.add_edge(parent, child, weight=weight, distance=1 / weight if weight != 0 else wt_nan)

    # Assign attributes to existing nodes
    for i, (idx, row) in enumerate(df.iterrows()):
        node_id = int(i)
        if node_id in G.nodes:
            G.nodes[node_id].update(row.to_dict())

    return G

def find_predecessor_and_leaves(G, query_accession,distance_to_walk=1):
    """
    Given a NetworkX tree-like graph G where only leaves have an 'accession' attribute, 
    find the node with the given query accession. Then, find its immediate predecessor (parent)
    and return the accession values of all leaves under that predecessor.

    Parameters:
    G (networkx.DiGraph): The tree-like directed graph.
    query_accession (str): The accession attribute to search for (only in leaves).

    Returns:
    list: Accession values of leaves under the predecessor.
    """
    # Find the node with the given accession (only leaves have accession)
    node = next((n for n, data in G.nodes(data=True) if data.get('Accession') == query_accession), None)
    
    if node is None:
        return []  # Query node not found
    
    # Find the immediate predecessor (parent)
    predecessors = list(G.predecessors(node))
    
    if not predecessors:
        return []  # No predecessor (root node case)
    
    predecessor = predecessors[0]
    
    # Find all leaves under the predecessor
    def is_leaf(n):
        return G.out_degree(n) == 0  # No outgoing edges (no children)
    
    decendants = []
    for i in range(distance_to_walk):
        decendants += nx.descendants_at_distance(G, predecessor,i+1)
    leaves = [n for n in  decendants if is_leaf(n)] 
    # Get accession values of the leaves
    leaf_accessions = [G.nodes[n]['Accession'] for n in leaves if 'Accession' in G.nodes[n]]
    
    return leaf_accessions
