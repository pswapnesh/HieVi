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

# def traverse_graph_and_get_accessions(G, query_accessions, n):
#     """
#     Traverse the graph G up and down by n steps starting from nodes with query_accessions
#     and return the unique accessions of nodes in the selected subgraph.
    
#     Parameters:
#     G (networkx.Graph): The input graph with node attributes.
#     query_accessions (list): List of accessions to use as starting points.
#     n (int): Number of steps to traverse.
    
#     Returns:
#     set: Unique accessions of nodes in the traversed subgraph.
#     """
#     # Find starting nodes based on query accessions
#     start_nodes = [node for node,acc in nx.get_node_attributes(G, "Accession").items() if acc in query_accessions]
    
#     if not start_nodes:
#         return set()  # No matching nodes found
    
#     # Perform BFS traversal up to n steps
#     visited = set()
#     for start_node in start_nodes:
#         for node in nx.single_source_shortest_path_length(G, start_node, cutoff=n):
#             visited.add(node)
    
#     # Collect unique accessions from visited nodes
#     unique_accessions = {
#         G.nodes[node].get('Accession') for node in visited if 'Accession' in G.nodes[node]
#     }
    
#     return unique_accessions

def traverse_graph_and_get_accessions(graph, query_accessions, n=16, attr_label='Accession'):
    """
    For each query accession, find the subtree of nodes (both ancestors and descendants)
    in the directed acyclic graph (DAG) and return the unique accessions in those subtrees,
    with a fixed number of upstream and downstream nodes.
    
    Args:
        graph (networkx.DiGraph): The directed acyclic graph (DAG).
        query_accessions (list): List of query accessions to start the traversal from.
        n (int): Number of steps to traverse up and down the graph.
        attr_label (str): The attribute key for accessions in the graph nodes.
    
    Returns:
        set: A set of accessions of nodes in the subgraphs (subtrees) for each query accession.
    """
    all_accessions = set()

    # Create a dictionary that maps accession values to node IDs    
    node_to_accession = nx.get_node_attributes(graph, attr_label)
    accession_to_node = {acc: node for node, acc in node_to_accession.items() if acc is not None}

    # Get topologically sorted list of nodes
    topologically_sorted = list(nx.algorithms.topological_sort(graph))

    # Iterate through each query accession
    for query in query_accessions:
        # Check if the query accession exists in the accession_to_node dictionary
        if query in accession_to_node:
            node = accession_to_node[query]
            idx = topologically_sorted.index(node)
            
            # Define the range to gather upstream and downstream nodes
            # Get up to `n` upstream nodes (ancestors)
            upstream_accessions = set()
            for i in range(idx, max(0, idx - n), -1):
                accession = node_to_accession.get(topologically_sorted[i])
                if accession is not None:
                    upstream_accessions.add(accession)

            # Get up to `n` downstream nodes (descendants)
            downstream_accessions = set()
            for i in range(idx + 1, min(len(topologically_sorted), idx + 1 + n)):
                accession = node_to_accession.get(topologically_sorted[i])
                if accession is not None:
                    downstream_accessions.add(accession)

            # Add accessions to the final set
            all_accessions.update(upstream_accessions)
            all_accessions.update(downstream_accessions)
        else:
            # If the query accession is not found in the graph, log a warning
            print(f"Warning: Accession '{query}' not found in the graph.")

    return all_accessions