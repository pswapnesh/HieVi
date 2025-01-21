import networkx as nx
import numpy as np

def make_network(hdb,df, wt_nan = 1e9):
    G = nx.DiGraph()
    for row in hdb._condensed_tree:
        if np.isfinite(row['lambda_val']):
            w = row['lambda_val']
        else:
            w = wt_nan
        G.add_edge(int(row['parent']), int(row['child']),weight = w)

    # Assign attributes to existing nodes
    for i,(idx, row) in enumerate(df.iterrows()):
        nodeId = int(i)
        G.nodes(data=True)[nodeId].update(row.to_dict())
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