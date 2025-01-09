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