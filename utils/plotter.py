
import networkx as nx
import plotly.graph_objects as go
import pydot
import random
import plotly.colors as pc
from networkx.drawing.nx_pydot import graphviz_layout

def plot_hierarchical_graph(G,layout = "radial"):
    if not isinstance(G, nx.DiGraph):
        raise ValueError("Graph must be a directed graph (nx.DiGraph)")

    # Ensure all node names are strings
    G = nx.relabel_nodes(G, {node: str(node) for node in G.nodes()})

    # Fix attributes: Rename "name" -> "name_old" if it exists
    for node in G.nodes():
        if "name" in G.nodes[node]:
            G.nodes[node]["name_old"] = G.nodes[node].pop("name")

    # Remove problematic attributes (lists, dicts, tuples)
    for node in G.nodes():
        for key, value in list(G.nodes[node].items()):
            if isinstance(value, (list, dict, tuple)):
                del G.nodes[node][key]

    # Convert all attributes to strings
    for node in G.nodes():
        for key in G.nodes[node]:
            G.nodes[node][key] = str(G.nodes[node][key])

    try:
        if layout =="radial":
            pos = graphviz_layout(G, prog="twopi")
        else:
            pos = graphviz_layout(G, prog="dot")
    except Exception as e:
        print("Error: Graphviz failed!", e)
        return

    # Get unique genera and assign random colors
    unique_genera = set(G.nodes[node].get("Genus", "Unclassified") for node in G.nodes())
    color_map = {genus: random.choice(pc.sequential.Electric) for genus in unique_genera} #DEFAULT_PLOTLY_COLORS

    # Edge positions
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, line=dict(width=1, color='black'),
        hoverinfo='none', mode='lines'
    )

    # Node positions, colors, tooltips
    node_x, node_y, hover_texts = [], [], []
    node_colors, node_borders = [], []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        # Get Genus attribute
        genus = G.nodes[node].get("Genus", "Unclassified")

        # Assign colors: Red border for "nan", orange for "Unclassified", others get unique colors
        if genus == "nan":
            node_colors.append("red")  # Node fill color
            node_borders.append("red")  # Node border color
        elif genus == "Unclassified":
            node_colors.append("gray")  # Unclassified nodes
            node_borders.append("black")
        else:
            node_colors.append(color_map.get(genus, "gray"))  # Assign random colormap
            node_borders.append("black")

        # Tooltip text
        attrs = G.nodes[node]
        tooltip = "<br>".join([f"{key}: {value}" for key, value in attrs.items()])
        hover_texts.append(tooltip)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers',
        marker=dict(size=15, color=node_colors,
                    line=dict(width=2, color=node_borders)),  # Border color
        hoverinfo="text", hovertext=hover_texts
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title="HieVi",
                        showlegend=False, hovermode="closest",
                        margin=dict(b=0, t=50, l=0, r=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))

    return fig  # Return the figure for further use