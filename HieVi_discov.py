import argparse
import pandas as pd
import numpy as np
import networkx as nx
import zarr
import faiss
import hdbscan
from sklearn.metrics.pairwise import euclidean_distances
from utils.network_utils import traverse_graph_and_get_accessions, make_network

def get_argument_parser():
    """
    Create and return the argument parser.
    """
    parser = argparse.ArgumentParser(description="Process and cluster proteome representations.")
    parser.add_argument("--experiment_name", type=str, required=True, help="Experiment name prefix.")
    parser.add_argument("--annotation_path", type=str, required=True, help="Path to annotation CSV.")
    parser.add_argument("--query_zarr_path", type=str, required=True, help="Path to query Zarr store.")
    parser.add_argument("--db_zarr_path", type=str, required=True, help="Path to database Zarr store.")
    parser.add_argument("--faiss_index_path", type=str, required=True, help="Path to FAISS index.")
    parser.add_argument("--hievi_tree_path", type=str, required=True, help="Path to HieVi tree in GEXF format.")
    parser.add_argument("--output_folder", type=str, required=True, help="Output folder path.")
    parser.add_argument("--k_neighbours", type=int, default=2, help="Number of neighbors for FAISS search.")
    parser.add_argument("--n_levels", type=int, default=3, help="Number of levels for graph traversal.")
    return parser

def main(args):
    """
    Main function to process and cluster proteome representations.
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    # Setup paths and prefix
    name_prefix = args.experiment_name + "_HieVi_"
    
    # Load annotation data and HieVi tree
    annotation_df = pd.read_csv(args.annotation_path)
    
    hievi_tree = nx.read_gexf(args.hievi_tree_path)

    
    # Load query Zarr data
    query_zarr_store = zarr.open(args.query_zarr_path, mode="r")
    query_phage_ids = query_zarr_store["accessions"][:]
    query_mprs = query_zarr_store["vectors_mean"][:]
    
    # Load FAISS index and search for nearest neighbors
    index = faiss.read_index(args.faiss_index_path)
    distances, indices = index.search(query_mprs, args.k_neighbours)
    indices = np.unique(np.ravel(indices))
    print(f"Nearest neighbor search completed. Found {len(indices)} unique neighbors.")
    
    
    
    # Load database accessions
    db_zarr_store = zarr.open(args.db_zarr_path, mode="r")
    phage_ids = db_zarr_store["accessions"][indices]
    mprs = db_zarr_store["vectors_mean"][indices]
    annotation_df= annotation_df[annotation_df["Accession"].isin(phage_ids)]
    annotation_df = annotation_df.set_index("Accession").loc[phage_ids].reset_index()

    
    # Get nearest accessions in tree    
    nearest_accessions = annotation_df[annotation_df["Accession"].isin(phage_ids)]

    #nearest_accessions_in_tree = traverse_graph_and_get_accessions(hievi_tree, nearest_accessions["Accession"].values,args.n_levels)
    #indices = annotation_df.index[annotation_df["Accession"].isin(nearest_accessions_in_tree)].tolist()
        

    print(f"Extracted {len(indices)} relevant phages. Re-clustering.")
    
    # Combine query and database data
    query_df = pd.DataFrame({"Accession": query_phage_ids})
    annotation_df = pd.concat([nearest_accessions, query_df], axis=0)
    mprs = np.concatenate((mprs, query_mprs), axis=0)
    
    # Perform clustering
    dist_scaled = euclidean_distances(mprs).astype("double")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=2,
        n_jobs=32,
        min_samples=1,
        allow_single_cluster=False,
        cluster_selection_method="leaf",
        metric="precomputed",
        gen_min_span_tree=True
    )
    clusterer.fit(dist_scaled)
    annotation_df["HieVi_l0"] = clusterer.labels_
    
    # Additional clustering at multiple levels
    max_score_eps = [0.015355000954658212, 0.016955190956681764, 0.02221874450992837]
    for i, eps in enumerate(max_score_eps):
        annotation_df[f"HieVi_l{i+1}"] = clusterer.dbscan_clustering(
            cut_distance=float(eps), min_cluster_size=2
        )
    
    # Prepare final DataFrame
    node_properties = [
        "Accession", "Virus_Description", "Virus_Genome_size",
        "Virus_molGC_(%)", "Virus_Number_CDS", "Realm", "Kingdom", "Phylum",
        "Class", "Order", "Family", "Subfamily", "Genus", "Lowest_taxa", "tRNAs",
        "VC_cluster", "VC", "VC Status", "VC_Size", "VC_Subcluster",
        "VC_Subcluster_Size", "VC_number", "VC_subcluster"
    ]
    node_properties += [f"HieVi_l{i}" for i in range(len(max_score_eps) + 1)]
    annotation_df = annotation_df[node_properties].fillna("Unclassified").astype(str)
    
    # Create and save network
    G = make_network(clusterer, annotation_df)
    nx.write_gexf(G, f"{args.output_folder}/{name_prefix}output.gexf")
    print(f"Network saved to {args.output_folder}/{name_prefix}output.gexf")

if __name__ == "__main__":
    parser = get_argument_parser()
    args = parser.parse_args()
    main(args)
