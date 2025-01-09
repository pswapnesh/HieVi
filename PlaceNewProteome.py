import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import networkx as nx
import zarr
from tqdm import tqdm
from utils.network_utils import *
import faiss
from utils.zarr_utils import *
import hdbscan
from sklearn.metrics.pairwise import cosine_distances,euclidean_distances
import os

## inputs
query_proteome_path = "Metavirome_INRAE_HieVi_layer35_3b_mean_normalized.zarr"
embeddings_database_path = "./db/HieVi_INPHARED_db.zarr"
faiss_index_path = "./db/HieVi_INPHARED_faiss_index.bin"
annotations_file_path = "HieVi_INPHARED_ordered_annotation.csv"
k = 512
layer = 35

# read files
annotations_df = pd.read_csv(annotations_file_path)
faiss_index = faiss.read_index(faiss_index_path)


phage_ids = annotations_df['Accession'].values
#zarr_store = zarr.open(embeddings_database_path, mode='r')

# load query phage representations
query_zarr_store = zarr.open(query_proteome_path, mode='r')
query_phage_ids = np.array([accession for accession in tqdm(query_zarr_store)])
# Initialize the reader
reader = AccessionZarrReader(query_zarr_store)
# List all accessions in the Zarr store
query_phage_ids = np.array(reader.list_accessions())
query_mprs = np.array([reader.read_accession(accession)["layer_embeddings"][str(layer)] for accession in tqdm(query_phage_ids)])

# find k nearest neighbours
index = faiss.read_index(faiss_index_path)
k = 512  # Number of nearest neighbors
distances, indices = index.search(query_mprs, k)
indices = np.unique(np.ravel(np.concatenate(indices)))
print("Nearest neighbor search completed.")

# load nearest neighbout representations
# Initialize the reader
reader = AccessionZarrReader(embeddings_database_path)
# List all accessions in the Zarr store
phage_ids = np.array(reader.list_accessions())
mprs = np.array([reader.read_accession(accession)["layer_embeddings"][str(layer)] for accession in tqdm(phage_ids[indices])])

df = pd.DataFrame({'Accession':np.unique(query_phage_ids)}) # possible error here
annotations_df = pd.concat([annotations_df.loc[indices],df],axis = 0)
mprs = np.concatenate((mprs,query_mprs),axis = 0)



dist_scaled = euclidean_distances(mprs).astype('float')
clusterer = hdbscan.HDBSCAN(min_cluster_size = 2,n_jobs = 32,min_samples = 1,allow_single_cluster = False,cluster_selection_method = "leaf",metric = 'precomputed',gen_min_span_tree=True)
clusterer.fit(dist_scaled)

df = annotations_df#[uids]
df['HieVi_l0'] = clusterer.labels_
max_score_eps = [0.015355000954658212,0.016955190956681764,0.02221874450992837]
for i,eps in enumerate(max_score_eps):
    annotations_df['HieVi_l' + str(i+1)] = clusterer.dbscan_clustering(cut_distance=float(eps), min_cluster_size=2)

node_properties = ['Accession', 'Virus_Description', 'Virus_Genome_size',
       'Virus_molGC_(%)', 'Virus_Number_CDS', 'Realm', 'Kingdom', 'Phylum',
       'Class', 'Order', 'Family', 'Subfamily', 'Genus', 'Lowest_taxa', 'tRNAs', 'VC_cluster',
       'VC', 'VC Status', 'VC_Size', 'VC_Subcluster', 'VC_Subcluster_Size',
       'VC_number', 'VC_subcluster']

node_properties += ['HieVi_l0'] + ['HieVi_l' + str(i+1) for i,eps in enumerate(max_score_eps)]

df = df[node_properties]
df = df.fillna('Unclassified')
df = df.astype(str)
G = make_network(clusterer,df)

nx.write_gexf(G, os.path.join(os.path.dirname(query_proteome_path),"output.gexf"))
print('Done. Tree saved in the input folder.')