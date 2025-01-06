import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import networkx as nx
import os
import zarr
from utils.loader import *


experiment_name = "newV"
path_ordered_inphrared_accessions ='metadata/Ordered_INPHRARED_Accessions.csv'
path_ordered_inphrared_annotations ='metadata/Ordered_INPHRARED_Annotations.csv'
output_folder = "/media/microscopie-lcb/swapnesh/protein/embeddings/phages/NewGenomes/"

name_prefix = experiment_name + '_HieVi_'

## load processed INPHRARED accessions in order
db_accessions = pd.read_csv(path_ordered_inphrared_accessions)
db_annotations = pd.read_csv(path_ordered_inphrared_annotations)


# load proteome representations
query_zarr_store_path = os.path.join(output_folder,name_prefix + 'phage_representations.zarr')
query_zarr_store = zarr.open(query_zarr_store_path, mode='r')
query_phage_ids = np.array([accession for accession in tqdm(query_zarr_store)])

layer = 35
query_mprs = fast_mprs_parallel(query_zarr_store,query_phage_ids, layer, max_workers=32)
glengths = fast_counts_parallel(query_zarr_store, query_phage_ids, layer, max_workers=32)
query_mprs = np.array([m/(gl*1.0) for m,gl in zip(query_mprs,glengths)])



