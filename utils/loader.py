import zarr
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def retrieve_layer(accession, layer, zarr_store):
    layer_data = zarr_store[accession]['layer_' + str(layer)][:]
    #counter_data = zarr_store[accession].attrs["count"]
    return layer_data#/ counter_data
def retrieve_counts(accession, zarr_store):
    return zarr_store[accession].attrs["count"]

def fast_mprs_parallel(zarr_store, phage_ids, layer, max_workers=8):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(retrieve_layer, accession, layer, zarr_store) for accession in phage_ids]
        
        # Collect the results from the futures
        results = [f.result() for f in tqdm(futures)]

    # Combine the results into a NumPy array
    mprs = np.array(results)
    return mprs

def fast_counts_parallel(zarr_store, phage_ids, layer, max_workers=8):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(retrieve_counts, accession, zarr_store) for accession in phage_ids]
        
        # Collect the results from the futures
        results = [f.result() for f in tqdm(futures)]

    # Combine the results into a NumPy array
    glengths = np.array(results)
    return glengths

def match_annotation_to_processed_ids(phage_ids,annotation_file):

    annotations = pd.read_csv(annotation_file,sep = '\t')
    annotations_indexed = annotations.set_index('Accession')

    # # Create a new DataFrame with 'ids' as the index
    new_df = pd.DataFrame(index=phage_ids, columns=annotations_indexed.columns)

    # # Fill the new DataFrame with values from 'annotations_indexed' where possible
    new_df = new_df.fillna(annotations_indexed.reindex(new_df.index))

    # # Fill missing values with 'Unclassified'
    new_df = new_df.fillna('Unclassified')

    # # Optionally, reset the index if you prefer a regular DataFrame
    new_df = new_df.reset_index().rename(columns={'index': 'Accession'})
    #annotations.head()
    print("new df, number of phages, annotation")
    print(len(new_df),len(phage_ids),len(annotations))
    return new_df