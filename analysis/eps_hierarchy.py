import hdbscan 
import zarr
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import metrics# import adjusted_rand_score, adjusted_mutual_info_score,silhouette_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx

def stable_eps_detect(vectors,eps_range = np.linspace(8e-3,4e-2,1024)):
    # Perform clustering
    dist_scaled = euclidean_distances(vectors).astype("double")
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

    #
    stability_scores = []
    labels_prev = clusterer.dbscan_clustering(cut_distance=eps_range[0],min_cluster_size=2)
    for eps in tqdm(eps_range):
        labels = clusterer.dbscan_clustering(cut_distance=eps,min_cluster_size=2)
        mi = metrics.mutual_info_score(labels_prev,labels)
        # idx = labels > -1
        # sil_score = metrics.silhouette_score(vectors[idx],labels[idx])
        sil_score = 0
        stability_scores += [[mi,sil_score]]
        labels_prev = labels
    stability_scores = np.array(stability_scores)    
    best_stability_index = np.argmax(stability_scores[:,0]) # use mutual infor
    stable_eps = eps_range[best_stability_index]

    #exemplars

    clusterer = hdbscan.HDBSCAN(
    min_cluster_size=2,
    min_samples=1,
    cluster_selection_epsilon=float(stable_eps),
    allow_single_cluster=False,
    cluster_selection_method="leaf",
    metric="euclidean",
    gen_min_span_tree=True
    )
    clusterer.fit(vectors)
    exemplars=clusterer.exemplars_
    vectors_exemplars = np.concatenate([[e[0]] for e in exemplars],axis = 0)

    return stable_eps,vectors_exemplars,stability_scores
