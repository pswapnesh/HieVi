# HieVi: Protein Large Language Model for proteome-based phage clustering
Swapnesh Panigrahi, Mireille Ansaldi, Nicolas Ginet

https://www.biorxiv.org/content/10.1101/2024.12.17.627486v1

### Abstract
Viral taxonomy is a challenging task due to the propensity of viruses for recombination. Recent updates from the ICTV and advancements in proteome-based clustering tools highlight the need for a unified framework to organize bacteriophages (phages) across multiscale taxonomic ranks, extending beyond genome-based clustering. Meanwhile, self-supervised large language models, trained on amino acid sequences, have proven effective in capturing the structural, functional, and evolutionary properties of proteins. Building on these advancements, we introduce HieVi, which uses embeddings from a protein language model to define a vector representation of phages and generate a hierarchical tree of phages. Using the INPHARED dataset of 24,362 complete and annotated viral genomes, we show that in HieVi, a multi-scale taxonomic ranking emerges that aligns well with current ICTV taxonomy. We propose that this method, unique in its integration of protein language models for viral taxonomy, can encode phylogenetic relationships, at least up to the family level. It therefore offers a valuable tool for biologists to discover and define new phage families while unraveling novel evolutionary connections.

### Interactive Phage Atlas
https://pswapnesh.github.io/HieVi/HieVi_UMAP.html

### Colab notebook
[Colab notebook with esm2-650m model](https://colab.research.google.com/drive/1d9tzxLrnHo9eUAQoaGDQyisO4q02ZtFX?usp=sharing)

### Download HieVi phage representations for INPHARED 1Sept2024 dataset
[esm2-3b]()
[esm2-650m]()
Save it somewhere, say ```./hievi_inphared_1sept24/```.

## Install Required Packages and Data

### Package Installation
```bash
# Install ESM (Evolutionary Scale Modeling) from Facebook Research's GitHub (main branch)
pip install git+https://github.com/facebookresearch/esm.git

# Install HDBSCAN clustering, pandas, and Biopython
pip install hdbscan pandas biopython

# Install FAISS for similarity search (CPU version)
pip install faiss-cpu

# Install specific version of Zarr for array storage
pip install zarr==2.18.4

# Clone HieVi repository (hievi branch)
git clone --branch hievi https://github.com/pswapnesh/HieVi.git

conda install anaconda::pydot
conda install -c conda-forge plotly

# Workflow
### Step 1: Generate a Multifasta Proteome File
Create a multifasta file containing your proteome using your preferred method. Make sure the unique names of phages donot contain underscore '_'. 

### Step 2: Format the Multifasta File
Ensure your multifasta file (e.g., proteome.faa) is properly formatted:
The accessions (or unique names of phages) must not contain underscores (_).
Unique names and protein number should be separated by underscore, for example,

```
PHAGE-01_0001
amino_acid_sequence
>PHAGE-01_0002
amino_acid_sequence
>PHAGE-02_0001
amino_acid_sequence
```

### Step 3: Generate Phage Representations
Use the following command to generate phage representations and output them to a specified directory. Ensure all paths are provided in full:
```python GenPhageRepresentationsESM2.py "experiment_name" "/path/to/outputfolder/" "/path/to/proteome.faa" "650m" "mean"```

**Notes:**
- In this example, the model "650m" is used, which consumes less GPU memory.
- If GPU memory is not a constraint, you can use the "3b" model instead.
- The output phage representations will be saved as a Zarr file at: ```/path/to/outputfolder/{experiment_name}_{model}.zarr```

### Step 4a: Generate hierarchical tree without reference dataset.
To compare only the phages for which the phage representations were generated, you may use the following code.
```
import zarr 
from utils.network_utils import *
zarr_path = "/path/to/outputfolder/{experiment_name}_{model}.zarr"
zarr_store = zarr.open(zarr_path,'r')
vectors = zarr_store['vectors_mean'][:]*1.0
accessions = zarr_store['accessions'][:]

query_df = pd.DataFrame({"Accession": accessions})

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
query_df["HieVi_granular_cluster"] = clusterer.labels_

# Create and save network
G = make_network(clusterer, annotation_df)
nx.write_gexf(G, zarr_path[:-5] + "_HieVi.gexf")
```

### Step 4b: Generate a hierarchical tree with reference dataset
A download database vectors corresponding to the model (in this case 650m)
To create an approximate hierarchical tree for the new proteomes, run the following command:
```python PlaceNewProteome.py "/path/to/HieVi_INPHARED_database_650m.zarr" "/path/to/outputfolder/{experiment_name}_{model}.zarr"```

**Notes:**
- The output files and network will be generated in:
```/path/to/outputfolder/```


## Visualization
### Step 1: Open Cytoscape
Launch Cytoscape on your system.

### Step 2: Install Required Apps/Plugins
Ensure the following apps/plugins are installed:
GEXF Exporter (gexf-app)
yFiles

### Step 3: Load and Visualize the Data
Load the GEXF file generated in the previous step.
Apply the yFiles hierarchical layout to organize the network.
Use the search function to find your phages by their accession names.

