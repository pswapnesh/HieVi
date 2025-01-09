# HieVi: Protein Large Language Model for proteome-based phage clustering
Swapnesh Panigrahi, Mireille Ansaldi, Nicolas Ginet

https://www.biorxiv.org/content/10.1101/2024.12.17.627486v1

### Abstract
Viral taxonomy is a challenging task due to the propensity of viruses for recombination. Recent updates from the ICTV and advancements in proteome-based clustering tools highlight the need for a unified framework to organize bacteriophages (phages) across multiscale taxonomic ranks, extending beyond genome-based clustering. Meanwhile, self-supervised large language models, trained on amino acid sequences, have proven effective in capturing the structural, functional, and evolutionary properties of proteins. Building on these advancements, we introduce HieVi, which uses embeddings from a protein language model to define a vector representation of phages and generate a hierarchical tree of phages. Using the INPHARED dataset of 24,362 complete and annotated viral genomes, we show that in HieVi, a multi-scale taxonomic ranking emerges that aligns well with current ICTV taxonomy. We propose that this method, unique in its integration of protein language models for viral taxonomy, can encode phylogenetic relationships, at least up to the family level. It therefore offers a valuable tool for biologists to discover and define new phage families while unraveling novel evolutionary connections.

### Interactive Phage Atlas
https://pswapnesh.github.io/HieVi/HieVi_UMAP.html

# Workflow
### Step 1: Generate a Multifasta Proteome File
Create a multifasta file containing your proteome using your preferred method, or use the provided script:
```python predict_proteome.py```

### Step 2: Format the Multifasta File
Ensure your multifasta file (e.g., proteome.faa) is properly formatted:
The accessions (or unique names of phages) must not contain underscores (_).

### Step 3: Generate Phage Representations
Use the following command to generate phage representations and output them to a specified directory. Ensure all paths are provided in full:
```python GenPhageRepresentations.py "experiment_name" "/path/to/outputfolder/" "/path/to/proteome.faa" "650m" "mean"```

**Notes:**
- In this example, the model "650m" is used, which consumes less GPU memory.
- If GPU memory is not a constraint, you can use the "3b" model instead.
- The output will be saved as a Zarr file at: ```/path/to/outputfolder/{experiment_name}_{model}.zarr```

### Step 4: Generate a Hierarchical Tree
To create an approximate hierarchical tree for the new proteomes, run the following command:
```python PlaceNewProteome.py "/path/to/HieVi_INPHARED_database_650m.zarr" "/path/to/outputfolder/{experiment_name}_{model}.zarr"```

**Notes:**
- Update the database path (HieVi_INPHARED_database_650m.zarr) as needed.
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

