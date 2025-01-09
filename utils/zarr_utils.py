import zarr
import numpy as np

class AccessionZarrWriter:
    def __init__(self, output_path):
        """
        Initialize a Zarr store for saving embeddings.
        
        Args:
            output_path (str): Path to the Zarr store.
        """
        self.zarr_store = zarr.open(output_path, mode='w')
        print(f"Initialized Zarr store at {output_path}")

    def write_accession(self, accession, sequence_count, layer_embeddings):
        """
        Write embeddings for an accession to the Zarr store.

        Args:
            accession (str): Accession identifier.
            sequence_count (int): Number of sequences for the accession.
            layer_embeddings (dict): Embeddings for each layer.
        """
        group = self.zarr_store.require_group(accession)
        group.attrs["sequence_count"] = sequence_count

        for layer, embedding in layer_embeddings.items():
            # Save each layer's embedding
            if embedding is not None:
                group.array(layer, embedding, chunks=True, overwrite=True)
        #print(f"Saved embeddings for accession: {accession}")

class AccessionZarrReader:
    def __init__(self, input_path):
        """
        Initialize a Zarr reader for accessing embeddings.

        Args:
            input_path (str): Path to the Zarr store.
        """
        self.zarr_store = zarr.open(input_path, mode='r')
        print(f"Initialized Zarr reader at {input_path}")

    def list_accessions(self):
        """
        List all accession identifiers in the Zarr store.

        Returns:
            list: List of accession identifiers.
        """
        return list(self.zarr_store.group_keys())

    def read_accession(self, accession):
        """
        Read embeddings for a specific accession.

        Args:
            accession (str): Accession identifier.

        Returns:
            dict: Dictionary containing sequence count and layer embeddings.
                  Example:
                  {
                      "sequence_count": int,
                      "layer_embeddings": {
                          "layer0": numpy.ndarray,
                          "layer1": numpy.ndarray,
                          ...
                      }
                  }
        """
        if accession not in self.zarr_store:
            raise KeyError(f"Accession {accession} not found in the Zarr store.")

        group = self.zarr_store[accession]
        sequence_count = group.attrs["sequence_count"]
        layer_embeddings = {layer: group[layer][:] for layer in group.array_keys()}
        
        return {
            "sequence_count": sequence_count,
            "layer_embeddings": layer_embeddings
        }
