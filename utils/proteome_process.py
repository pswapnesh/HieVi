import zarr
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import torch
import logging

class VectorProcessor:
    def __init__(self, predict, ndim, zarr_path, chunk_size=16, log_path="failed_pnames.log"):
        """
        Initializes the VectorProcessor.
        
        Parameters:
            predict (function): Function to generate vectors from (pname, seq) pairs.
            ndim (int): Dimensionality of the vectors produced by predict.
            zarr_path (str): Path to store the Zarr file.
            chunk_size (int): Size of the chunks for Zarr datasets.
            log_path (str): Path to the log file to store failed pnames.
        """
        self.predict = predict
        self.ndim = ndim
        self.zarr_path = zarr_path
        self.chunk_size = chunk_size
        self.failed_pnames = defaultdict(list)
        self.log_path = log_path

        # Setup the logging configuration
        logging.basicConfig(filename=self.log_path, 
                            level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')

    def _process_vector(self, pname, seq, accession):
        """
        Helper method to process a vector and handle errors.
        If a vector cannot be processed, it is logged and the method returns None.
        
        Parameters:
            pname (str): The protein name.
            seq (str): The sequence.
            accession (str): The accession ID.

        Returns:
            torch.Tensor: The processed vector or None if failed.
        """
        try:
            # Predict the vector for the current (pname, seq)
            vector = self.predict([(pname, seq)])  # A list of one pair
            if vector.ndim == 1:  # If predict returns a 1D array, reshape to (1, ndim)
                vector = vector[None, :]
            return vector

        except Exception as e:
            # Log failed (pname, seq) pairs for the given accession
            logging.error(f"Failed to process {accession} - {pname}: {e}")
            return None  # Return None if processing failed
        
    def compute_mean_vector(self, data, accession):
        """
        Compute the mean vector and count for a given accession's data.
        
        Parameters:
            data (list of tuples): List of (pname, seq) pairs for a single accession.
            accession (str): The accession ID to log failures for.

        Returns:
            tuple: (mean_vector, count), where:
                - mean_vector (ndarray): Mean vector of shape (ndim,).
                - count (int): Number of vectors used to compute the mean.
        """
      
        # Use list comprehension to directly generate all vectors, logging failures inside the helper function
        all_vectors = [
            self._process_vector(pname, seq, accession)  # Process vector and log failure if necessary
            for pname, seq in data
        ]        

        all_vectors = [vector for vector in all_vectors if vector is not None]
        
        count = len(all_vectors)
        
        # Combine all vectors into a single array
        all_vectors = torch.cat(all_vectors, dim=0)  # Shape: (total_vectors, ndim)
        
        norms = all_vectors.norm(p=2, dim=1, keepdim=True)  # Compute L2 norm along axis 1 (for each vector)
        normalized_vectors = all_vectors / norms  # Normalize each vector
        
        # Compute mean and count
        mean_vector = normalized_vectors.mean(axis=0)  # Mean along axis 0
        count = all_vectors.shape[0]  # Total number of vectors
        
        return mean_vector.to('cpu').numpy(), count

    def process_and_store(self, generator):
        """
        Processes data from a generator and stores results in a Zarr file.
        
        Parameters:
            generator (generator): Yields (accession, pname, seq) for processing.
        
        Returns:
            str: Path to the Zarr file containing processed data.
        """
        accession_data = defaultdict(list)

        # Group by accession
        for accession, pname, seq in generator:
            accession_data[accession].append((pname, seq))

        accessions = list(accession_data.keys())
        n_accessions = len(accessions)

        # Initialize Zarr store with chunk_size
        store = zarr.open(self.zarr_path, mode='w')
        store.create_dataset('accessions', shape=(n_accessions,), dtype='<U50', chunks=(self.chunk_size,), compressor=None)
        store.create_dataset('counts', shape=(n_accessions,), dtype=np.int32, chunks=(self.chunk_size,), compressor=None)
        store.create_dataset('vectors', shape=(n_accessions, self.ndim), dtype=np.float32, chunks=(self.chunk_size, self.ndim), compressor=None)

        # Process each accession and store results in Zarr
        for i, accession in tqdm(enumerate(accessions)):
            data = accession_data[accession]
            mean_vector, count = self.compute_mean_vector(data, accession)
            store['vectors'][i] = mean_vector  # Append mean_vector to 'vectors' dataset
            store['counts'][i] = count  # Append count to 'counts' dataset
            store['accessions'][i] = accession

        return self.zarr_path

    def get_failed_items(self):
        """
        Retrieve accessions and (pname, seq) pairs that failed processing.
        
        Returns:
            tuple: (failed_accessions, failed_pnames), where:
                - failed_accessions (list): List of accessions that failed completely.
                - failed_pnames (dict): Dictionary of accessions mapping to failed pnames.
        """
        return self.failed_pnames
