import logging
import numpy as np

def normalize(matrix):
    row_norms = np.linalg.norm(matrix, axis=1)
    return matrix / row_norms[:, np.newaxis]

class AccessionProcessorWithLogging:
    def __init__(self, generator, predict_function, log_file="error_log.txt"):
        """
        Initialize the processor with the given parameters.
        
        Args:
            generator (generator): The generator that yields accessions and their sequences.
            predict_function (function): The function used to predict embeddings for sequences.
            log_file (str): Path to the log file to store errors (default: 'error_log.txt').
        """
        self.generator = generator
        self.predict_function = predict_function
        self.log_file = log_file
        self._setup_logging()

    def _setup_logging(self):
        """Set up the logging configuration."""
        logging.basicConfig(
            filename=self.log_file, 
            level=logging.ERROR, 
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def process_accessions(self, mode='mean'):
        """
        Process accessions and generate embeddings for each sequence in the accession.

        Yields:
            dict: Dictionary containing accession, sequence count, and embeddings.
        """
        for accession, sequences in self.generator:
            try:
                embeddings = self.process_sequences(accession,sequences, mode)
                yield {
                    "Accession": accession,
                    "SequenceCount": len(sequences),
                    "LayerEmbeddings": embeddings
                }
            except Exception as e:
                # Log error if something goes wrong
                logging.error(f"Error processing accession {accession}: {e}")
                continue

    def process_sequences(self,accession, sequences, mode):
        """
        Process the sequences of a given accession to obtain embeddings.

        Args:
            sequences (list): List of sequences for the current accession.
            mode (str): Mode for generating embeddings ('mean' or 'cls').

        Returns:
            dict: Embeddings from each layer.
        """
        embeddings = {}
        for i,sequence in enumerate(sequences):
            try:
                embedding = self.predict_function([(accession +'_' + str(i),sequence)], mode)
                for layer, layer_embedding in embedding.items():
                    if layer not in embeddings:
                        embeddings[layer] = []
                    embeddings[layer].append(layer_embedding)
            except Exception as e:
                # Log individual sequence errors and continue processing
                logging.error(f"Error processing sequence {sequence}: {e}")
                continue

        # Average embeddings across sequences for each layer
        for layer in embeddings:
            embeddings[layer] = np.mean(normalize(embeddings[layer]), axis=0)

        return embeddings
