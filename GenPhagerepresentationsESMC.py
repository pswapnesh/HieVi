import os
import argparse
from Bio import SeqIO
from utils.esm_utils import *
from utils.fasta_utils import *
from utils.process_proteome import *
from utils.zarr_utils import *
from utils.prefetcher import *
from tqdm import tqdm 

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
class EsmCambrian:
    def __init__(self,model_name = "esmc_600m"):        
        # model_name esmc_300m,esmc_600m , esmc_6b
        self.client = ESMC.from_pretrained(model_name).to("cuda") # or "cpu"

    def predict(self,data):
        name,seq = data[0]
        protein = ESMProtein(sequence=seq)
        
        protein_tensor = self.client.encode(protein)
        logits_output = self.client.logits(
        protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
        )
        token_reps = {}
        token_reps['-1'] = logits_output.embeddings[:,1:-1,:].mean(axis = 1).to('cpu')
        return token_reps

def main(expt_name, output_folder, fasta_path, model_name, mode, chunk_size=128):
    """
    Main function to process sequences, compute embeddings, and save results.

    Args:
        expt_name (str): Experiment name.
        output_folder (str): Output folder for results.
        fasta_path (str): Path to the input FASTA file.
        model_name (str): Name of the ESM model to use.
        mode (str): Mode for generating embeddings ('mean' or 'cls').
        chunk_size (int): Number of accessions to accumulate before writing to disk.
    """


    # Create output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Set file paths based on input folder and experiment name
    zarr_store_path = os.path.join(output_folder, f"{expt_name}_{model_name}.zarr")
    log_file = os.path.join(output_folder, f"{expt_name}_{model_name}_error_log.txt")

    # Initialize the Zarr writer for storing the results
    zarr_writer = AccessionZarrWriter(zarr_store_path)

    # Initialize the FastaReader for reading the FASTA file
    fasta_reader = FastaReader(fasta_path)

    # Get the unique accession generator from FastaReader
    accession_generator = fasta_reader.unique_accession_generator()

    # Wrap the accession generator with PrefetchCache to enable prefetching
    prefetcher = PrefetchCache(generator=accession_generator, prefetch_size=8)

    # Initialize the EsmEmbedding model
    esm_embedding_model = EsmCambrian(model_name)
    #esm_embedding_model = EsmEmbedding(model=model_name)

    # Initialize the AccessionProcessorWithLogging, passing the prefetcher generator and the predict function
    processor = AccessionProcessorWithLogging(generator=prefetcher, 
                                              predict_function=esm_embedding_model.predict, 
                                              log_file=log_file)

    # Initialize a buffer to hold chunks of results
    chunk_buffer = []

    # Process the accessions and save results in chunks
    for result in tqdm(processor.process_accessions(mode=mode)):
        chunk_buffer.append(result)  # Accumulate results in the buffer

        # If buffer reaches the chunk size, write results to disk
        if len(chunk_buffer) >= chunk_size:
            for chunk in chunk_buffer:
                accession = chunk["Accession"]
                seq_count = chunk["SequenceCount"]
                embeddings = chunk["LayerEmbeddings"]
                zarr_writer.write_accession(accession, seq_count, embeddings)
            chunk_buffer = []  # Clear the buffer

    # Write any remaining results in the buffer
    if chunk_buffer:
        for chunk in chunk_buffer:
            accession = chunk["Accession"]
            seq_count = chunk["SequenceCount"]
            embeddings = chunk["LayerEmbeddings"]
            zarr_writer.write_accession(accession, seq_count, embeddings)


if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Process FASTA sequences and get embeddings.")
    
    # Define expected command line arguments
    parser.add_argument('expt_name', type=str, help="Experiment name")
    parser.add_argument('output_folder', type=str, help="Output folder path")
    parser.add_argument('fasta_path', type=str, help="Path to the FASTA file")
    parser.add_argument('model_name', type=str, choices=["esmc_600m", "esmc_300m"], help="Model name")
    parser.add_argument('mode', type=str, choices=["mean", "cls"], help="Mode for processing embeddings (mean/cls)")

    # Parse command line arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.expt_name, args.output_folder, args.fasta_path, args.model_name, args.mode)


#
#python GenPhageRepresentations.py "Sept1_2024" "/media/microscopie-lcb/swapnesh/protein/embeddings/phages/Sept1_2024/" "/media/microscopie-lcb/swapnesh/protein/embeddings/phages/Sept1_2024/" "/media/microscopie-lcb/swapnesh/protein/embeddings/phages/Sept1_2024/1Sep2024_vConTACT2_proteins.faa" "650m" "mean"
#python GenPhageRepresentations.py "MV_INRAE" "/media/microscopie-lcb/swapnesh/protein/embeddings/phages/Metavirome_INRAE_HieVi/" "/media/microscopie-lcb/swapnesh/protein/embeddings/phages/Metavirome_INRAE_HieVi/Metavirome_INRAE_HieVi.faa" "3b" "mean"