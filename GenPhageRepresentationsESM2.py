from utils.proteome_process import *
from utils.fasta_utils import *
from utils.esm_utils import *
from utils.prefetcher import *
import os
import argparse

def main(expt_name, output_folder, fasta_path, model_name, mode, chunk_size=16):
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

    esm_model = EsmEmbedding(model_name)
    fasta_reader = FastaReader(fasta_path)
    accession_generator = fasta_reader.generator()
    # Wrap the accession generator with PrefetchCache to enable prefetching
    #prefetcher = PrefetchCache(generator=accession_generator, prefetch_size=32)

    v,_ = esm_model.predict([('name','M')])
    ndim = v.to(device="cpu").numpy().shape[0]

    #processor = VectorProcessor(predict=esm_model.predict, ndim=ndim, zarr_path=zarr_store_path)
    processor = VectorProcessor(predict=esm_model.predict, ndim=ndim,mode= mode, zarr_path=zarr_store_path,log_path = log_file)

    # Process data and store in Zarr
    zarr_path = processor.process_and_store(accession_generator)



if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Process FASTA sequences and get embeddings.")
    
    # Define expected command line arguments
    parser.add_argument('expt_name', type=str, help="Experiment name")
    parser.add_argument('output_folder', type=str, help="Output folder path")
    parser.add_argument('fasta_path', type=str, help="Path to the FASTA file")
    parser.add_argument('model_name', type=str, choices=["650m", "3b", "15b"], help="Model name")
    parser.add_argument('mode', type=str, choices=["mean", "cls"], help="Mode for processing embeddings (mean/cls)")

    # Parse command line arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.expt_name, args.output_folder, args.fasta_path, args.model_name, args.mode)


#
#python GenPhageRepresentations.py "Expt_name" "/path/to/outputfolder/" "path/to/proteomemultifasta.faa" "650m" "mean"