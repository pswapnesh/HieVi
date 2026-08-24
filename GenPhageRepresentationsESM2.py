from utils.proteome_process import *
from utils.fasta_utils import *
from utils.esm_utils import *
from utils.prefetcher import *
from utils.gene_caller import call_genes_from_fasta
import os
import argparse

def main(expt_name, output_folder, fasta_path, model_name, mode, aa_fasta=False,
         is_circular=True, min_gene_len=60, chunk_size=16):
    """
    Main function to process sequences, compute embeddings, and save results.

    Args:
        expt_name (str): Experiment name.
        output_folder (str): Output folder for results.
        fasta_path (str): Path to the input FASTA file (DNA by default, amino acid if aa_fasta=True).
        model_name (str): Name of the ESM model to use.
        mode (str): Mode for generating embeddings ('mean' or 'cls').
        aa_fasta (bool): If True, fasta_path is already an amino-acid multi-FASTA and gene
            calling is skipped. If False (default), fasta_path is a DNA multi-FASTA and
            genes are called internally with pyrodigal before embedding.
        is_circular (bool): Whether input DNA sequences are circular (only used when aa_fasta=False).
        min_gene_len (int): Minimum gene length for gene calling (only used when aa_fasta=False).
        chunk_size (int): Number of accessions to accumulate before writing to disk.
    """

    # Create output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    if not aa_fasta:
        print("Input treated as a DNA multi-FASTA; calling genes with pyrodigal...")
        protein_fasta_path = os.path.join(output_folder, f"{expt_name}_genes.faa")
        fasta_path = call_genes_from_fasta(
            fasta_path, protein_fasta_path, is_circular=is_circular, min_gene_len=min_gene_len
        )
        print(f"Gene calling complete. Predicted proteins written to {fasta_path}")

    # Set file paths based on input folder and experiment name
    zarr_store_path = os.path.join(output_folder, f"{expt_name}_{model_name}.zarr")
    log_file = os.path.join(output_folder, f"{expt_name}_{model_name}_error_log.txt")

    esm_model = EsmEmbedding(model_name)
    fasta_reader = FastaReader(fasta_path)
    accession_generator = fasta_reader.unique_accession_generator()
    # Wrap the accession generator with PrefetchCache to enable prefetching
    prefetcher = PrefetchCache(generator=accession_generator, prefetch_size=32)

    v,_ = esm_model.predict([('name','M')])
    ndim = v.to(device="cpu").numpy().shape[0]

    #processor = VectorProcessor(predict=esm_model.predict, ndim=ndim, zarr_path=zarr_store_path)
    processor = VectorProcessor(predict=esm_model.predict, ndim=ndim,mode= mode, zarr_path=zarr_store_path,log_path = log_file)

    # Process data and store in Zarr
    zarr_path = processor.process_and_store(prefetcher,fasta_reader.unique_accessions)



if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Process FASTA sequences and get embeddings.")
    
    # Define expected command line arguments
    parser.add_argument('expt_name', type=str, help="Experiment name")
    parser.add_argument('output_folder', type=str, help="Output folder path")
    parser.add_argument('fasta_path', type=str, help="Path to the FASTA file (DNA multi-FASTA by default, amino acid multi-FASTA if --aa_fasta is set)")
    parser.add_argument('model_name', type=str, choices=["650m", "3b", "15b"], help="Model name")
    parser.add_argument('mode', type=str, choices=["mean", "cls", "mean+cls"], help="Mode for processing embeddings (mean/cls)")
    parser.add_argument('--aa_fasta', action='store_true', help="Treat fasta_path as an amino-acid multi-FASTA and skip internal gene calling")
    parser.add_argument('--linear', action='store_true', help="Treat input DNA sequences as linear instead of circular (only used without --aa_fasta)")
    parser.add_argument('--min_gene_len', type=int, default=60, help="Minimum gene length for gene calling (only used without --aa_fasta)")

    # Parse command line arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.expt_name, args.output_folder, args.fasta_path, args.model_name, args.mode,
         aa_fasta=args.aa_fasta, is_circular=not args.linear, min_gene_len=args.min_gene_len)


#
#python GenPhageRepresentationsESM2.py "Expt_name" "/path/to/outputfolder/" "path/to/genomes.fasta" "650m" "mean"
#python GenPhageRepresentationsESM2.py "Expt_name" "/path/to/outputfolder/" "path/to/proteomemultifasta.faa" "650m" "mean" --aa_fasta