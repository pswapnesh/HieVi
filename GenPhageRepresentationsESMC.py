import os
import argparse
from Bio import SeqIO
from utils.proteome_process import *
from utils.fasta_utils import *
from utils.prefetcher import *
import os
import argparse

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

class EsmCambrian:
    def __init__(self,model_name = "esmc_600m",mode = "mean"):        
        # model_name esmc_300m,esmc_600m , esmc_6b
        self.client = ESMC.from_pretrained(model_name).to("cuda") # or "cpu"
        self.mode = mode

    def predict(self,data):
        name,seq = data[0]
        protein = ESMProtein(sequence=seq)
        
        protein_tensor = self.client.encode(protein)
        logits_output = self.client.logits(
        protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
        )
        # token_representations = {}
        
        # if 'mean' in self.mode:
        #     token_representations['mean'] = logits_output.embeddings[0,1:-1,:].mean(axis = 0)
        # if 'cls' in self.mode:
        #     token_representations['cls'] = logits_output.embeddings[0,0,:]
        return logits_output.embeddings[-1,1:-1,:].mean(axis = 0),logits_output.embeddings[-1,0,:]


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

    esm_model = EsmCambrian(model_name,mode)

    fasta_reader = FastaReader(fasta_path)
    
    accession_generator = fasta_reader.unique_accession_generator()
    
    # Wrap the accession generator with PrefetchCache to enable prefetching
    #prefetcher = PrefetchCache(generator=accession_generator, prefetch_size=32)

    v_mean,v_cls = esm_model.predict([('name','M')])    
    ndim = v_cls.to(device="cpu").numpy().shape[0]

    processor = VectorProcessor(predict=esm_model.predict, ndim=ndim,mode= mode, zarr_path=zarr_store_path,log_path = log_file)

    # Process data and store in Zarr
    zarr_path = processor.process_and_store(accession_generator,fasta_reader.unique_accessions)



if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Process FASTA sequences and get embeddings.")
    
    # Define expected command line arguments
    parser.add_argument('expt_name', type=str, help="Experiment name")
    parser.add_argument('output_folder', type=str, help="Output folder path")
    parser.add_argument('fasta_path', type=str, help="Path to the FASTA file")
    parser.add_argument('model_name', type=str, choices=["esmc_600m", "esmc_300m"], help="Model name")
    parser.add_argument('mode', type=str, choices=["mean", "cls", "mean+cls"], help="Mode for processing embeddings (mean/cls)")

    # Parse command line arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args.expt_name, args.output_folder, args.fasta_path, args.model_name, args.mode)


#
#python GenPhageRepresentations.py "Expt_name" "/path/to/outputfolder/" "path/to/proteomemultifasta.faa" "650m" "mean"
#
#python GenPhageRepresentations.py "Sept1_2024" "/media/microscopie-lcb/swapnesh/protein/embeddings/phages/Sept1_2024/" "/media/microscopie-lcb/swapnesh/protein/embeddings/phages/Sept1_2024/" "/media/microscopie-lcb/swapnesh/protein/embeddings/phages/Sept1_2024/1Sep2024_vConTACT2_proteins.faa" "650m" "mean"
#python GenPhageRepresentations.py "MV_INRAE" "/media/microscopie-lcb/swapnesh/protein/embeddings/phages/Metavirome_INRAE_HieVi/" "/media/microscopie-lcb/swapnesh/protein/embeddings/phages/Metavirome_INRAE_HieVi/Metavirome_INRAE_HieVi.faa" "3b" "mean"