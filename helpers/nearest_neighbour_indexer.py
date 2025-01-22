import faiss
import zarr
import argparse
import os
def get_argument_parser():
    """
    Create and return the argument parser.
    """
    parser = argparse.ArgumentParser(description="Process and cluster proteome representations.")        
    parser.add_argument("--zarr_path", type=str, required=True, help="Path to query Zarr store.")    
    return parser



def main(args):
    zarr_store = zarr.open(args.zarr_path)
    vectors = zarr_store['vectors_mean'][:]
    phage_accesssions = zarr_store['accessions'][:]

    index = faiss.IndexFlatL2(vectors.shape[1])  # Use L2 (Euclidean) distance
    index.add(vectors)
    print(f"Number of vectors in the index: {index.ntotal}")

    # Step 3: Save the FAISS index to a file
    output_name = args.zarr_path + 'faiss_index.bin'
    faiss.write_index(index, output_name)
    print(f"FAISS index saved to {output_name}.")


if __name__ == "__main__":
    parser = get_argument_parser()
    args = parser.parse_args()
    main(args)
