import hashlib
from pyrodigal import GeneFinder
from multiprocessing import Pool, cpu_count
from functools import partial
from Bio import SeqIO

def call_genes(sequence, seq_name, is_circular=True, min_gene_len=60):
    seq_bytes = sequence.encode()
    seq_hash = hashlib.md5(seq_bytes).hexdigest()
    use_meta = len(sequence) < 100000
    custom_overlap = min(60, min_gene_len - 1)

    gf = GeneFinder(
        meta=use_meta,
        closed=is_circular,
        min_gene=min_gene_len,
        max_overlap=custom_overlap
    )
    if not use_meta:
        gf.train(seq_bytes)
    genes = gf.find_genes(seq_bytes)

    return [{
        "gene_id": f"{seq_hash}_{idx}",
        "readable_gene_id": f"{seq_name}_{idx}",
        "phage_name": seq_name,
        "order": idx,
        "begin": pred.begin,
        "end": pred.end,
        "strand": "+" if pred.strand == 1 else "-",
        "score": pred.score,
        "confidence": pred.confidence(),
        "sequence": pred.translate(),
    } for idx, pred in enumerate(genes, start=1)]

# Parallel wrapper
def process_sequences(sequences, seq_names, **kwargs):
    with Pool(processes=cpu_count()) as pool:
        return pool.starmap(
            partial(call_genes, **kwargs),
            zip(sequences, seq_names)
        )

def call_genes_from_fasta(dna_fasta_path, protein_fasta_path, is_circular=True, min_gene_len=60):
    """
    Gene-call every sequence in a DNA multi-FASTA file with pyrodigal and
    write the predicted proteins to `protein_fasta_path` (accession_idx headers,
    matching the convention expected by FastaReader downstream).
    """
    seq_names = []
    sequences = []
    with open(dna_fasta_path, 'r') as fasta_file:
        for record in SeqIO.parse(fasta_file, 'fasta'):
            seq_names.append(record.id)
            sequences.append(str(record.seq))

    gene_results = process_sequences(
        sequences, seq_names, is_circular=is_circular, min_gene_len=min_gene_len
    )

    with open(protein_fasta_path, 'w') as out_fasta:
        for genes in gene_results:
            for gene in genes:
                out_fasta.write(f">{gene['readable_gene_id']}\n{gene['sequence']}\n")

    return protein_fasta_path

# # Example usage
# sequences = ["ATGC...", "CGTA..."]  # Your sequences
# seq_names = ["seq1", "seq2"]         # Your sequence names
# gene_results = process_sequences(sequences, seq_names)