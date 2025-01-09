from Bio import SeqIO

class FastaReader:
    def __init__(self, fasta_path, unknown_token='<unk>'):
        """
        Initialize the FastaReader with the given parameters.
        
        Args:
            fasta_path (str): Path to the multi-FASTA file.
            unknown_token (str): String to replace `*` in sequences (default: '<unk>').
        """
        self.fasta_path = fasta_path
        self.unknown_token = unknown_token
        self.unique_accessions = self._count_unique_accessions()
        print(f"FastaReader initialized. Number of unique accessions: {self.unique_accessions}")

    def _count_unique_accessions(self):
        """
        Count the number of unique accessions in the FASTA file.

        Returns:
            int: Count of unique accessions.
        """
        accessions = set()
        with open(self.fasta_path, 'r') as fasta_file:
            for record in SeqIO.parse(fasta_file, 'fasta'):
                accession = record.id.split('_')[0]  # Extract the accession from the ID
                accessions.add(accession)
        return len(accessions)

    def generator(self):
         with open(self.fasta_path, 'r') as fasta_file:
            for record in SeqIO.parse(fasta_file, 'fasta'):
                accession = record.id.split('_')[0]  # Extract accession
                yield accession,record.id,str(record.seq).replace('*', self.unknown_token)

    def unique_accession_generator(self):
        """
        Generate sequences grouped by unique accession one at a time,
        without loading all data into memory.

        Yields:
            Tuple (accession, list of sequences).
        """
        current_accession = None
        current_sequences = []

        with open(self.fasta_path, 'r') as fasta_file:
            for record in SeqIO.parse(fasta_file, 'fasta'):
                accession = record.id.split('_')[0]  # Extract accession
                
                # If a new accession is encountered, yield the previous one
                if current_accession and accession != current_accession:
                    yield current_accession, current_sequences
                    current_sequences = []  # Reset for the next accession
                
                current_accession = accession
                # Replace `*` with the unknown token
                current_sequences.append(str(record.seq).replace('*', self.unknown_token))

            # Yield the last accession group
            if current_accession:
                yield current_accession, current_sequences