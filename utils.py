import numpy as np 


def motiv(sequence: str):
    for i in range(len(sequence) - 4):
        triplet = sequence[i:i+5]
        print(triplet)


def one_hot_encode(seq, max_len):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_index = {aa: i for i, aa in enumerate(amino_acids)}
    encoding = np.zeros((max_len, len(amino_acids)), dtype=np.float32)
    for i, aa in enumerate(seq[:max_len]):
        if aa in aa_to_index:
            encoding[i, aa_to_index[aa]] = 1.0
    return encoding

