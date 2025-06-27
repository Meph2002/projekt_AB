import numpy as np 
import random
DEFAULT_AA = 'L'
def encode(sequence: str, max_length=400):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_index = {aa: i for i, aa in enumerate(amino_acids)}

    one_hot = np.zeros((max_length, len(amino_acids)))

    for i in range(max_length):
        if i < len(sequence):
            aa = sequence[i]
            index = aa_to_index.get(aa, aa_to_index[DEFAULT_AA])
        else:
            index = aa_to_index[DEFAULT_AA]

        one_hot[i, index] = 1

    return one_hot


def split_data(proteins, split_frac=0.8):
    sequences = np.array([obj.sequence for obj in proteins])
    # print(sequences)
    np.random.shuffle(sequences)

    split_idx = int(split_frac * len(sequences))

    train_data = sequences[:split_idx]
    test_data = sequences[split_idx:]

    return train_data, test_data

def motiv(sequence: str):
    for i in range(len(sequence) - 4):
        triplet = sequence[i:i+5]
        print(triplet)

