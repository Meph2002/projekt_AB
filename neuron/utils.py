import numpy as np 
import random
def encode(sequence: str, max_length=100):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_index = {aa: i for i, aa in enumerate(amino_acids)}
    
    one_hot = np.zeros((max_length, len(amino_acids)))
    
    for i, aa in enumerate(sequence[:max_length]):  # Truncate if too long
        if aa in aa_to_index:
            one_hot[i, aa_to_index[aa]] = 1
            
    return one_hot


def split_data(proteins, split_frac=0.8):
    sequences = np.array([obj.sequence for obj in proteins])
    # print(sequences)
    np.random.shuffle(sequences)

    split_idx = int(split_frac * len(sequences))

    train_data = sequences[:split_idx]
    test_data = sequences[split_idx:]

    return train_data, test_data