from data_provider import get_positive, get_negative,prepare_data_folder

from protein_analysis import analyze
import numpy as np 
from neuron.utils import split_data
from protein import Protein

def init():
    prepare_data_folder()


init()
marked_proteins = get_positive()
negative_proteins = get_negative()
all_proteins = np.array([])
all_proteins = np.append(all_proteins, marked_proteins)
all_proteins = np.append(all_proteins, negative_proteins)


print(f"positive protein sequence max len: {Protein.positive_max_len}")
print(f"negative protein sequence max len: {Protein.negative_max_len}")


analyze(marked_proteins=marked_proteins, all_proteins=all_proteins)
train, test =  split_data(all_proteins)
print(test)
print(train.shape)
print(test.shape)