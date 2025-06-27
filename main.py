from data_provider import get_positive, get_negative,prepare_data_folder
from cnn_model import ProteinDataset, train_model, plot_training
from protein_analysis import analyze
import numpy as np 
from utils import split_data
from protein import Protein

from utils import motiv


def init():
    prepare_data_folder()

positives = get_positive()
negatives = get_negative()

all_proteins = np.concatenate([positives, negatives])
np.random.shuffle(all_proteins)

split_idx = int(0.8 * len(all_proteins))
train_proteins = all_proteins[:split_idx]
test_proteins = all_proteins[split_idx:]

# print(f"positive protein sequence max len: {Protein.positive_max_len}")
# print(f"negative protein sequence max len: {Protein.negative_max_len}")


analyze(marked_proteins=train_proteins, all_proteins=all_proteins)

train_set = ProteinDataset(train_proteins)
test_set = ProteinDataset(test_proteins)

model, losses, accs = train_model(train_set, test_set, epochs=10)

plot_training(losses, accs)

