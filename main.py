from data_provider import get_positive, get_negative,prepare_data_folder
from cnn_model import ProteinDataset, train_model, plot_training
from protein_analysis import analyze
import numpy as np 


def init():
    prepare_data_folder()


def main():
    """ These parameters can be passed through optparse module """
    max_len = 500
    positive_dataset_path = 'data/marked.fasta'
    negative_dataset_path = "data/UniProtKB_seq_1_200.fasta"

    positives = get_positive(positive_dataset_path)
    negatives = get_negative(negative_dataset_path)

    all_proteins = np.concatenate([positives, negatives])
    np.random.shuffle(all_proteins)

    split_idx = int(0.8 * len(all_proteins))
    train_proteins = all_proteins[:split_idx]
    test_proteins = all_proteins[split_idx:]

    analyze(marked_proteins=train_proteins, all_proteins=all_proteins)

    train_set = ProteinDataset(train_proteins, max_len)
    test_set = ProteinDataset(test_proteins, max_len)

    model, losses, accs = train_model(train_set, test_set, epochs=10, max_len=max_len)

    plot_training(losses, accs)


if __name__ == '__main__':
    main()
