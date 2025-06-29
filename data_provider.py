#!/usr/bin/env python3
import sys
from Bio import SeqIO
from protein import Protein 
import numpy as np 
import os


def prepare_data_folder():
    folder_name = "data" 
    if not os.path.exists(folder_name):
        print(f"brak folderu z danymi ./data")
        sys.exit(0)


def get_positive(positives_path):
    marked_proteins = []

    with open(positives_path) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            new_protein = Protein.from_fasta_record(record=record, positive=True)
            sequence = new_protein.sequence
            if len(sequence)< 400: 
                marked_proteins.append(new_protein)  
    return np.array(marked_proteins, dtype=object)


def get_negative(negatives_path):
    negative_proteins = []
    with open(negatives_path) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            new_protein = Protein.from_fasta_record(record=record, positive=False)
            sequence = new_protein.sequence
            if len(sequence)< 400:  
                negative_proteins.append(new_protein)
    return np.array(negative_proteins, dtype=object)