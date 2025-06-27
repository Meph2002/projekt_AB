import numpy as np 
import random

def motiv(sequence: str):
    for i in range(len(sequence) - 4):
        triplet = sequence[i:i+5]
        print(triplet)

