

def count_domains(proteins_with_domains):
    number_of_domains = 0
    for protein in proteins_with_domains:
        length = len(protein.domains)
        number_of_domains += length
    
    return number_of_domains


def mean_number_of_domains(proteins_with_domains):
    number_of_domains = count_domains(proteins_with_domains=proteins_with_domains)
    lenght = len(proteins_with_domains)
    return round(number_of_domains/lenght, 2)


def find_max_number_of_domains(proteins):
    number_of_domains = 0
    for protein in proteins:
        lenght = len(protein.domains)
        if  lenght > number_of_domains: 
            number_of_domains = lenght 
    return number_of_domains


def count_proteins_with_domains(proteins):
    with_domains = 0
    for protein in proteins:
        lenght = len(protein.domains)
        if lenght > 0: 
            with_domains += 1

    all_proteins_len = len(proteins)
    return with_domains, all_proteins_len-with_domains


def sequece_len(proteins): 
     for protein in proteins:
        lenght = len(protein.sequence)
        # print(lenght)

        
def analyze(marked_proteins,all_proteins):
    sequece_len(all_proteins)
    number_of_domains = count_domains(marked_proteins)
    mean_domains = mean_number_of_domains(marked_proteins)
    max_domains = find_max_number_of_domains(all_proteins)
    protains_counter =  count_proteins_with_domains(all_proteins)

    print(f"nmumber of domains: {number_of_domains}")
    print(f"mean number of domains: {mean_domains}")
    print(f"max number of domains {max_domains}")
    print(f"proteins with domains: {protains_counter[0]}")
    print(f"proteins without domains: {protains_counter[1]}")


from collections import Counter
def count_domain_sequence_frequencies(protein_list):
    domain_seqs = []
    for protein in protein_list:
        for domain in protein.domains:
            # Zabezpieczenie: sprawdzamy poprawność indeksów
            if 0 <= domain.start < domain.end <= len(protein.sequence):
                domain_seq = protein.sequence[domain.start:domain.end]
                domain_seqs.append(domain_seq)
    return Counter(domain_seqs)

import matplotlib.pyplot as plt

def plot_domain_pie_chart(domain_counter):
    # Odfiltrowanie zera i posortowanie po liczności
    labels = list(domain_counter.keys())
    sizes = list(domain_counter.values())

    # Możesz ograniczyć do np. top 10 domen, a resztę połączyć w "inne"
    max_labels = 10
    if len(labels) > max_labels:
        labels_sorted = sorted(domain_counter.items(), key=lambda x: x[1], reverse=True)
        top = labels_sorted[:max_labels]
        other = labels_sorted[max_labels:]
        
        labels = [label for label, _ in top] + ['inne']
        sizes = [count for _, count in top] + [sum([count for _, count in other])]

    # Wykres kołowy
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title("Częstotliwość unikalnych sekwencji domen")
    plt.axis('equal')  # Równe osie dla pełnego koła
    plt.show()

    