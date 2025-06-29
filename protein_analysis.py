

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
    