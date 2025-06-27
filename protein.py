import numpy as np 

class Protein:
    max_sequence_len = 0
    negative_max_len = 0 
    positive_max_len = 0
    def __init__(self, name: str, id: str, sequence: str,label, domains:list = None):
        self.name = name
        self.id = id
        self.sequence =sequence
        self.domains = np.array([]) if domains is None else np.array(domains)
        self.label = label
    
    def from_fasta_record(record, positive):
        parts = record.description.split("|")
        id = parts[0]
        name = parts[-1]
        domains =[]
        sequence = str(record.seq)

        sequence_len = len(record.seq)         
        # if (sequence_len > Protein.max_sequence_len):
        #     Protein.max_sequence_len = sequence_len
        # if positive and sequence_len > Protein.positive_max_len: 
        #     Protein.positive_max_len = sequence_len
        # if not positive and sequence_len > Protein.negative_max_len: 
        #     Protein.negative_max_len = sequence_len 

        if len(parts) > 1 and "(" in parts[1]:
                domain_info = parts[1]
                domain_acc = domain_info.split("(")[0]
                
                for loc in domain_info.split("(")[1].replace(")", "").split(";"):
                    start, end = map(int, loc.split("..."))
                    domains.append(Domain(start=start, end=end))

        return Protein(
                id=id,
                name=name,
                sequence=sequence,
                domains=domains,
                label=1 if positive else 0

            )

    def __str__(self):
        domain_info = "".join(
            f"  {domain}"
            for domain in self.domains
        ) if self.domains.size > 0 else "  Brak domen\n"
        return (
              f"Protein: {self.id} - {self.name}) \n"
              f"sekwencja: {self.sequence} \n"
              f"domeny: \n {domain_info}"
         )


class Domain: 
        def __init__(self, start = 0, end = 0):
            self.start = start
            self.end = end

        def __str__(self):
                return (f"Domain: {self.start} - {self.end}) \n")
        def __repr__(self):
            return f"Domain(start={self.start}, end={self.end})"
        

        