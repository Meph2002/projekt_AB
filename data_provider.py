#!/usr/bin/env python3
import sys, errno, re, json, ssl
from urllib import request
from urllib.error import HTTPError
from time import sleep
from Bio import SeqIO
from protein import Protein 
import numpy as np 
import os
import requests
import re
import requests
from requests.adapters import HTTPAdapter, Retry


# rember to delete file from path when URL CHANGES!
POSITIVE_URL = "https://www.ebi.ac.uk:443/interpro/api/protein/reviewed/entry/InterPro/IPR014720/?page_size=200&extra_fields=sequence"
NEGATIVE_URL = "https://rest.uniprot.org/uniprotkb/search?format=fasta&query=%28IPR014720%29+AND+%28length%3A%5B1+TO+200%5D%29&size=500"

MARKED_PATH = 'data/marked.fasta'
NEGATIVE_PATH = "data/UniProtKB_seq_1_200.fasta"


def prepare_data_folder():
    folder_name = "data" 
    if not os.path.exists(folder_name):
        print(f"brak folderu z danymi ./data")
        sys.exit(0)

def get_positive():
    marked_proteins = []

    with open(MARKED_PATH) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            new_protein = Protein.from_fasta_record(record=record, positive=True)
            sequence = new_protein.sequence
            if len(sequence)< 400: 
                marked_proteins.append(new_protein)  
    return np.array(marked_proteins, dtype=object)

def get_negative():
    negative_proteins = []
    with open(NEGATIVE_PATH) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            new_protein = Protein.from_fasta_record(record=record, positive=False)
            sequence = new_protein.sequence
            if len(sequence)< 400:  
                negative_proteins.append(new_protein)
    return np.array(negative_proteins, dtype=object)


# def download_negative():
#     re_next_link = re.compile(r'<(.+)>; rel="next"')
#     retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
#     session = requests.Session()
#     session.mount("https://", HTTPAdapter(max_retries=retries))

#     def get_next_link(headers):
#         if "Link" in headers:
#             match = re_next_link.match(headers["Link"])
#             if match:
#                 return match.group(1)

#     def get_batch(batch_url):
#         while batch_url:
#             response = session.get(batch_url)
#             response.raise_for_status()
#             total = response.headers["x-total-results"]
#             yield response, total
#             batch_url = get_next_link(response.headers)
#     url = NEGATIVE_URL
#     progress = 0
#     with open(NEGATIVE_PATH, 'w') as f:
#         for batch, total in get_batch(url):
#             lines = batch.text.splitlines()
#             if not progress:
#                 print(lines[0], file=f)
#             for line in lines[1:]:
#                 print(line, file=f)
#             progress += len(lines[1:])
#             print(f'Downloaded negative sequences: {progress} / {total}')


# HEADER_SEPARATOR = "|"
# LINE_LENGTH = 80
# def download_positive():
#     # Wyłącz weryfikację SSL, aby uniknąć problemów z konfiguracją
#     context = ssl._create_unverified_context()

#     next = POSITIVE_URL
#     last_page = False
#     attempts = 0
    
#     # Pobierz całkowitą liczbę wyników dla paska postępu
#     try:
#         req = request.Request(POSITIVE_URL, headers={"Accept": "application/json"})
#         res = request.urlopen(req, context=context)
#         payload = json.loads(res.read().decode())
#         total_results = payload["count"]
#     except Exception as e:
#         total_results = "unknown"
#         print(f"Could not get total count: {e}")

#     progress = 0
    
#     with open(MARKED_PATH, 'w') as fasta_file:
#         while next:
#             try:
#                 req = request.Request(next, headers={"Accept": "application/json"})
#                 res = request.urlopen(req, context=context)
                
#                 if res.status == 408:
#                     sleep(61)
#                     continue
#                 elif res.status == 204:
#                     break
                
#                 payload = json.loads(res.read().decode())
#                 next = payload["next"]
#                 attempts = 0
                
#                 if not next:
#                     last_page = True
                    
#             except HTTPError as e:
#                 if e.code == 408:
#                     sleep(61)
#                     continue
#                 else:
#                     if attempts < 3:
#                         attempts += 1
#                         sleep(61)
#                         continue
#                     else:
#                         sys.stderr.write("LAST URL: " + next)
#                         raise e

#             current_batch_size = len(payload["results"])
#             progress += current_batch_size
#             print(f'Downloaded positive sequences: {progress} / {total_results}')

#             for item in payload["results"]:
#                 entries = None
#                 if "entries" in item:
#                     entries = item["entries"]
                
#                 if entries is not None:
#                     entries_header = "-".join(
#                         [entry["accession"] + "(" + ";".join(
#                             [
#                                 ",".join(
#                                     [str(fragment["start"]) + "..." + str(fragment["end"]) 
#                                      for fragment in locations["fragments"]]
#                                 ) for locations in entry["entry_protein_locations"]
#                             ]
#                         ) + ")" for entry in entries]
#                     )
#                     fasta_file.write(">" + item["metadata"]["accession"] + HEADER_SEPARATOR
#                                     + entries_header + HEADER_SEPARATOR
#                                     + item["metadata"]["name"] + "\n")
#                 else:
#                     fasta_file.write(">" + item["metadata"]["accession"] + HEADER_SEPARATOR 
#                                     + item["metadata"]["name"] + "\n")

#                 seq = item["extra_fields"]["sequence"]
#                 fastaSeqFragments = [seq[0+i:LINE_LENGTH+i] for i in range(0, len(seq), LINE_LENGTH)]
#                 for fastaSeqFragment in fastaSeqFragments:
#                     fasta_file.write(fastaSeqFragment + "\n")
            
#             if next:
#                 sleep(1)


# if __name__ == "__main__":
#     prepare_data_folder()
#     download_positive()
#     download_negative()
#     print("Dane zostały zapisane do plików marked.fasta i UniProtKB_seq_1_200.fasta")