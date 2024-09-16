import numpy as np
from Bio import PDB
import os
import torch
import pandas as pd

def calculate_distance(atom1, atom2):
  # Oblicza odległość euklidesową między dwoma atomami CA.
    diff_vector = atom1.get_vector() - atom2.get_vector()
    distance = np.linalg.norm(diff_vector)
    return distance

def contact_map(structure, threshold=9.5):
   # Generuje mapę kontaktów dla danej struktury białka.
    ca_atoms = [atom for atom in structure.get_atoms() if atom.get_id() == 'CA']
    num_atoms = len(ca_atoms)
    print(num_atoms)
    contact_matrix = np.zeros((num_atoms, num_atoms), dtype=int)

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            distance = calculate_distance(ca_atoms[i], ca_atoms[j])
            if distance <= threshold:
                contact_matrix[i, j] = 1
                contact_matrix[j, i] = 1

    return contact_matrix

def is_memory_address(s):
    if isinstance(s, str) and s.startswith('0x') and all(c in '0123456789abcdefABCDEF' for c in s[2:]):
        return True
    return False

if __name__ == "__main__":
    filepath_phb = "data/pdb_file/drugbank/"
    df = pd.read_csv("data/drugbank_aft.tsv", sep=' ')

    # 获取数据
    data = df[['protein', 'protein_id', 'seq']]

    id_name = dict()
    id_seq = dict()

    for index, row in data.iterrows():
        name = row['protein']
        id = row['protein_id']
        seq = row['seq']
        if id in id_name:
            continue
        id_name[id] = name
        id_seq [id] = seq
    print(len(id_name))
    print(len(id_seq))

    protein_edge_index = dict()
    error_list = []
    seq_error_list = []
    for id, name in id_name.items():
        pdb_file_name = filepath_phb + name + '.pdb'

        if not os.path.exists(pdb_file_name):
            print(f"File {id} does not exist. Skipping...")
            protein_edge_index[id] = ''
            continue

        pdb_parser = PDB.PDBParser(QUIET=True)
        structure = pdb_parser.get_structure(name, pdb_file_name)

        if not structure:
            error_list.append(id)
            protein_edge_index[id] = ''
            continue

        contact_matrix = contact_map(structure)

        edge_index = []
        num_nodes = contact_matrix.shape[0]
        seq_len = len(id_seq [id])
        
        if num_nodes != seq_len:
            protein_edge_index[id] = ''
            seq_error_list.append(id)
            continue
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if contact_matrix[i][j] == 1:
                    edge_index.append([i, j])
        edge_index = torch.tensor(edge_index).t().contiguous()
        edge_index = edge_index.numpy()
        protein_edge_index[id] = edge_index

    print(len(protein_edge_index))
    print('错误seq有：{}'.format(seq_error_list))
    print(len(seq_error_list))
    np.save('data/protein/protein_edge_index.npy', protein_edge_index)
    # print(error_list)
