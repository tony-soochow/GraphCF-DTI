from rdkit import Chem
import torch
import numpy as np
from rdkit.Chem import AllChem
import pandas as pd

df = pd.read_csv("pre_file/drugcentral_1.tsv", sep=' ')

# 获取"SMILES"列数据
data = df[['drug_id', 'SMIELS']]

# unique_smiles_data = smiles_data.unique()

smiles_data = dict()

for index, row in data.iterrows():
    drug_id = row['drug_id']
    smiles = row['SMIELS']
    if drug_id in smiles_data:
        continue
    smiles_data[drug_id] = smiles
print(len(smiles_data))


num = 0
atom_dis = {}
atom_pos = {}
drug_edge_index = {}

for id, smiles in smiles_data.items():
    mol = Chem.MolFromSmiles(smiles)
    edge_index = []
    bonds = mol.GetBonds()
    for bond in bonds:
        begin_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()
        edge_index.append([begin_atom_idx, end_atom_idx])

    if len(edge_index) == 0:
        atom_dis[id] = ''
        atom_pos[id] = ''
        drug_edge_index[id] = ''
        continue

    edge_index = torch.tensor(edge_index).t().contiguous()
    drug_edge_index[id] = edge_index.numpy()
    edge_index = edge_index.to('cuda')

    if AllChem.EmbedMolecule(mol) >= 0:
        pos = mol.GetConformer(0).GetPositions()
    else:
        num_atoms = mol.GetNumAtoms()
        pos = np.zeros((num_atoms, 3), dtype=np.float32)
    pos = torch.tensor(pos, dtype=torch.float32)
    pos = pos.to('cuda')

    row = edge_index[0]
    col = edge_index[1]
    sent_pos = pos[row]
    received_pos = pos[col]
    length = (sent_pos - received_pos).norm(dim=-1).unsqueeze(-1)
    data_numpy = length.cpu().detach().numpy()
    atom_dis[id] = data_numpy

    pos = pos.cpu().detach().numpy()
    atom_pos[id] = pos
    num += 1
print(len(atom_dis))
np.save('drug_data/drugcentral/atom_pos.npy', atom_pos)
np.save('drug_data/drugcentral/atom_dis.npy', atom_dis)
np.save('drug_data/drugcentral/drug_edge_index.npy', drug_edge_index)
print('有效文件为：{}'.format(num))

