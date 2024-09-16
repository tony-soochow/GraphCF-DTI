import pandas as pd
from protein_feature_H import *
import os

df = pd.read_csv("data/drugbank_aft.tsv", sep=' ')

# 获取数据
data = df[['protein_id', 'seq']]

seq_data = dict()

for index, row in data.iterrows():
    protein_id = row['protein_id']
    seq = row['seq']
    if protein_id in seq_data:
        continue
    seq_data[protein_id] = seq
print(len(seq_data))

protein_feature = dict()

protein_edge_index = np.load('data/protein/protein_edge_index.npy', allow_pickle=True)

num = 0
for id, seq in seq_data.items():
    if protein_edge_index.item().get(id) == '':
        protein_feature[id] = ''
        continue
    protein_node_feature = seq_feature(seq)
    protein_feature[id] = protein_node_feature
    num += 1
print(num)
print(len(protein_feature))
np.save('data/protein/protein_feature.npy', protein_feature)
