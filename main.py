import torch
import numpy as np

from parms_setting import settings
from data_preprocess import load_data, get_smiles_seq
from instantiation import Create_model
from train import train_model


# parameters setting
args = settings()

args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# load data
data_o, data_s, data_a, train_loader, val_loader, test_loader = load_data(args)

# load smiles_protein
smiles2vec, seq2vec, drug_H_data, protein_H_data = get_smiles_seq(args)

# train and test model

model, optimizer = Create_model(args)

print('Training...')
train_model(model, optimizer, data_o, data_s, data_a, train_loader, val_loader, test_loader, args, smiles2vec, seq2vec, drug_H_data, protein_H_data)
print('done....')
