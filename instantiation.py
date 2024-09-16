from torch.optim import Adam
# from layer import *
from gnn_layer import *

def Create_model(args):

    model = GraphCF(args)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    return model, optimizer