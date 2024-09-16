import networkx as nx
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
from utils import *


class Data_class(Dataset):

    def __init__(self, triple):
        self.entity1 = triple[:, 0]
        self.entity2 = triple[:, 1]
        self.label = triple[:, 2]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):

        return self.label[index], (self.entity1[index], self.entity2[index])


def load_data(args, val_ratio=0.1, test_ratio=0.1):
    """Read data from path, convert data into loader, return features and symmetric adjacency"""
    # read data
    print('Loading {0} seed{1} dataset...'.format(args.in_file, args.seed))

    edgelist_file = 'data/{}.edgelist'.format(args.in_file)
    
    data = np.loadtxt(edgelist_file, dtype=np.int64)
    positive = data[data[:, 2] == 1]
    negative = data[data[:, 2] == 0]
    unique_entity = len(positive)
    np.random.seed(args.seed)
    np.random.shuffle(positive)
    np.random.shuffle(negative)

    print("positve examples: %d, negative examples: %d." % (positive.shape[0], negative.shape[0]))
    positive_g = positive[:,:2]
    G = nx.Graph()
    G.add_edges_from(positive_g)
    print(nx.info(G))
    # split data
    val_size = int(val_ratio * positive.shape[0])
    test_size = int(test_ratio * positive.shape[0])

    positive = np.concatenate([positive, np.ones(positive.shape[0], dtype=np.int64).reshape(positive.shape[0], 1)], axis=1)
    negative = np.concatenate([negative, np.zeros(negative.shape[0], dtype=np.int64).reshape(negative.shape[0], 1)], axis=1)

    train_data = np.vstack((positive[: -(val_size + test_size)], negative[: -(val_size + test_size)]))
    val_data = np.vstack((positive[-(val_size + test_size): -test_size], negative[-(val_size + test_size): -test_size]))
    test_data = np.vstack((positive[-test_size:], negative[-test_size:]))
    
    # construct adjacency
    train_positive = positive[: -(val_size + test_size)]
    adj = sp.coo_matrix((np.ones(train_positive.shape[0]), (train_positive[:, 0], train_positive[:, 1])),       # (7343, 7343)      稀疏邻接矩阵例：(2727, 2199)	1.0
                        shape=(unique_entity, unique_entity), dtype=np.float32)

    # symmetrization
    adj_o = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # build 2-hop adjacency
    adj_s = adj.dot(adj)        # 计算邻接矩阵的平方
    adj_s = adj_s.sign()        # 正数变为1 负数变为-1 0不变？？？

    # construct edges
    edges_o = adj_o.nonzero()   # 获取对称化之后非零元素的索引
    edge_index_o = torch.tensor(np.vstack((edges_o[0], edges_o[1])), dtype=torch.long)

    edges_s = adj_s.nonzero()   # 获取2-hop邻接矩阵非零元素的索引
    edge_index_s = torch.tensor(np.vstack((edges_s[0], edges_s[1])), dtype=torch.long)

    # build data loader
    params = {'batch_size': args.batch, 'shuffle': True, 'num_workers': args.workers, 'drop_last': True}

    training_set = Data_class(train_data)
    train_loader = DataLoader(training_set, **params)

    validation_set = Data_class(val_data)
    val_loader = DataLoader(validation_set, **params)

    test_set = Data_class(test_data)
    test_loader = DataLoader(test_set, **params)

    # extract features
    print('Extracting features...')
    if args.feature_type == 'one_hot':
        features = np.eye(unique_entity)    # (7343,7343)

    elif args.feature_type == 'uniform':
        np.random.seed(args.seed)
        features = np.random.uniform(low=0, high=1, size=(unique_entity, args.dimensions))  # [0,1)之间的随机均匀分布的特征矩阵   (7343, 128)

    elif args.feature_type == 'normal':
        np.random.seed(args.seed)
        features = np.random.normal(loc=0, scale=1, size=(unique_entity, args.dimensions))  # 均值为0，标准差为1的随机正态分布的特征矩阵    (7343, 128)

    elif args.feature_type == 'position':
        features = adj_o.todense()      # 转换为稠密矩阵，空缺值为0 (7343, 7343)

    features_o = normalize(features)    # 将特征进行归一化处理

    args.dimensions = features_o.shape[1]
    print('dimension:{}'.format(features_o.shape[1]))
    
    # adversarial nodes
    np.random.seed(args.seed)
    id = np.arange(features_o.shape[0])
    id = np.random.permutation(id)
    features_a = features_o[id]


    x_o = torch.tensor(features_o, dtype=torch.float)
    data_o = Data(x=x_o, edge_index=edge_index_o)       # data_o包含正常节点特征矩阵和边索引

    data_s = Data(edge_index=edge_index_s)          # data_s包含2-hop的边索引

    x_a = torch.tensor(features_a, dtype=torch.float)   
    y_a = torch.cat((torch.ones(adj.shape[0], 1), torch.zeros(adj.shape[0], 1)), dim=1)     # 生成对抗标签torch.Size([7343, 2])  1.0 0.0 
    data_a = Data(x=x_a, y=y_a)             # data_a包含对抗节点特征矩阵和边索引

    print('Loading  data finished!')
    return data_o, data_s, data_a, train_loader, val_loader, test_loader


from seq_words import *


def get_smiles_seq(args):

    file_name = args.in_file
        
    atom_pos_file = 'data/drug_data/{}/atom_pos.npy'.format(file_name)
    atom_dis_file = 'data/drug_data/{}/atom_dis.npy'.format(file_name)
    drug_edge_index_file = 'data/drug_data/{}/drug_edge_index.npy'.format(file_name)
    
    protein_feature_file = 'data/protein_data/{}/protein_feature.npy'.format(file_name)
    protein_edge_index_file = 'data/protein_data/{}/protein_edge_index.npy'.format(file_name)
    
    
    pos_dict = np.load(atom_pos_file, allow_pickle=True)
    dis_dict = np.load(atom_dis_file, allow_pickle=True)
    drug_edge_index = np.load(drug_edge_index_file, allow_pickle=True)
    
    protein_node_feature = np.load(protein_feature_file, allow_pickle=True)
    protein_edge_index = np.load(protein_edge_index_file, allow_pickle=True)
    
    dti_file = 'data/{}.tsv'.format(args.in_file)
    smiles2vec = dict()
    seq2vec = dict()
    with open(dti_file, 'r') as f:
        for line in f:
            l = line.strip().split()
            drug_id = int(l[4])
            target_id = int(l[1])
            seq = l[2]
            smiles = l[5]
            smiles2vec[drug_id] = label_smiles(smiles, CHARISOSMISET, 100)
            seq2vec[target_id] = label_sequence(seq, CHARPROTSET, 1000)

    return smiles2vec, seq2vec, [pos_dict, dis_dict, drug_edge_index], [protein_node_feature, protein_edge_index]