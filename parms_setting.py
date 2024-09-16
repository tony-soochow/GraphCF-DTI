import argparse


def settings():
    parser = argparse.ArgumentParser()

    # public parameters
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed. Default is 0.')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')

    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')

    parser.add_argument('--workers', type=int, default=0,
                        help='Number of parallel workers. Default is 0.')

    parser.add_argument('--in_file', choices=['drugbank', 'drugcentral', 'human', 'celegans'], required=True,
                        help='Path to data fold. e.g., data/DDI.edgelist')

    parser.add_argument('--out_file', required=True,
                        help='Path to data result file. e.g., result.txt')

    parser.add_argument('--feature_type', type=str, default='uniform', choices=['one_hot', 'uniform', 'normal', 'position'],
                        help='Initial node feature type. Default is position.')

    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Initial learning rate. Default is 5e-4.')

    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate (1 - keep probability). Default is 0.5.')

    parser.add_argument('--weight_decay', default= 0.0001,
                        help='Weight decay (L2 loss on parameters) Default is 5e-4.')

    parser.add_argument('--batch', type=int, default=128,
                        help='Batch size. Default is 256.')
    parser.add_argument('--num_heads', type=int, default=1, help='Number of the attention heads. Default is 8.')

    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs to train. Default is 30.')

    parser.add_argument('--network_ratio', type=float, default=1,
                        help='Remain links in network. Default is 1')

    parser.add_argument('--loss_ratio1', type=float, default=0.8,
                        help='Ratio of task1. Default is 1')

    parser.add_argument('--loss_ratio2', type=float, default=0.1,
                        help='Ratio of task2. Default is 0.1')

    parser.add_argument('--loss_ratio3', type=float, default=0.1,
                        help='Ratio of task3. Default is 0.1')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='dimensions of feature. Default is 128.')

    parser.add_argument('--hidden1', default=64,
                        help='Number of hidden units for encoding layer 1 for CSGNN. Default is 64.')

    args = parser.parse_args()

    return args
