import argparse
from datetime import datetime


def parse_args():
        parser = argparse.ArgumentParser(description='PyTorch TreeLSTM for Sentiment Analysis Trees')
        parser.add_argument('--name',
                            default='{date:%Y%m%d_%H%M}'.format(
                                date=datetime.now()),
                            help='name for log and saved models')
        parser.add_argument('--saved', default='models/saved_model',
                            help='name for log and saved models')
        parser.add_argument('--data', default='training-treebank',
                            help='path to dataset')
        parser.add_argument('--emb_dir', default='resources/pol/fasttext/',
                            help='directory with embeddings')
        parser.add_argument('--emb_file', default='wiki.aa',
                            help='file with embeddings')
        parser.add_argument('--batchsize', default=25, type=int,
                            help='batchsize for optimizer updates')
        parser.add_argument('--epochs', default=25, type=int,
                            help='number of total epochs to run')
        parser.add_argument('--mem_dim', default=300, type=int,
                            help='size of LSTM hidden state')
        parser.add_argument('--recurrent_dropout_c', default=0.15, type=float,
                            help='probability of recurrent dropout for cell state')
        parser.add_argument('--recurrent_dropout_h', default=0.15, type=float,
                            help='probability of recurrent dropout for hidden state')
        parser.add_argument('--zoneout_choose_child', default=False, type=bool,
                            help='tba')
        parser.add_argument('--common_mask', default=False, type=bool,
                            help='tba')
        parser.add_argument('--lr', default=0.05, type=float,
                            metavar='LR', help='initial learning rate')
        parser.add_argument('--emblr', default=0.1, type=float,
                            metavar='EMLR', help='initial embedding learning rate')
        parser.add_argument('--wd', default=1e-4, type=float,
                            help='weight decay (default: 1e-4)')
        parser.add_argument('--reg', default=1e-4, type=float,
                            help='l2 regularization (default: 1e-4)')
        parser.add_argument('--optim', default='adagrad',
                            help='optimizer (default: adagrad)')
        parser.add_argument('--seed', default=123, type=int,
                            help='random seed (default: 123)')
        parser.add_argument('--reweight', default=False, type=bool,
                            help='reweight loss per class to the distrubition '
                                 'of classess in the public dataset')
        parser.add_argument('--folds', default=1, type=int,
                            help='Number of folds for k-fold cross validation '
                                 '(default: 1; this corresponds to simple '
                                 'validation).')
        #Arguments necessary to determine what program should do
        parser.add_argument('--train', dest='train', action='store_true', help='Train new model')
        parser.add_argument('--predict', '-p', type=str, help='Tagging test file \'--predict output_file_path\' ')
        parser.set_defaults()

        cuda_parser = parser.add_mutually_exclusive_group(required=False)
        cuda_parser.add_argument('--cuda', dest='cuda', action='store_true')
        cuda_parser.add_argument('--no-cuda', dest='cuda', action='store_false')
        parser.set_defaults(cuda=False, train=False)

        args = parser.parse_args()
        return args
