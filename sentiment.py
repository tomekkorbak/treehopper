from __future__ import print_function

import os
import numpy as np
from sklearn.model_selection import KFold

from model import *
from split_datasets import split_dataset_simple, split_dataset_random, split_dataset_kfold
from trainer import train
from vocab import Vocab
from dataset import SSTDataset
from utils import build_vocab
from config import parse_args


def set_arguments():
    args = parse_args(type=1)
    args.input_dim = 25
    args.mem_dim = 168
    args.num_classes = 3  # -1 0 1
    args.cuda = args.cuda and torch.cuda.is_available()
    args.split = ("kfold",10) #("simple",size_of_train),("random",size_of_dev),("kfold", number_of_folds)
    print(args)
    return args


def main():
    args = set_arguments()

    train_dir = 'training-treebank'
    token_files = os.path.join(train_dir, 'sents.toks')
    vocab_file = 'vocab.txt'
    build_vocab([
        'training-treebank/rev_sentence.txt',
        'training-treebank/sklad_sentence.txt'
    ], 'vocab.txt')
    vocab = Vocab(filename=vocab_file)
    dataset = SSTDataset(train_dir, vocab, args.num_classes,
                               args.fine_grain, args.model_name)

    train_dataset, dev_dataset = SSTDataset(num_classes=args.num_classes), SSTDataset(num_classes=args.num_classes)

    if(args.split[0] == "simple"):
        train_dataset, dev_dataset = split_dataset_simple(dataset,train_dataset,dev_dataset,args.split[1])
        max_dev_epoch, max_dev = train(train_dataset, dev_dataset, vocab,args)
    elif(args.split[0] == "random"):
        train_dataset, dev_dataset = split_dataset_random(dataset,train_dataset,dev_dataset,args.split[1])
        max_dev_epoch, max_dev = train(train_dataset, dev_dataset, vocab, args)
    else:
        all_dev_epoch, all_dev =  kfold_training(dataset, vocab,args.split[1], train_dataset, dev_dataset,args)
        max_dev_epoch, max_dev=np.mean(all_dev_epoch), np.mean(all_dev)

    with open("results.csv", "a") as myfile:
        myfile.write(str(args.input_dim)+","+args.split[0]+","+ str(max_dev_epoch)+","+str(max_dev)+"\n")

def kfold_training(dataset,vocab, split, train_dataset, dev_dataset,args):
    kf = KFold(n_splits=split)
    X = np.array([(x, y) for x, y in zip(dataset.trees, dataset.sentences)])
    y = np.array(dataset.labels)
    max_dev_epoch, max_dev = [], []
    for train_index, test_index in kf.split(X):
        train_dataset, dev_dataset = split_dataset_kfold(X, y, train_index, test_index, train_dataset, dev_dataset)
        dev_epoch, dev = train(train_dataset, dev_dataset, vocab,args)
        max_dev_epoch.append(dev_epoch), max_dev.append(dev)
    return max_dev_epoch, max_dev

if __name__ == "__main__":
    main()