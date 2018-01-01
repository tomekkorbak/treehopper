import re

import numpy as np
import torch
from sklearn.model_selection import KFold
from config import set_arguments
from data.split_datasets import split_dataset_kfold, split_dataset_simple
from data.vocab import build_vocab, Vocab
from model.training import train

from data.dataset import SSTDataset



def create_full_dataset(args):
    train_dir = 'training-treebank'
    vocab_file = 'tmp/vocab.txt'
    build_vocab([
        'training-treebank/rev_sentence.txt',
        'training-treebank/sklad_sentence.txt',
        'test/polevaltest_sentence.txt',
        args.emb_dir+args.emb_file+'.vec' #full vocabulary in model
    ], 'tmp/vocab.txt')
    vocab = Vocab(filename=vocab_file)
    full_dataset = SSTDataset(train_dir, vocab, args.num_classes)
    return vocab, full_dataset


def main(grid_args = None):
    args = set_arguments(grid_args)
    vocab, full_dataset = create_full_dataset(args)

    if args.test:
        test_dir = 'test'
        test_dataset = SSTDataset(test_dir, vocab, args.num_classes)
        max_dev_epoch, max_dev_acc, max_model_filename = train(full_dataset, test_dataset, vocab, args)
    else:

        train_dataset = SSTDataset(num_classes=args.num_classes)
        dev_dataset   = SSTDataset(num_classes=args.num_classes)

        train_dataset, dev_dataset = split_dataset_simple(
            full_dataset,
            train_dataset,
            dev_dataset,
            split=args.split
        )
        max_dev_epoch, max_dev_acc, max_model_filename = train(train_dataset, dev_dataset, vocab, args)

    with open(args.name + '_results', 'a') as result_file:
        result_file.write(str(args) + '\nEpoch {epoch}, accuracy {acc:.4f}\n'.format(
            epoch=max_dev_epoch,
            acc=max_dev_acc
        ))
    return max_dev_epoch, max_dev_acc, max_model_filename

if __name__ == "__main__":
    main()