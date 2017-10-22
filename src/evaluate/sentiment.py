import re

import numpy as np
import torch
from sklearn.model_selection import KFold
from src.config import parse_args
from src.data.dataset import SSTDataset
from src.data.split_datasets import split_dataset_simple, split_dataset_random, split_dataset_kfold
from src.data.vocab import build_vocab, Vocab
from src.models.training import train


def set_arguments(grid_args):
    args = parse_args()
    if grid_args !=None:
        if "embeddings" in grid_args:
            args.emb_dir = grid_args["embeddings"][0]
            args.emb_file = grid_args["embeddings"][1]
        for key, val in grid_args.items():
            setattr(args,key,val)
        args.calculate_new_words = True

    embedding_dim = "((\d+)d$)|((\d+)$)"
    dim_from_file = re.search(embedding_dim, args.emb_file)
    args.input_dim = int(dim_from_file.group(0)) if dim_from_file else 300
    args.num_classes = 3  # -1 0 1

    args.cuda = args.cuda and torch.cuda.is_available()

    args.split = ('simple', 0.1) if args.folds == 1 else ('kfold', args.folds)
    args.test = True #
    #("simple",(dev_size,test_size)),("random",size_of_dev),("kfold", number_of_folds)
    print(args)
    return args


def create_train_dataset(args):
    train_dir = 'training-treebank'
    vocab_file = 'tmp/vocab.txt'
    build_vocab([
        'training-treebank/rev_sentence.txt',
        'training-treebank/sklad_sentence.txt',
        'test/polevaltest_sentence.txt'
    ], 'tmp/vocab.txt')
    vocab = Vocab(filename=vocab_file)
    full_dataset = SSTDataset(train_dir, vocab, args.num_classes)
    return vocab, full_dataset


def main(grid_args = None):
    args = set_arguments(grid_args)
    vocab, full_dataset = create_train_dataset(args)

    if args.test:
        test_dir = 'test'
        test_dataset = SSTDataset(test_dir, vocab, args.num_classes)
        max_dev_epoch, max_dev, _ = train(full_dataset, test_dataset, vocab, args)
    else:

        train_dataset = SSTDataset(num_classes=args.num_classes)
        dev_dataset   = SSTDataset(num_classes=args.num_classes)

        if args.split[0] == "simple":
            train_dataset, dev_dataset = split_dataset_simple(
                full_dataset,
                train_dataset,
                dev_dataset,
                split=args.split[1]
            )
            max_dev_epoch, max_dev, _ = train(train_dataset, dev_dataset, vocab, args)
        elif args.split[0] == "random":
            train_dataset, dev_dataset = split_dataset_random(
                full_dataset,
                train_dataset,
                dev_dataset,
                test_size=args.split[1]
            )
            max_dev_epoch, max_dev, _ = train(train_dataset, dev_dataset, vocab, args)
        else:
            all_dev_epoch, all_dev = kfold_training(
                full_dataset,
                vocab,
                args.split[1],
                train_dataset,
                dev_dataset,
                args
            )
            max_dev_epoch, max_dev = np.mean(all_dev_epoch), np.mean(all_dev)

    with open(args.name + '_results', 'a') as result_file:
        result_file.write(str(args) + '\nEpoch {epoch}, accuracy {acc:.4f}\n'.format(
            epoch=max_dev_epoch,
            acc=max_dev
        ))
    return max_dev_epoch, max_dev


def kfold_training(dataset,vocab, split, train_dataset, dev_dataset,args):
    kf = KFold(n_splits=split)
    X = np.array([(x, y) for x, y in zip(dataset.trees, dataset.sentences)])
    y = np.array(dataset.labels)
    max_dev_epoch, max_dev = [], []
    for train_index, test_index in kf.split(X):
        train_dataset, dev_dataset = split_dataset_kfold(X, y,
                                                         train_index,
                                                         test_index,
                                                         train_dataset,
                                                         dev_dataset)
        dev_epoch, dev, _ = train(train_dataset, dev_dataset, vocab, args)
        max_dev_epoch.append(dev_epoch), max_dev.append(dev)
    return max_dev_epoch, max_dev

if __name__ == "__main__":
    main()
