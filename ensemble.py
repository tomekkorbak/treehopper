from datetime import datetime

from sklearn.metrics import accuracy_score
from torch import nn

import sentiment
from sklearn.model_selection import ParameterGrid

from dataset import SSTDataset
from sentiment_trainer import SentimentTrainer
from split_datasets import split_dataset_simple
from training import train, torch
from vocab import Vocab, build_vocab
import numpy as np


def test_acc(all_outputs,test_dataset, type):
    ensemble_output = []
    all_outputs = zip(*all_outputs)
    for elem in all_outputs:
        elem = np.asarray(elem)
        if type == "vote":
            max_elems = np.argmax(elem, axis=2)
            output = np.argmax(np.bincount(np.reshape(max_elems, len(max_elems))))
        else:
            avg_elems = np.mean(elem, axis=1)
            output = np.argmax(avg_elems)
        ensemble_output.append(output)
    return accuracy_score(test_dataset.labels, ensemble_output)


def ensemble_train():
    type = "avg"
    train = True

    args = sentiment.set_arguments({})
    models_filenames = ["models/saved_model14_model_20170731_2027.pth",
                        "models/saved_model14_model_20170731_2209.pth"
                        ]
    train_dir = 'training-treebank'
    vocab_file = 'vocab.txt'
    build_vocab([
        'training-treebank/rev_sentence.txt',
        'training-treebank/sklad_sentence.txt'
    ], 'vocab.txt')
    vocab = Vocab(filename=vocab_file)
    dataset = SSTDataset(train_dir, vocab, args.num_classes)

    train_dataset, dev_dataset = SSTDataset(num_classes=args.num_classes), SSTDataset(num_classes=args.num_classes)
    test_dataset = SSTDataset(num_classes=args.num_classes)

    train_dataset, dev_dataset, test_dataset = split_dataset_simple(dataset, train_dataset, dev_dataset,
                                                                    test_dataset, args.split[1])

    if train:
        models = train_and_load_models(train_dataset, dev_dataset, vocab)
    else:
        models = load_best_models(models_filenames)

    all_outputs = []
    all_trees = []
    all_dev_outputs = []
    for model in models:
        _, _, test_output, test_trees = model.test(test_dataset)
        all_outputs.append(test_output)
        all_trees.append(test_trees)

        # _, _, dev_output, dev_trees = model.test(dev_dataset)
        # all_dev_outputs.append(dev_output)

    all_trees = zip(*all_trees)
    accuracies  = []
    for i in all_trees:
        accuracies.append(compute_accuracy_for_ensemble(i))
    print(np.mean(np.asarray(accuracies)))
    # accuracy = test_acc(all_outputs,test_dataset, type)
    # print("Test")
    # print(accuracy)
    #
    # accuracy = test_acc(all_dev_outputs,dev_dataset, type)
    # print("Dev")
    # print(accuracy)

def compute_accuracy_for_ensemble(list_trees):

    def _compute_accuracy(list_trees, accuracies=None):
        """
        Recursively compute accuracies for every subtree
        """
        if accuracies is None:
            accuracies = []


        out = np.asarray([np.reshape(x.output.data.numpy(),3) for x in list_trees])
        avg_elems = np.mean(out, axis=0)
        output = np.argmax(avg_elems)
        accuracies.append(1 if output == list_trees[0].gold_label else 0)
        subtrees = zip(*[x.children for x in list_trees])
        for subtree in subtrees:
            _compute_accuracy(subtree, accuracies)
        return accuracies

    total_accuracies = _compute_accuracy(list_trees)
    return np.mean(total_accuracies)

def load_best_models(models_filenames):
    models = []
    for model_filename in models_filenames:
        model = torch.load(model_filename)
        emb = torch.load(model_filename.replace("model_", "embedding_"))
        trainer = SentimentTrainer(None, model, emb,
                                   criterion=nn.NLLLoss(), optimizer=None)
        models.append(trainer)
    return models


def train_and_load_models(train_dataset, dev_dataset, vocab):
    models = {}

    # models["embeddings"] = [("data/pol/orth", "w2v_allwiki_nkjp300_300"),
    #                             ("data/pol/lemma", "w2v_allwiki_nkjp300_300"),
    #                             ("data/pol/fasttext", "wiki.pl")]

    models["embeddings"] = [
        ("data/pol/orth", "w2v_allwiki_nkjp300_50"),
        ("data/pol/lemma", "w2v_allwiki_nkjpfull_50"),
        ("data/pol/fasttext", "wiki.aa")]

    models = ParameterGrid(models)

    loaded_models = []
    for params in models:
        args = sentiment.set_arguments(params)

        max_dev_epoch, max_dev, max_model_filename = train(train_dataset, dev_dataset, vocab, args)
        model = torch.load(max_model_filename)
        emb = torch.load(max_model_filename.replace("model_", "embedding_"))
        trainer = SentimentTrainer(None, model, emb,
                                   criterion=nn.NLLLoss(), optimizer=None)
        loaded_models.append(trainer)
    return loaded_models

def all_same(items):
    return all(x == items[0] for x in items)

if __name__ == "__main__":
    ensemble_train()
