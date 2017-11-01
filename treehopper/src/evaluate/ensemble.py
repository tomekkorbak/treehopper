from datetime import datetime

import numpy as np
from sklearn.model_selection import ParameterGrid
from treehopper.src.data.split_datasets import split_dataset_simple
from treehopper.src.data.vocab import Vocab, build_vocab
from treehopper.src.evaluate import sentiment
from treehopper.src.model.sentiment_trainer import SentimentTrainer
from treehopper.src.model.training import train, torch
from torch import nn

from treehopper.src.data.dataset import SSTDataset


def test_ensemble(models, test_dataset):
    all_outputs = []
    all_trees = []
    for model in models:
        _, _, test_output, test_trees = model.test(test_dataset)
        all_outputs.append(test_output)
        all_trees.append(test_trees)

    all_trees = zip(*all_trees)
    accuracies = []
    for i in all_trees:
        accuracies.append(compute_accuracy_for_ensemble(i))
    print(np.mean(np.asarray(accuracies)))

def get_output_as_string(trees):
    out = np.asarray([np.reshape(x.output.data.numpy(), 3) for x in trees])
    avg_elems = np.mean(out, axis=0)
    output = np.argmax(avg_elems) - 1
    return str(output)

def get_predicted_labels(trees):
    ordered_trees = []
    for tree in trees:
        assert tree.parent is None, 'This method should only be called on ' \
                                    'root nodes'
        ordered_trees.append(tree.list_children_in_order())
    ordered_trees = zip(*ordered_trees)
    outputs = []
    for trees in ordered_trees:
        outputs.append(get_output_as_string(trees))
    return outputs

    # return ' '.join(tree.
    #                 for tree in tree.list_children_in_order())


def predict_ensemble(models):

    train_dir = 'test'
    vocab_file = 'tmp/vocab_test.txt'
    build_vocab([
        'test/polevaltest_sentence.txt',
    ], 'tmp/vocab_test.txt')
    vocab = Vocab(filename=vocab_file)
    test_dataset = SSTDataset(train_dir, vocab, num_classes=3)

    all_trees = []
    for model in models:
        test_trees = model.predict(test_dataset)
        all_trees.append(test_trees)

    all_trees = zip(*all_trees)
    filename = 'finals/submission_ens.txt'.format(date=datetime.now())
    with open(filename, 'w') as submission_file:
        for sentence in all_trees:
            submission_file.write(' '.join(get_predicted_labels(sentence)) + '\n')
    print('Good luck!')

def ensemble_train():
    train = False

    args = sentiment.set_arguments({})
    models_filenames = ['finals/ensemble/10_model_.pth',
                        'finals/ensemble/15_model_.pth',
                        'finals/ensemble/19_model_.pth']
    train_dir = 'training-treebank'
    vocab_file = 'vocab.txt'
    build_vocab([
        'training-treebank/rev_sentence.txt',
        'training-treebank/sklad_sentence.txt'
    ], 'vocab.txt')
    vocab = Vocab(filename=vocab_file)
    full_dataset = SSTDataset(train_dir, vocab, args.num_classes)


    if args.create_test != 0:
        split_point = int(len(full_dataset) * (args.create_test))
        test_dataset = SSTDataset(num_classes=args.num_classes)
        test_dataset.trees, full_dataset.trees = full_dataset.trees[:split_point], full_dataset.trees[split_point:]
        test_dataset.sentences, full_dataset.sentences = full_dataset.sentences[:split_point], full_dataset.sentences[
                                                                                               split_point:]
        test_dataset.labels, full_dataset.labels = full_dataset.labels[:split_point], full_dataset.labels[split_point:]

    train_dataset, dev_dataset = SSTDataset(num_classes=args.num_classes), SSTDataset(num_classes=args.num_classes)

    train_dataset, dev_dataset = split_dataset_simple(full_dataset, train_dataset, dev_dataset,
                                                                    args.split[1])

    if train:
        models = train_and_load_models(train_dataset, dev_dataset, vocab, args)
    else:
        models = load_best_models(models_filenames, args)


    #test_ensemble(models, test_dataset)

    predict_ensemble(models)


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

def load_best_models(models_filenames, args):
    models = []
    for model_filename in models_filenames:
        model = torch.load(model_filename)
        emb = torch.load(model_filename.replace("model_", "embedding_"))
        trainer = SentimentTrainer(args, model, emb,
                                   criterion=nn.NLLLoss(), optimizer=None)
        models.append(trainer)
    return models


def train_and_load_models(train_dataset, dev_dataset, vocab,args):
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
        trainer = SentimentTrainer(args, model, emb,
                                   criterion=nn.NLLLoss(), optimizer=None)
        loaded_models.append(trainer)
    return loaded_models

def all_same(items):
    return all(x == items[0] for x in items)

if __name__ == "__main__":
    ensemble_train()
