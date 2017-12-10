import argparse

import torch
from treehopper.data.vocab import Vocab
from treehopper.predict import load_best_model

from treehopper.data.dataset import SSTDataset
from treehopper import train


def eval(args):
    test_dataset = SSTDataset(args.input, Vocab(filename=args.vocab), num_classes=3)
    trainer_instance = load_best_model(args.model_path, args)
    loss, accuracies, outputs, output_trees = trainer_instance.test(test_dataset)
    test_acc = torch.mean(accuracies)
    return test_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch TreeLSTM for Sentiment Analysis Trees - Evaluation')
    parser.add_argument('--model_path', help='Path to saved model', required=True)
    parser.add_argument('--input', help='Path to input directory', default="test")
    parser.add_argument('--vocab', help='Path to vocabulary', default="tmp/vocab.txt")
    args = train.set_arguments({}, parser)
    print("Accuracy {}".format(eval(args)))